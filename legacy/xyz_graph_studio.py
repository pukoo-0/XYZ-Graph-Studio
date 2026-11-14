"""
XYZ → Graph Studio
- Bond rule: pairwise cutoff table (C/N/O/H) + fallback for all other pairs
- Light mode only
- Atom sizes vary (spheres)
- TRUE white 3D background (forced glClearColor)
- Bonds thicker via multi-line offsets (no cylinders)

Matrix / spectra:
- "Adjacency Matrix" shown/exported is identity-encoded:
    W_ij = 1 if bonded (i!=j), else 0
    W_ii = atomic number Z_i
- Eigenvalues tab: eigenvalues of W
- Energy tab (scalar): E(G) = Σ |λ_i|
- k-Moment tab (curve): μ_k(G) = Σ (λ_i^k), for k = 1..K (integer k)

Topological (Wikipedia Wiener index):
- Wiener index on the HYDROGEN-SUPPRESSED chemical graph:
    W(G) = Σ_{u<v} d(u,v)
  where d(u,v) is shortest-path distance in NUMBER OF BONDS (edges).
- Topological tab shows TWO graphs visually:
    (1) Full graph (with H)       [nonstandard for Wiener, shown for reference]
    (2) Hydrogen-suppressed graph  [THIS matches Wikipedia definition]
  and prints W for both, clearly labeled.
- Export Wiener CSV: saves W for both graphs.

Estrada index (adjacency spectrum):
- EE(G) = Σ exp(λ_i) = tr(exp(A))  where λ_i are eigenvalues of adjacency A.
- Node subgraph centrality: EE(v) = (exp(A))_vv.
- Export Estrada CSV: saves total EE and per-node subgraph centrality for both graphs.

Watermark:
- "Made by George" in status bar

Dependencies:
  python -m pip install pyside6 numpy networkx pyqtgraph pyopengl PyOpenGL PyOpenGL_accelerate
"""

import os
import sys
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import networkx as nx

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QSplitter, QDockWidget, QPlainTextEdit, QLabel, QToolBar, QStatusBar,
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import MeshData
from OpenGL import GL  # force white clear color


# ---------------- Atomic number <-> symbol ----------------
PERIODIC = [
    None,
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]
Z_TO_SYMBOL = {i: sym for i, sym in enumerate(PERIODIC) if sym is not None}
SYMBOL_TO_Z = {sym: i for i, sym in Z_TO_SYMBOL.items()}


def parse_atom_token(token: str) -> Tuple[str, int]:
    """Accept either element symbol ('O') OR atomic number ('8'). Returns (symbol, Z)."""
    t = token.strip()
    if t.isdigit():
        z = int(t)
        sym = Z_TO_SYMBOL.get(z, f"Z{z}")
        return sym, z
    sym = t[0].upper() + t[1:].lower() if len(t) > 1 else t.upper()
    z = SYMBOL_TO_Z.get(sym, 0)
    return sym, z


# ---------------- Bond thresholds ----------------
PAIR_CUTOFFS_ANGSTROM: Dict[Tuple[str, str], float] = {
    ("C", "C"): 1.55,
    ("C", "N"): 1.50,
    ("C", "O"): 1.45,
    ("C", "H"): 1.12,
    ("N", "N"): 1.40,
    ("N", "O"): 1.37,
    ("N", "H"): 1.00,
    ("O", "O"): 1.35,
    ("O", "H"): 0.97,
    ("H", "H"): 0.80,
}
FALLBACK_SCALE = 1.20

COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22,
    "Br": 1.20, "I": 1.39,
}
DEFAULT_RADIUS = 0.77

ELEMENT_COLORS = {
    "H": (0.95, 0.95, 0.95, 1.0),
    "C": (0.20, 0.20, 0.20, 1.0),
    "N": (0.20, 0.40, 0.95, 1.0),
    "O": (0.95, 0.20, 0.20, 1.0),
    "F": (0.10, 0.80, 0.10, 1.0),
    "Cl": (0.10, 0.80, 0.10, 1.0),
    "Br": (0.60, 0.20, 0.10, 1.0),
    "I": (0.50, 0.10, 0.70, 1.0),
    "S": (0.90, 0.80, 0.10, 1.0),
    "P": (0.95, 0.60, 0.20, 1.0),
}


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def cutoff_for_pair(a: str, b: str) -> float:
    key = _pair_key(a, b)
    if key in PAIR_CUTOFFS_ANGSTROM:
        return PAIR_CUTOFFS_ANGSTROM[key]
    ra = COVALENT_RADII.get(a, DEFAULT_RADIUS)
    rb = COVALENT_RADII.get(b, DEFAULT_RADIUS)
    return FALLBACK_SCALE * (ra + rb)


@dataclass
class Frame:
    comment: str
    elements: List[str]
    atomic_numbers: List[int]
    coords: np.ndarray  # (n,3)


def read_xyz_frames(path: str) -> List[Frame]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = [ln.rstrip("\n") for ln in f]
    lines = [ln.strip() for ln in raw if ln.strip() != ""]

    frames: List[Frame] = []
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i])
        except ValueError:
            raise ValueError(f"Parse error at line {i+1}: expected atom count.")
        if i + 1 >= len(lines):
            raise ValueError("Parse error: missing comment line.")
        comment = lines[i + 1]

        start = i + 2
        end = start + n
        if end > len(lines):
            raise ValueError("Parse error: incomplete atom block.")

        elems: List[str] = []
        Zs: List[int] = []
        coords = np.zeros((n, 3), dtype=float)
        for k, ln in enumerate(lines[start:end]):
            parts = ln.split()
            if len(parts) < 4:
                raise ValueError(f"Parse error at line {start+k+1}: expected 'El x y z'.")
            sym, Z = parse_atom_token(parts[0])
            x, y, zc = float(parts[1]), float(parts[2]), float(parts[3])
            elems.append(sym)
            Zs.append(Z)
            coords[k] = (x, y, zc)

        frames.append(Frame(comment=comment, elements=elems, atomic_numbers=Zs, coords=coords))
        i = end

    if not frames:
        raise ValueError("No frames found.")
    return frames


def infer_bonds_cutoff_table(elements: List[str], coords: np.ndarray) -> List[Tuple[int, int]]:
    n = len(elements)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = cutoff_for_pair(elements[i], elements[j])
            if dist[i, j] <= cutoff and dist[i, j] > 1e-9:
                bonds.append((i, j))
    return bonds


def build_graph(elements: List[str], atomic_numbers: List[int], coords: np.ndarray, bonds: List[Tuple[int, int]]) -> nx.Graph:
    """
    Molecular graph with node attributes:
      element, Z, x,y,z
    """
    G = nx.Graph()
    for idx, (el, Z, (x, y, z)) in enumerate(zip(elements, atomic_numbers, coords)):
        G.add_node(idx, element=el, Z=int(Z), x=float(x), y=float(y), z=float(z))
    for i, j in bonds:
        G.add_edge(i, j)
    return G


def hydrogen_suppressed_graph(G: nx.Graph, elements: List[str]) -> nx.Graph:
    H_nodes = [i for i, el in enumerate(elements) if el == "H"]
    G2 = G.copy()
    G2.remove_nodes_from(H_nodes)
    return G2


def wiener_index_sum_components(G: nx.Graph) -> float:
    """
    Standard (unweighted) Wiener index:
      W(G) = Σ_{u<v} d(u,v)
    where d(u,v) is shortest-path distance in number of bonds (edges).
    For disconnected graphs, sums within each connected component.
    """
    if G.number_of_nodes() <= 1:
        return 0.0

    total = 0.0
    for nodes in nx.connected_components(G):
        Gc = G.subgraph(nodes)
        if Gc.number_of_nodes() <= 1:
            continue

        lengths = dict(nx.all_pairs_shortest_path_length(Gc))
        vs = list(Gc.nodes())

        s = 0.0
        for i in range(len(vs)):
            u = vs[i]
            for j in range(i + 1, len(vs)):
                v = vs[j]
                s += float(lengths[u][v])
        total += s

    return float(total)


# ---------------- Estrada Index ----------------
def estrada_index_and_node_centrality_from_adj(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estrada index:
      EE(G) = Σ exp(λ_i) = tr(exp(A))
    Node subgraph centrality:
      EE(v) = (exp(A))_vv = Σ_j (phi_j(v)^2 * exp(λ_j))
    Computed via symmetric eigendecomposition (A is symmetric for undirected graphs).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return 0.0, np.zeros((0,), dtype=float)

    vals, vecs = np.linalg.eigh(A)  # symmetric
    expvals = np.exp(vals)

    EE = float(expvals.sum())
    node_EE = (vecs * vecs) @ expvals  # diag(exp(A))
    return EE, np.asarray(node_EE, dtype=float)


def graph_energy(eigs: np.ndarray) -> float:
    """E(G) = Σ |λ_i|"""
    if eigs.size == 0:
        return 0.0
    return float(np.sum(np.abs(eigs)))


def k_moments(eigs: np.ndarray, K: int = 30) -> np.ndarray:
    """μ_k(G) = Σ λ_i^k, for k=1..K integer."""
    if eigs.size == 0 or K <= 0:
        return np.zeros((0,), dtype=float)
    ks = np.arange(1, int(K) + 1, dtype=int)
    return (eigs[:, None] ** ks[None, :]).sum(axis=0)


# ---------------- UI models ----------------
class AtomTableModel(QAbstractTableModel):
    HEADERS = ["#", "Element", "x", "y", "z"]

    def __init__(self):
        super().__init__()
        self.elements: List[str] = []
        self.coords = np.zeros((0, 3), dtype=float)

    def set_data(self, elements: List[str], coords: np.ndarray):
        self.beginResetModel()
        self.elements = list(elements)
        self.coords = np.array(coords, dtype=float)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.elements)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 5

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        return str(section)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role == Qt.DisplayRole:
            if c == 0:
                return str(r)
            if c == 1:
                return self.elements[r]
            val = self.coords[r, c - 2]
            return f"{val:.6f}"
        return None


class WhiteGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opts["bgcolor"] = (1.0, 1.0, 1.0, 1.0)

    def initializeGL(self):
        super().initializeGL()
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)

    def paintGL(self, *args, **kwargs):
        self.opts["bgcolor"] = (1.0, 1.0, 1.0, 1.0)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        super().paintGL(*args, **kwargs)


# ---------------- Main window ----------------
class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XYZ → Graph Studio")
        self.resize(1450, 850)
        self.setAcceptDrops(True)

        self.frames: List[Frame] = []
        self.current_frame = 0
        self.current_path: Optional[str] = None

        self._sphere_mesh = MeshData.sphere(rows=16, cols=32)
        self._atom_mesh_items: List[gl.GLMeshItem] = []
        self._bond_line_items: List[gl.GLLinePlotItem] = []

        self._last_W: Optional[np.ndarray] = None
        self._last_eigs: Optional[np.ndarray] = None
        self._last_mu: Optional[np.ndarray] = None
        self._last_energy: Optional[float] = None

        # Wiener cached (for export)
        self._last_wiener_full: Optional[float] = None
        self._last_wiener_heavy: Optional[float] = None

        # Estrada cached (for export / display)
        self._last_EE_full: Optional[float] = None
        self._last_EE_heavy: Optional[float] = None
        self._last_EE_nodes_full: Optional[np.ndarray] = None
        self._last_EE_nodes_heavy: Optional[np.ndarray] = None

        self._build_ui()
        self._apply_light_theme()
        self._update_views()

    def _build_ui(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("Open", self)
        act_open.triggered.connect(self.open_xyz)
        tb.addAction(act_open)

        self.frame_combo = pg.QtWidgets.QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        tb.addWidget(QLabel("   Frame: "))
        tb.addWidget(self.frame_combo)

        act_export_adj = QAction("Export Adjacency CSV", self)
        act_export_adj.triggered.connect(self.export_adjacency_csv)
        tb.addAction(act_export_adj)

        act_export_eigs = QAction("Export Eigenvalues CSV", self)
        act_export_eigs.triggered.connect(self.export_eigenvalues_csv)
        tb.addAction(act_export_eigs)

        act_export_mu = QAction("Export k-Moment CSV", self)
        act_export_mu.triggered.connect(self.export_k_moment_csv)
        tb.addAction(act_export_mu)

        act_export_wiener = QAction("Export Wiener CSV", self)
        act_export_wiener.triggered.connect(self.export_wiener_csv)
        tb.addAction(act_export_wiener)

        act_export_estrada = QAction("Export Estrada CSV", self)
        act_export_estrada.triggered.connect(self.export_estrada_csv)
        tb.addAction(act_export_estrada)

        self.setStatusBar(QStatusBar())
        wm = QLabel("Made by George and Hafiz")
        wm.setStyleSheet("color: #666; padding: 0 10px; font-style: italic;")
        self.statusBar().addPermanentWidget(wm)

        center = QSplitter(Qt.Horizontal)

        # 3D view
        self.view3d = WhiteGLViewWidget()
        self.view3d.setCameraPosition(distance=20, elevation=20, azimuth=45)
        self._grid = gl.GLGridItem()
        self._grid.setSize(20, 20, 1)
        self._grid.setSpacing(1, 1, 1)
        self.view3d.addItem(self._grid)

        right_tabs = QTabWidget()

        # ---- Graph tab (2D) ----
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        self.plot2d = pg.PlotWidget()
        self.plot2d.setMenuEnabled(False)
        self.plot2d.getPlotItem().hideAxis("bottom")
        self.plot2d.getPlotItem().hideAxis("left")
        self.plot2d.setBackground("w")
        self.graph_item = pg.GraphItem()
        self.plot2d.addItem(self.graph_item)
        graph_layout.addWidget(self.plot2d)

        self.adj_list_text = QPlainTextEdit()
        self.adj_list_text.setReadOnly(True)

        self.adj_matrix_text = QPlainTextEdit()
        self.adj_matrix_text.setReadOnly(True)

        self.eigs_text = QPlainTextEdit()
        self.eigs_text.setReadOnly(True)

        # ---- Energy tab (scalar) ----
        self.energy_text = QPlainTextEdit()
        self.energy_text.setReadOnly(True)

        # ---- k-Moment tab (curve) ----
        kM_tab = QWidget()
        kM_layout = QVBoxLayout(kM_tab)
        self.kM_plot = pg.PlotWidget()
        self.kM_plot.setBackground("w")
        self.kM_plot.showGrid(x=True, y=True, alpha=0.2)
        self.kM_plot.setLabel("bottom", "k")
        self.kM_plot.setLabel("left", "μ_k = Σ λ^k")
        kM_layout.addWidget(self.kM_plot)
        self._kM_zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("k", width=1))
        self.kM_plot.addItem(self._kM_zero)

        # ---- Topological tab (TWO graphs) ----
        topo_tab = QWidget()
        topo_layout = QVBoxLayout(topo_tab)

        plots_row = QWidget()
        row_layout = QHBoxLayout(plots_row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self.topo_plot_full = pg.PlotWidget()
        self.topo_plot_full.setBackground("w")
        self.topo_plot_full.getPlotItem().hideAxis("bottom")
        self.topo_plot_full.getPlotItem().hideAxis("left")
        self.topo_plot_full.setTitle("With Hydrogen (Full Graph) — nonstandard for Wiener")
        self.topo_graph_full = pg.GraphItem()
        self.topo_plot_full.addItem(self.topo_graph_full)

        self.topo_plot_heavy = pg.PlotWidget()
        self.topo_plot_heavy.setBackground("w")
        self.topo_plot_heavy.getPlotItem().hideAxis("bottom")
        self.topo_plot_heavy.getPlotItem().hideAxis("left")
        self.topo_plot_heavy.setTitle("Without Hydrogen (H-suppressed) — Wiener definition")
        self.topo_graph_heavy = pg.GraphItem()
        self.topo_plot_heavy.addItem(self.topo_graph_heavy)

        row_layout.addWidget(self.topo_plot_full, 1)
        row_layout.addWidget(self.topo_plot_heavy, 1)

        topo_layout.addWidget(plots_row, 4)

        self.topo_text = QPlainTextEdit()
        self.topo_text.setReadOnly(True)
        topo_layout.addWidget(self.topo_text, 2)

        # ---- Estrada tab ----
        self.estrada_text = QPlainTextEdit()
        self.estrada_text.setReadOnly(True)

        right_tabs.addTab(graph_tab, "Graph")
        right_tabs.addTab(self.adj_list_text, "Adjacency List")
        right_tabs.addTab(self.adj_matrix_text, "Adjacency Matrix")
        right_tabs.addTab(self.eigs_text, "Eigenvalues")
        right_tabs.addTab(self.energy_text, "Energy")
        right_tabs.addTab(kM_tab, "k-Moment")
        right_tabs.addTab(topo_tab, "Wiener Index")
        right_tabs.addTab(self.estrada_text, "Estrada Index")

        center.addWidget(self.view3d)
        center.addWidget(right_tabs)
        center.setStretchFactor(0, 2)
        center.setStretchFactor(1, 2)

        self.setCentralWidget(center)

        # ---- Atom table dock ----
        dock_atoms = QDockWidget("Atoms", self)
        dock_atoms.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        from PySide6.QtWidgets import QTableView
        self.atom_model = AtomTableModel()
        self.atom_table = QTableView()
        self.atom_table.setModel(self.atom_model)
        self.atom_table.setAlternatingRowColors(True)
        self.atom_table.horizontalHeader().setStretchLastSection(True)

        dock_atoms.setWidget(self.atom_table)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_atoms)

    def _apply_light_theme(self):
        app = QApplication.instance()
        app.setPalette(app.style().standardPalette())
        pg.setConfigOption("foreground", "k")

    # --------- Drag & drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".xyz"):
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".xyz"):
                self._load_xyz(path)

    def open_xyz(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open XYZ", "", "XYZ files (*.xyz);;All files (*.*)")
        if path:
            self._load_xyz(path)

    def _load_xyz(self, path: str):
        try:
            self.frames = read_xyz_frames(path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to open XYZ", str(e))
            return

        self.current_path = path
        self.current_frame = 0

        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        for i, fr in enumerate(self.frames):
            self.frame_combo.addItem(f"{i+1}/{len(self.frames)} — {fr.comment[:60]}")
        self.frame_combo.setCurrentIndex(0)
        self.frame_combo.blockSignals(False)

        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)} | Frames: {len(self.frames)}", 5000)
        self._update_views()

    def _on_frame_changed(self, idx: int):
        if not self.frames:
            return
        self.current_frame = max(0, min(idx, len(self.frames) - 1))
        self._update_views()

    def _current(self):
        if not self.frames:
            return None
        fr = self.frames[self.current_frame]
        bonds = infer_bonds_cutoff_table(fr.elements, fr.coords)
        G = build_graph(fr.elements, fr.atomic_numbers, fr.coords, bonds)
        return fr, bonds, G

    # --------- 3D helpers ----------
    def _clear_3d_atoms(self):
        for item in self._atom_mesh_items:
            try:
                self.view3d.removeItem(item)
            except Exception:
                pass
        self._atom_mesh_items = []

    def _clear_bond_lines(self):
        for it in self._bond_line_items:
            try:
                self.view3d.removeItem(it)
            except Exception:
                pass
        self._bond_line_items = []

    def _draw_thick_bonds_lines(
        self,
        coords: np.ndarray,
        bonds: List[Tuple[int, int]],
        color=(0.45, 0.45, 0.45, 0.9),
        offset=0.03,
        layers=7,
    ):
        self._clear_bond_lines()
        if not bonds:
            return

        half = layers // 2
        offsets = [0.0]
        for k in range(1, half + 1):
            offsets += [k * offset, -k * offset]

        for layer_idx, o in enumerate(offsets):
            segs = []
            for i, j in bonds:
                p1 = coords[i]
                p2 = coords[j]
                v = p2 - p1
                L = float(np.linalg.norm(v))
                if L < 1e-9:
                    continue
                vhat = v / L

                a = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(vhat, a))) > 0.95:
                    a = np.array([0.0, 1.0, 0.0])

                perp1 = np.cross(vhat, a)
                n1 = float(np.linalg.norm(perp1))
                if n1 < 1e-9:
                    continue
                perp1 /= n1
                perp2 = np.cross(vhat, perp1)

                offvec = (perp1 if (layer_idx % 2 == 0) else perp2) * o
                segs.append(p1 + offvec)
                segs.append(p2 + offvec)

            if segs:
                segs = np.array(segs, dtype=float)
                item = gl.GLLinePlotItem(pos=segs, mode="lines", antialias=True, color=color, width=1)
                self.view3d.addItem(item)
                self._bond_line_items.append(item)

    # --------- Update all views ----------
    def _update_views(self):
        cur = self._current()
        if cur is None:
            self.atom_model.set_data([], np.zeros((0, 3)))
            self.adj_list_text.setPlainText("Drop or open a .xyz file to begin.")
            self.adj_matrix_text.setPlainText("")
            self.eigs_text.setPlainText("")
            self.energy_text.setPlainText("")
            self.estrada_text.setPlainText("")
            self.kM_plot.clear()
            self.kM_plot.addItem(self._kM_zero)
            self.topo_plot_full.clear()
            self.topo_plot_heavy.clear()
            self.topo_text.setPlainText("")
            self._last_W = None
            self._last_eigs = None
            self._last_mu = None
            self._last_energy = None
            self._last_wiener_full = None
            self._last_wiener_heavy = None
            self._last_EE_full = None
            self._last_EE_heavy = None
            self._last_EE_nodes_full = None
            self._last_EE_nodes_heavy = None
            return

        fr, bonds, G = cur
        self.atom_model.set_data(fr.elements, fr.coords)

        self._render_3d(fr, bonds)
        self._render_2d(G)

        # identity-encoded adjacency matrix W (your original spectral descriptor)
        A_graph = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=float)
        W = A_graph.copy()
        np.fill_diagonal(W, np.array(fr.atomic_numbers, dtype=float))
        eigsW = np.sort(np.linalg.eigvalsh(W))[::-1]

        K_DEFAULT = 30
        E = graph_energy(eigsW)
        mu = k_moments(eigsW, K=K_DEFAULT)

        # Wiener (standard, unweighted distances)
        Gh = hydrogen_suppressed_graph(G, fr.elements)
        wiener_full = wiener_index_sum_components(G)
        wiener_heavy = wiener_index_sum_components(Gh)

        self._last_wiener_full = float(wiener_full)
        self._last_wiener_heavy = float(wiener_heavy)

        # Estrada on adjacency matrices (use explicit nodelists to keep node order aligned)
        full_nodes = sorted(G.nodes())
        heavy_nodes = sorted(Gh.nodes())
        A_full = nx.to_numpy_array(G, nodelist=full_nodes, dtype=float)
        A_heavy = nx.to_numpy_array(Gh, nodelist=heavy_nodes, dtype=float)

        EE_full, EE_nodes_full = estrada_index_and_node_centrality_from_adj(A_full)
        EE_heavy, EE_nodes_heavy = estrada_index_and_node_centrality_from_adj(A_heavy)

        self._last_W = W
        self._last_eigs = eigsW
        self._last_mu = mu
        self._last_energy = E

        self._last_EE_full = float(EE_full)
        self._last_EE_heavy = float(EE_heavy)
        self._last_EE_nodes_full = EE_nodes_full
        self._last_EE_nodes_heavy = EE_nodes_heavy

        self._render_adjacency_list(fr, G)
        self._render_adjacency_matrix(fr, W)
        self._render_eigenvalues(fr, eigsW)
        self._render_energy(fr, E)
        self._render_k_moment(mu)
        self._render_topological_graphs(fr, G, Gh, wiener_full, wiener_heavy)
        self._render_estrada(fr, G, Gh, full_nodes, heavy_nodes, EE_full, EE_heavy, EE_nodes_full, EE_nodes_heavy)

        self.statusBar().showMessage(
            f"Atoms: {G.number_of_nodes()} | Bonds: {G.number_of_edges()} | "
            f"E=Σ|λ|: {E:.6f} | Wiener(H-supp): {wiener_heavy:.0f} | Estrada(H-supp): {EE_heavy:.6f}",
            7000
        )

    # --------- Render 3D ----------
    def _render_3d(self, fr: Frame, bonds: List[Tuple[int, int]]):
        self._clear_3d_atoms()
        self._clear_bond_lines()

        coords = fr.coords.astype(float)
        radii = np.array([COVALENT_RADII.get(el, DEFAULT_RADIUS) for el in fr.elements], dtype=float)

        VISUAL_SCALE = 0.55
        for (x, y, z), el, r in zip(coords, fr.elements, radii):
            color = ELEMENT_COLORS.get(el, (0.8, 0.8, 0.85, 1.0))
            sphere = gl.GLMeshItem(meshdata=self._sphere_mesh, smooth=True, shader="shaded", color=color, drawEdges=False)
            s = float(r * VISUAL_SCALE)
            sphere.scale(s, s, s)
            sphere.translate(float(x), float(y), float(z))
            self.view3d.addItem(sphere)
            self._atom_mesh_items.append(sphere)

        if bonds:
            self._draw_thick_bonds_lines(coords=coords, bonds=bonds, color=(0.45, 0.45, 0.45, 0.9), offset=0.03, layers=7)

        center = coords.mean(axis=0)
        spread = np.linalg.norm(coords - center, axis=1).max() if len(coords) else 5.0
        self.view3d.opts["center"] = pg.Vector(float(center[0]), float(center[1]), float(center[2]))
        self.view3d.setCameraPosition(distance=max(10, float(spread) * 3))

    # --------- Render 2D graph (main Graph tab) ----------
    def _render_2d(self, G: nx.Graph):
        self.plot2d.clear()
        self.plot2d.addItem(self.graph_item)
        if G.number_of_nodes() == 0:
            return

        pos = nx.spring_layout(G, seed=42)
        nodes = sorted(G.nodes())
        xy = np.array([pos[n] for n in nodes], dtype=float)

        index_map = {n: i for i, n in enumerate(nodes)}
        edges = np.array([[index_map[u], index_map[v]] for u, v in G.edges()], dtype=int) if G.number_of_edges() else np.zeros((0, 2), dtype=int)

        labels = [G.nodes[n].get("element", "?") for n in nodes]
        self.graph_item.setData(pos=xy, adj=edges, size=18, symbol="o", pxMode=True, texts=labels, textPen=pg.mkPen("k"))
        self.plot2d.autoRange(padding=0.15)

    # --------- Render Topological graphs ----------
    def _render_graphitem(self, plot: pg.PlotWidget, item: pg.GraphItem, G: nx.Graph, title: str):
        plot.clear()
        plot.setTitle(title)
        plot.getPlotItem().hideAxis("bottom")
        plot.getPlotItem().hideAxis("left")
        plot.setBackground("w")
        plot.addItem(item)

        if G.number_of_nodes() == 0:
            return

        if G.number_of_nodes() == 1:
            n0 = next(iter(G.nodes()))
            xy = np.array([[0.0, 0.0]], dtype=float)
            item.setData(
                pos=xy,
                adj=np.zeros((0, 2), dtype=int),
                size=22,
                symbol="o",
                pxMode=True,
                texts=[G.nodes[n0].get("element", "?")],
                textPen=pg.mkPen("k"),
            )
            plot.autoRange(padding=0.4)
            return

        pos = nx.spring_layout(G, seed=42)
        nodes = sorted(G.nodes())
        xy = np.array([pos[n] for n in nodes], dtype=float)
        idx = {n: i for i, n in enumerate(nodes)}
        edges = np.array([[idx[u], idx[v]] for u, v in G.edges()], dtype=int) if G.number_of_edges() else np.zeros((0, 2), dtype=int)
        labels = [G.nodes[n].get("element", "?") for n in nodes]

        item.setData(pos=xy, adj=edges, size=18, symbol="o", pxMode=True, texts=labels, textPen=pg.mkPen("k"))
        plot.autoRange(padding=0.2)

    def _render_topological_graphs(self, fr: Frame, G: nx.Graph, Gh: nx.Graph, w_full: float, w_heavy: float):
        self._render_graphitem(self.topo_plot_full, self.topo_graph_full, G, "With Hydrogen (Full Graph) — nonstandard for Wiener")
        self._render_graphitem(self.topo_plot_heavy, self.topo_graph_heavy, Gh, "Without Hydrogen (H-suppressed) — Wiener definition")

        self.topo_text.setPlainText(
            "\n".join([
                f"Comment: {fr.comment}",
                "",
                "Wiener index (standard chemical graph theory):",
                "  W(G) = Σ_{u<v} d(u,v)",
                "  d(u,v) = topological distance (shortest path in number of bonds)",
                "  Chemical graph uses non-hydrogen atoms (H-suppressed).",
                "",
                f"Full graph (with H) [nonstandard]: nodes={G.number_of_nodes()} edges={G.number_of_edges()}  W={w_full:.0f}",
                f"H-suppressed graph [standard]:     nodes={Gh.number_of_nodes()} edges={Gh.number_of_edges()}  W={w_heavy:.0f}",
            ])
        )

    # --------- Render Estrada ----------
    def _render_estrada(
        self,
        fr: Frame,
        G: nx.Graph,
        Gh: nx.Graph,
        full_nodes: List[int],
        heavy_nodes: List[int],
        EE_full: float,
        EE_heavy: float,
        EE_nodes_full: np.ndarray,
        EE_nodes_heavy: np.ndarray,
    ):
        lines: List[str] = [
            f"Comment: {fr.comment}",
            "",
            "Estrada index (adjacency spectrum):",
            "  EE(G) = Σ_j exp(λ_j) = tr(exp(A))",
            "",
            "Node subgraph centrality:",
            "  EE(v) = (exp(A))_vv",
            "",
            f"Full graph (with H): nodes={G.number_of_nodes()} edges={G.number_of_edges()}  EE={EE_full:.10f}",
            f"H-suppressed graph:  nodes={Gh.number_of_nodes()} edges={Gh.number_of_edges()}  EE={EE_heavy:.10f}",
            "",
            "Top nodes by subgraph centrality (H-suppressed):"
        ]

        if len(heavy_nodes) == 0:
            lines.append("  (no nodes)")
            self.estrada_text.setPlainText("\n".join(lines))
            return

        pairs = [(heavy_nodes[i], float(EE_nodes_heavy[i])) for i in range(len(heavy_nodes))]
        pairs.sort(key=lambda t: t[1], reverse=True)

        for n, score in pairs[:min(12, len(pairs))]:
            el = Gh.nodes[n].get("element", "?")
            Z = Gh.nodes[n].get("Z", "?")
            lines.append(f"  node {n:>3d}  {el}({Z})   EE(v)={score:.10f}")

        self.estrada_text.setPlainText("\n".join(lines))

    # --------- Text tabs ----------
    def _render_adjacency_list(self, fr: Frame, G: nx.Graph):
        lines = [
            f"Comment: {fr.comment}",
            f"Atoms: {G.number_of_nodes()}  Bonds: {G.number_of_edges()}",
            "Bond rule: pairwise cutoffs (C/N/O/H) + fallback for other elements",
            "",
            "Atoms (index: symbol(Z)):",
            "  " + " ".join(f"{i}:{fr.elements[i]}({fr.atomic_numbers[i]})" for i in range(len(fr.elements))),
            "",
            "Adjacency list:"
        ]
        for n in sorted(G.nodes()):
            el = G.nodes[n].get("element", "?")
            neigh = sorted(G.neighbors(n))
            lines.append(f"{n} ({el}): {list(neigh)}")
        self.adj_list_text.setPlainText("\n".join(lines))

    def _render_adjacency_matrix(self, fr: Frame, W: np.ndarray):
        n = W.shape[0]
        header = [
            f"Comment: {fr.comment}",
            f"N = {n}",
            "",
            "Adjacency Matrix (identity-encoded): diag=atomic numbers (Z), off-diagonal=bonds (0/1)",
            "",
        ]
        if n <= 200:
            Wi = np.rint(W).astype(int)
            body = "\n".join(" ".join(map(str, row.tolist())) for row in Wi)
            self.adj_matrix_text.setPlainText("\n".join(header) + body)
        else:
            self.adj_matrix_text.setPlainText("\n".join(header) + "(Matrix too large to display. Export CSV.)")

    def _render_eigenvalues(self, fr: Frame, eigsW: np.ndarray):
        lines = [
            f"Comment: {fr.comment}",
            "",
            "Eigenvalues of W (descending):"
        ]
        for i, v in enumerate(eigsW):
            lines.append(f"{i:4d}: {v:.10f}")
        self.eigs_text.setPlainText("\n".join(lines))

    def _render_energy(self, fr: Frame, E: float):
        self.energy_text.setPlainText(
            "\n".join([
                f"Comment: {fr.comment}",
                "",
                "Energy:",
                "  E(G) = Σ |λ_i|   (using eigenvalues of W)",
                "",
                f"E(G) = {E:.10f}",
            ])
        )

    def _render_k_moment(self, mu: np.ndarray):
        self.kM_plot.clear()
        self.kM_plot.addItem(self._kM_zero)
        if mu.size == 0:
            return
        x = np.arange(1, mu.size + 1, dtype=float)
        self.kM_plot.plot(x, mu, pen=pg.mkPen("k", width=1))
        self.kM_plot.autoRange(padding=0.05)

    # ---------------- Exports ----------------
    def export_adjacency_csv(self):
        if self._last_W is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return

        default = "adjacency.csv"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default = f"{base}_frame{self.current_frame+1}_adjacency.csv"

        out, _ = QFileDialog.getSaveFileName(self, "Save adjacency CSV", default, "CSV (*.csv)")
        if not out:
            return

        Wi = np.rint(self._last_W).astype(int)
        try:
            np.savetxt(out, Wi, fmt="%d", delimiter=",")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Exported", f"Saved: {out}")

    def export_eigenvalues_csv(self):
        if self._last_eigs is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return

        default = "eigenvalues.csv"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default = f"{base}_frame{self.current_frame+1}_eigenvalues.csv"

        out, _ = QFileDialog.getSaveFileName(self, "Save eigenvalues CSV", default, "CSV (*.csv)")
        if not out:
            return

        try:
            np.savetxt(out, self._last_eigs, fmt="%.18e", delimiter=",")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Exported", f"Saved: {out}")

    def export_k_moment_csv(self):
        if self._last_mu is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return

        default = "k_moment.csv"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default = f"{base}_frame{self.current_frame+1}_k_moment.csv"

        out, _ = QFileDialog.getSaveFileName(self, "Save k-moment CSV", default, "CSV (*.csv)")
        if not out:
            return

        k = np.arange(1, self._last_mu.size + 1, dtype=int)
        data = np.column_stack([k, self._last_mu])
        try:
            np.savetxt(out, data, fmt=["%d", "%.18e"], delimiter=",", header="k,mu_k", comments="")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Exported", f"Saved: {out}")

    def export_wiener_csv(self):
        if self._last_wiener_full is None or self._last_wiener_heavy is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return

        default = "wiener.csv"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default = f"{base}_frame{self.current_frame+1}_wiener.csv"

        out, _ = QFileDialog.getSaveFileName(self, "Save Wiener CSV", default, "CSV (*.csv)")
        if not out:
            return

        cur = self._current()
        if cur is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return
        fr, bonds, G = cur
        Gh = hydrogen_suppressed_graph(G, fr.elements)

        rows = [
            ["graph_type", "nodes", "edges", "wiener"],
            ["full_with_H", G.number_of_nodes(), G.number_of_edges(), f"{self._last_wiener_full:.0f}"],
            ["H_suppressed", Gh.number_of_nodes(), Gh.number_of_edges(), f"{self._last_wiener_heavy:.0f}"],
        ]

        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        QMessageBox.information(self, "Exported", f"Saved: {out}")

    def export_estrada_csv(self):
        if self._last_EE_heavy is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return

        default = "estrada.csv"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            default = f"{base}_frame{self.current_frame+1}_estrada.csv"

        out, _ = QFileDialog.getSaveFileName(self, "Save Estrada CSV", default, "CSV (*.csv)")
        if not out:
            return

        cur = self._current()
        if cur is None:
            QMessageBox.information(self, "No data", "Load an XYZ first.")
            return

        fr, bonds, G = cur
        Gh = hydrogen_suppressed_graph(G, fr.elements)

        full_nodes = sorted(G.nodes())
        heavy_nodes = sorted(Gh.nodes())
        A_full = nx.to_numpy_array(G, nodelist=full_nodes, dtype=float)
        A_heavy = nx.to_numpy_array(Gh, nodelist=heavy_nodes, dtype=float)

        EE_full, EE_nodes_full = estrada_index_and_node_centrality_from_adj(A_full)
        EE_heavy, EE_nodes_heavy = estrada_index_and_node_centrality_from_adj(A_heavy)

        rows = [["graph_type", "total_EE", "node", "element", "Z", "subgraph_centrality"]]

        for i, n in enumerate(full_nodes):
            el = G.nodes[n].get("element", "?")
            Z = G.nodes[n].get("Z", "")
            rows.append(["full_with_H", f"{EE_full:.12f}", n, el, Z, f"{float(EE_nodes_full[i]):.12f}"])

        for i, n in enumerate(heavy_nodes):
            el = Gh.nodes[n].get("element", "?")
            Z = Gh.nodes[n].get("Z", "")
            rows.append(["H_suppressed", f"{EE_heavy:.12f}", n, el, Z, f"{float(EE_nodes_heavy[i]):.12f}"])

        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        QMessageBox.information(self, "Exported", f"Saved: {out}")


def main():
    app = QApplication(sys.argv)
    w = StudioWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
