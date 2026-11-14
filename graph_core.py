"""
graph_core.py — the chemistry behind XYZ Graph Studio.

Turns an .xyz file into a molecular graph (atoms = nodes, bonds = edges) and computes:
  * bonds, with a check that every atom has a usual number of them
  * the adjacency matrix A and the vertex-weighted matrix W = A + diag(d)
  * eigenvalues of W, graph energies, spectral moments mu_k = tr(W^k)
  * Wiener index, Estrada index (with subgraph centrality), Zagreb and Randic indices,
    for the full graph and for the hydrogen-suppressed graph

Only needs numpy, so the same file runs on a desktop and inside the browser (Pyodide).
Atom indices inside this module always start at 0; the user interface decides how to number them.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- elements
SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
Z_OF = {s: i for i, s in enumerate(SYMBOLS) if i > 0}

# Covalent radii in Å (Cordero et al., Dalton Trans. 2008, 2832; low-spin values for Mn, Fe, Co; sp3 for C).
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98,
    "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45,
    "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}
DEFAULT_RADIUS = 1.50

# The cutoff table from the original desktop app (Å). Kept so old results can be reproduced.
LEGACY_PAIR_TABLE = {
    "C-C": 1.55, "C-N": 1.50, "C-O": 1.45, "C-H": 1.12, "N-N": 1.40,
    "N-O": 1.37, "N-H": 1.00, "O-O": 1.35, "O-H": 0.97, "H-H": 0.80,
}

# Usual numbers of neighbours (bond order is ignored, so sp2 carbon has 3 and sp carbon has 2).
USUAL_DEGREE = {
    "H": {1}, "B": {3, 4}, "C": {2, 3, 4}, "N": {1, 2, 3, 4}, "O": {1, 2}, "F": {1}, "Si": {4},
    "P": {3, 4, 5}, "S": {1, 2, 3, 4, 6}, "Cl": {1}, "Br": {1}, "I": {1},
}

DEFAULT_SETTINGS = {
    "rule": "radii",          # "radii": d <= scale*(r_a + r_b);  "table": pair table, radii*fallback for other pairs
    "scale": 1.20,
    "table": dict(LEGACY_PAIR_TABLE),
    "fallback": 1.20,
    "window": 0.10,           # pairs this close to their cutoff (Å) are listed for review
    "diagonal": "Z",          # diagonal of W: "Z" (atomic number) or "zero"
    "kmax": 30,
    "add": [],                # bonds you added by hand, [[i, j], ...]
    "remove": [],             # bonds you removed by hand
}


def element_symbol(token: str) -> str:
    """'c' -> 'C', 'CL' -> 'Cl', '6' -> 'C', 'C12' -> 'C', 'Fe1' -> 'Fe'."""
    t = token.strip()
    if t.isdigit():
        z = int(t)
        return SYMBOLS[z] if 0 < z < len(SYMBOLS) else f"Z{z}"
    letters = "".join(ch for ch in t if ch.isalpha())
    if not letters:
        raise ValueError(f"'{token}' is not an element")
    two = letters[:2].capitalize()
    if len(letters) >= 2 and two in Z_OF:
        return two
    one = letters[0].upper()
    if one in Z_OF:
        return one
    raise ValueError(f"'{token}' is not an element")


def atomic_number(sym: str) -> int:
    return Z_OF.get(sym, 0)


def radius(sym: str) -> float:
    return COVALENT_RADII.get(sym, DEFAULT_RADIUS)


# ---------------------------------------------------------------- reading
@dataclass
class Frame:
    comment: str
    symbols: list
    coords: np.ndarray  # shape (n, 3), Å

    @property
    def Z(self):
        return [atomic_number(s) for s in self.symbols]

    @property
    def n(self):
        return len(self.symbols)


def parse_xyz(text: str) -> list:
    """Read every frame of an .xyz file (also multi-frame trajectories from ORCA, xtb, CREST ...)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        head = lines[i].split()
        try:
            n = int(head[0])
        except ValueError:
            raise ValueError(f"Line {i + 1} should hold the number of atoms, but it reads “{lines[i].strip()[:40]}”.")
        if n <= 0:
            raise ValueError(f"Line {i + 1}: the number of atoms must be positive.")
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        syms, xyz = [], []
        for k in range(n):
            ln = i + 2 + k
            if ln >= len(lines) or not lines[ln].strip():
                raise ValueError(f"Frame {len(frames) + 1} says it has {n} atoms, but only {k} were found.")
            parts = lines[ln].split()
            if len(parts) < 4:
                raise ValueError(f"Line {ln + 1} should look like “C 0.0 0.0 0.0”.")
            try:
                syms.append(element_symbol(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError as e:
                raise ValueError(f"Line {ln + 1}: {e}") from None
        frames.append(Frame(comment, syms, np.array(xyz, dtype=float)))
        i += 2 + n
    if not frames:
        raise ValueError("The file has no atoms in it.")
    return frames


# ---------------------------------------------------------------- bonds
def pair_name(a: str, b: str) -> str:
    """Sorted so that C comes first, then H, then alphabetical (Hill order): 'C-H', 'C-O', 'H-O', 'N-O'."""
    def key(s):
        return (0 if s == "C" else 1 if s == "H" else 2, s)
    a, b = sorted((a, b), key=key)
    return f"{a}-{b}"


def _table_lookup(table: dict, a: str, b: str):
    for k in (f"{a}-{b}", f"{b}-{a}"):
        if k in table:
            return float(table[k])
    return None


def cutoff(a: str, b: str, settings: dict) -> float:
    if settings.get("rule") == "table":
        v = _table_lookup(settings.get("table") or {}, a, b)
        if v is not None:
            return v
        return float(settings.get("fallback", 1.2)) * (radius(a) + radius(b))
    return float(settings.get("scale", 1.2)) * (radius(a) + radius(b))


def distance_matrix(X: np.ndarray) -> np.ndarray:
    d = X[:, None, :] - X[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def cutoff_matrix(symbols: list, settings: dict) -> np.ndarray:
    kinds = sorted(set(symbols))
    idx = {s: k for k, s in enumerate(kinds)}
    small = np.array([[cutoff(a, b, settings) for b in kinds] for a in kinds])
    v = np.array([idx[s] for s in symbols])
    return small[v][:, v]


def _clean_pairs(pairs, n):
    out = set()
    for p in pairs or []:
        i, j = int(p[0]), int(p[1])
        if i != j and 0 <= i < n and 0 <= j < n:
            out.add((min(i, j), max(i, j)))
    return out


def find_bonds(frame: Frame, settings: dict) -> dict:
    """Bonds from the distance rule, then your manual additions and removals."""
    n = frame.n
    D = distance_matrix(frame.coords)
    C = cutoff_matrix(frame.symbols, settings)
    iu, ju = np.triu_indices(n, 1)
    d, c = D[iu, ju], C[iu, ju]
    by_rule = (d <= c) & (d > 1e-6)
    added = _clean_pairs(settings.get("add"), n)
    removed = _clean_pairs(settings.get("remove"), n)

    bonds = []
    for k in np.nonzero(by_rule)[0]:
        i, j = int(iu[k]), int(ju[k])
        if (i, j) not in removed:
            bonds.append((i, j, float(d[k]), float(c[k]), "rule"))
    rule_set = {(b[0], b[1]) for b in bonds} | {p for p in removed}
    for i, j in sorted(added):
        if (i, j) not in rule_set:
            bonds.append((i, j, float(D[i, j]), float(C[i, j]), "added"))
    bonds.sort(key=lambda b: (b[0], b[1]))

    window = float(settings.get("window", 0.10))
    near = []
    for k in np.nonzero(np.abs(d - c) <= window)[0]:
        i, j = int(iu[k]), int(ju[k])
        near.append({"i": i, "j": j, "d": float(d[k]), "cut": float(c[k]),
                     "by_rule": bool(by_rule[k]), "pair": pair_name(frame.symbols[i], frame.symbols[j])})
    near.sort(key=lambda r: abs(r["d"] - r["cut"]))
    overlaps = [(int(iu[k]), int(ju[k]), float(d[k])) for k in np.nonzero(d < 0.5)[0]]
    return {"bonds": bonds, "near": near, "overlaps": overlaps,
            "removed": sorted(removed), "added": sorted(added), "D": D, "C": C}


def adjacency(n: int, bonds) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for b in bonds:
        A[b[0], b[1]] = A[b[1], b[0]] = 1.0
    return A


# ---------------------------------------------------------------- graph theory
def neighbours(A: np.ndarray) -> list:
    return [list(np.nonzero(A[i])[0]) for i in range(A.shape[0])]


def components(A: np.ndarray) -> list:
    nb, seen, out = neighbours(A), set(), []
    for s in range(A.shape[0]):
        if s in seen:
            continue
        comp, q = [], deque([s])
        seen.add(s)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in nb[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        out.append(sorted(comp))
    return out


def path_lengths(A: np.ndarray) -> np.ndarray:
    """Shortest path length in bonds between every pair (breadth-first search); -1 if not connected."""
    n = A.shape[0]
    nb = neighbours(A)
    L = np.full((n, n), -1, dtype=int)
    for s in range(n):
        L[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if L[s, v] < 0:
                    L[s, v] = L[s, u] + 1
                    q.append(v)
    return L


def wiener(A: np.ndarray) -> float:
    """W = sum over pairs u<v of d(u,v). For a disconnected graph, pairs in different fragments are skipped."""
    if A.shape[0] < 2:
        return 0.0
    L = path_lengths(A)
    iu = np.triu_indices(A.shape[0], 1)
    vals = L[iu]
    return float(vals[vals > 0].sum())


def estrada(A: np.ndarray):
    """EE = sum exp(lambda_i) = tr(exp(A)); subgraph centrality of node v = (exp(A))_vv."""
    if A.shape[0] == 0:
        return 0.0, np.zeros(0)
    vals, vecs = np.linalg.eigh(A)
    e = np.exp(vals)
    return float(e.sum()), (vecs * vecs) @ e


def degree_indices(A: np.ndarray) -> dict:
    deg = A.sum(axis=1)
    iu, ju = np.nonzero(np.triu(A, 1))
    prod = deg[iu] * deg[ju]
    return {
        "zagreb1": float((deg ** 2).sum()),
        "zagreb2": float(prod.sum()),
        "randic": float((1.0 / np.sqrt(prod)).sum()) if len(prod) else 0.0,
    }


def spectrum(W: np.ndarray, kmax: int = 30) -> dict:
    """Eigenvalues of W (largest first), energies and spectral moments."""
    n = W.shape[0]
    lam = np.sort(np.linalg.eigvalsh(W))[::-1] if n else np.zeros(0)
    tr = float(np.trace(W))
    mean = tr / n if n else 0.0
    kmax = int(max(1, min(int(kmax), 200)))
    with np.errstate(over="ignore", invalid="ignore"):
        mu = [float(np.sum(lam ** k)) for k in range(1, kmax + 1)]
    mu = [m if math.isfinite(m) else None for m in mu]
    e_abs = float(np.abs(lam).sum())
    return {
        "eigenvalues": lam.tolist(),
        "trace": tr,
        "energy_abs": e_abs,
        "energy_centred": float(np.abs(lam - mean).sum()),
        "energy_is_trace": bool(n and lam.min() >= -1e-9),
        "moments": mu,
    }


def weighted_matrix(A: np.ndarray, symbols: list, diagonal: str = "Z") -> np.ndarray:
    W = A.copy()
    if diagonal == "Z":
        np.fill_diagonal(W, [atomic_number(s) for s in symbols])
    return W


# ---------------------------------------------------------------- checks
def valence_check(symbols: list, A: np.ndarray) -> list:
    deg = A.sum(axis=1).astype(int)
    issues = []
    for i, s in enumerate(symbols):
        d = int(deg[i])
        if d == 0 and len(symbols) > 1:
            issues.append({"i": i, "level": "error", "degree": d, "text": "has no bonds at all"})
        elif s in USUAL_DEGREE and d not in USUAL_DEGREE[s]:
            usual = sorted(USUAL_DEGREE[s])
            want = str(usual[0]) if len(usual) == 1 else f"{usual[0]} to {usual[-1]}"
            level = "error" if (s == "H" or d > max(usual)) else "warn"
            issues.append({"i": i, "level": level, "degree": d,
                           "text": f"has {d} bond{'s' if d != 1 else ''}; {s} usually has {want}"})
    return issues


# ---------------------------------------------------------------- everything at once
def merged_settings(settings: dict | None) -> dict:
    s = dict(DEFAULT_SETTINGS)
    s.update({k: v for k, v in (settings or {}).items() if v is not None})
    return s


def analyse(frame: Frame, settings: dict | None = None) -> dict:
    s = merged_settings(settings)
    syms, n = frame.symbols, frame.n
    fb = find_bonds(frame, s)
    A = adjacency(n, fb["bonds"])
    W = weighted_matrix(A, syms, s["diagonal"])
    spec = spectrum(W, s["kmax"])
    lam_A = np.linalg.eigvalsh(A) if n else np.zeros(0)

    heavy = [i for i, e in enumerate(syms) if e != "H"]
    Ah = A[np.ix_(heavy, heavy)]
    comps, comps_h = components(A), components(Ah) if heavy else []
    ee_full, sc_full = estrada(A)
    ee_h, sc_h = estrada(Ah)
    sc_heavy_full = np.full(n, np.nan)
    sc_heavy_full[heavy] = sc_h

    graphs = {}
    for key, M, comp, ee in (("full", A, comps, ee_full), ("heavy", Ah, comps_h, ee_h)):
        graphs[key] = {"nodes": int(M.shape[0]), "edges": int(np.triu(M, 1).sum()), "fragments": len(comp),
                       "wiener": wiener(M), "estrada": ee, **degree_indices(M)}

    return {
        "settings": s,
        "n": n,
        "symbols": syms,
        "Z": frame.Z,
        "coords": frame.coords.tolist(),
        "comment": frame.comment,
        "formula": formula(syms),
        "bonds": fb["bonds"],
        "near": fb["near"],
        "overlaps": fb["overlaps"],
        "added": fb["added"],
        "removed": fb["removed"],
        "degree": A.sum(axis=1).astype(int).tolist(),
        "fragments": comps,
        "issues": valence_check(syms, A),
        "spectrum": spec,
        "energy_A": float(np.abs(lam_A).sum()),
        "graphs": graphs,
        "heavy": heavy,
        "centrality_full": sc_full.tolist(),
        "centrality_heavy": [None if math.isnan(v) else float(v) for v in sc_heavy_full],
        "_A": A, "_W": W, "_D": fb["D"], "_C": fb["C"],
    }


def formula(symbols: list) -> str:
    c = Counter(symbols)
    order = (["C", "H"] if "C" in c else []) + sorted(k for k in c if not ("C" in c and k in ("C", "H")))
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)


def histogram(res: dict, settings: dict, pair: str | None = None, lo: float = 0.4, width: float = 0.02):
    """Counts of interatomic distances, split into bonded and not bonded, up to a little past the largest cutoff."""
    D, C, syms, n = res["_D"], res["_C"], res["symbols"], res["n"]
    kinds = sorted(set(syms))
    t = np.array([kinds.index(s) for s in syms])
    iu, ju = np.triu_indices(n, 1)
    a, b = np.minimum(t[iu], t[ju]), np.maximum(t[iu], t[ju])
    code = a * len(kinds) + b
    names = {int(c): pair_name(kinds[int(c) // len(kinds)], kinds[int(c) % len(kinds)]) for c in np.unique(code)}
    d, c = D[iu, ju], C[iu, ju]
    cuts = {names[int(k)]: float(c[np.argmax(code == k)]) for k in np.unique(code)}
    if pair:
        keep = np.isin(code, [k for k, v in names.items() if v == pair])
        d, c, iu, ju = d[keep], c[keep], iu[keep], ju[keep]
    hi = float(min(max(c.max() if len(c) else 2.0, 1.0) + 0.9, 6.0))
    bonded = res["_A"][iu, ju] > 0
    edges = np.arange(lo, hi + width / 2, width)
    hb, _ = np.histogram(d[bonded], bins=edges)
    hn, _ = np.histogram(d[~bonded], bins=edges)
    return {"lo": lo, "width": width, "bonded": hb.tolist(), "other": hn.tolist(), "cutoffs": cuts,
            "pairs": sorted(set(names.values()))}


# ---------------------------------------------------------------- exports
def _num(i, base):
    return i + base


def export_files(res: dict, stem: str = "molecule", base: int = 1) -> dict:
    """Text files for download. The adjacency, eigenvalue, k-moment, Wiener and Estrada files keep the
    layout of the original desktop app; node numbers follow `base` (0 = like the original, 1 = like the file)."""
    syms, n, W = res["symbols"], res["n"], res["_W"]
    out = {}
    if np.allclose(W, np.rint(W)):
        out[f"{stem}_adjacency.csv"] = "\n".join(",".join(str(int(v)) for v in row) for row in np.rint(W)) + "\n"
    else:
        out[f"{stem}_adjacency.csv"] = "\n".join(",".join(f"{v:.6g}" for v in row) for row in W) + "\n"
    out[f"{stem}_eigenvalues.csv"] = "\n".join(f"{v:.18e}" for v in res["spectrum"]["eigenvalues"]) + "\n"
    out[f"{stem}_k_moment.csv"] = "k,mu_k\n" + "\n".join(
        f"{k},{'' if m is None else f'{m:.18e}'}" for k, m in enumerate(res["spectrum"]["moments"], 1)) + "\n"
    g = res["graphs"]
    out[f"{stem}_wiener.csv"] = ("graph_type,nodes,edges,wiener\n"
                                 f"full_with_H,{g['full']['nodes']},{g['full']['edges']},{g['full']['wiener']:.0f}\n"
                                 f"H_suppressed,{g['heavy']['nodes']},{g['heavy']['edges']},{g['heavy']['wiener']:.0f}\n")
    rows = ["graph_type,total_EE,node,element,Z,subgraph_centrality"]
    for i in range(n):
        rows.append(f"full_with_H,{g['full']['estrada']:.12f},{_num(i, base)},{syms[i]},{res['Z'][i]},{res['centrality_full'][i]:.12f}")
    for i in res["heavy"]:
        rows.append(f"H_suppressed,{g['heavy']['estrada']:.12f},{_num(i, base)},{syms[i]},{res['Z'][i]},{res['centrality_heavy'][i]:.12f}")
    out[f"{stem}_estrada.csv"] = "\n".join(rows) + "\n"
    rows = ["atom_i,atom_j,element_i,element_j,distance_A,cutoff_A,source"]
    for i, j, d, c, src in res["bonds"]:
        rows.append(f"{_num(i, base)},{_num(j, base)},{syms[i]},{syms[j]},{d:.4f},{c:.4f},{src}")
    out[f"{stem}_bonds.csv"] = "\n".join(rows) + "\n"
    out[f"{stem}_adjacency_list.txt"] = adjacency_list_text(res, base)
    out[f"{stem}_report.txt"] = report_text(res, base)
    return out


def adjacency_list_text(res: dict, base: int = 1) -> str:
    A, syms = res["_A"], res["symbols"]
    lines = []
    for i in range(res["n"]):
        nb = [f"{syms[j]}{_num(j, base)}" for j in np.nonzero(A[i])[0]]
        lines.append(f"{syms[i]}{_num(i, base)}: {' '.join(nb) if nb else '(none)'}")
    return "\n".join(lines) + "\n"


def report_text(res: dict, base: int = 1) -> str:
    s, sp, g = res["settings"], res["spectrum"], res["graphs"]
    rule = (f"covalent radii (Cordero 2008), bonded if d <= {s['scale']:.2f} x (r_a + r_b)" if s["rule"] == "radii"
            else "pair table " + ", ".join(f"{k} {v}" for k, v in s["table"].items())
            + f"; other pairs {s['fallback']:.2f} x (r_a + r_b)")
    L = [
        f"XYZ Graph Studio report",
        f"Molecule: {res['formula']}   atoms: {res['n']}   bonds: {len(res['bonds'])}   fragments: {len(res['fragments'])}",
        f"Comment line: {res['comment']}",
        f"Bond rule: {rule}",
        f"Bonds added by hand: {len(res['added'])}   removed by hand: {len(res['removed'])}",
        f"Atom numbering in this report starts at {base}.",
        "",
        "Bond check:",
    ]
    if res["issues"]:
        L += [f"  {res['symbols'][x['i']]}{x['i'] + base} {x['text']}" for x in res["issues"]]
    else:
        L.append("  every atom has a usual number of bonds")
    L += [
        "",
        f"Matrix W = A + diag({'Z' if s['diagonal'] == 'Z' else '0'})",
        f"  trace(W)                       = {sp['trace']:.6f}",
        f"  sum |lambda_i| of W            = {sp['energy_abs']:.10f}",
        f"  sum |lambda_i - trace/n| of W  = {sp['energy_centred']:.10f}",
        f"  graph energy of A (Gutman)     = {res['energy_A']:.10f}",
    ]
    if sp["energy_is_trace"] and s["diagonal"] == "Z":
        L.append("  note: all eigenvalues of W are >= 0, so sum |lambda| equals trace(W) = sum of Z and does not depend on the bonds.")
    L += ["", f"{'':24}{'full graph':>18}{'H-suppressed':>18}"]
    for key, label, fmt in (("nodes", "nodes", "{:.0f}"), ("edges", "edges", "{:.0f}"), ("fragments", "fragments", "{:.0f}"),
                            ("wiener", "Wiener index", "{:.0f}"), ("estrada", "Estrada index", "{:.10f}"),
                            ("zagreb1", "first Zagreb M1", "{:.0f}"), ("zagreb2", "second Zagreb M2", "{:.0f}"),
                            ("randic", "Randic index", "{:.10f}")):
        L.append(f"  {label:<22}{fmt.format(g['full'][key]):>18}{fmt.format(g['heavy'][key]):>18}")
    L.append("")
    L.append("The Wiener, Zagreb and Randic indices are usually reported for the H-suppressed graph.")
    return "\n".join(L) + "\n"
