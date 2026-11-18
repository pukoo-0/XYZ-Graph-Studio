# XYZ Graph Studio

Turns an `.xyz` file into two edge-weighted molecular graph representations across five atomic
properties (**MIREZ**: Mass, first Ionisation energy, covalent Radius, Allen Electronegativity, and
atomic number Z) for the 16 main-group covalent elements (`H` + Groups 13–17 across Periods 2–4,
including `SCHNOP`):

1. **Undirected sum graph ($W_{\text{sum}}$):** $W_{ij} = p_i + p_j$ on covalent bonds, $W_{ii} = 0$ (symmetric).
2. **Directed target-weighted graph ($W_{\text{dir}}$):** $W_{ij} = p_j$ for directed arc $i \to j$, $W_{ii} = 0$ (non-symmetric).

Runs entirely in the browser: Python (Pyodide + NumPy) is downloaded once, and files never leave the
visitor's computer.

## Deploy

**Streamlit Community Cloud:** new app, pick this repository, main file path **`streamlit_app.py`**.

**GitHub Pages (optional, loads faster):** Settings → Pages → branch `main`, folder `/docs`.

**Offline:** open `docs/index.html` in a browser (it needs internet the first time to fetch Pyodide).

## What it does

| Tab | Contents |
|---|---|
| Bonds | Bond check per atom, histogram of all distances with the cutoff marked, bond rule, close calls, manual add/remove |
| Undirected | MIREZ property selector, 2D graph with $w_{ij} = p_i + p_j$ edge weights, symmetric matrix $W_{\text{sum}}$ (map & numbers), weighted adjacency list, eigenvalues, energies, and spectral moments $\mu_k$ |
| Directed | MIREZ property selector, 2D directed graph with curved arcs $i \to j$ ($p_j$) and $j \to i$ ($p_i$), non-symmetric matrix $W_{\text{dir}}$ (map & numbers), directed adjacency list, eigenvalues, energies, and spectral moments $\mu_k$ |
| Indices | Wiener, Estrada, Zagreb $M_1$ and $M_2$, Randić for the full and the H-suppressed graph; subgraph centrality |
| Atoms | Coordinates, atomic number $Z$, active MIREZ property value, number of bonds, neighbours |

### Supported elements & MIREZ constant sources

Supported 16 main-group covalent elements: `H, B, C, N, O, F, Al, Si, P, S, Cl, Ga, Ge, As, Se, Br`.
Passing any atom outside this set raises a `ValueError`.

* **`Z` (Atomic number):** exact proton count (`1` to `35`).
* **`M` (Atomic mass, u):** IUPAC abridged standard atomic weights (2024).
* **`R` (Covalent radius, Å):** Cordero et al., *Dalton Trans.* 2008, 2832.
* **`E` (Electronegativity):** Allen spectroscopic electronegativity (L. C. Allen, *J. Am. Chem. Soc.* 1989, 111, 9003, Table 1).
* **`I` (First ionisation energy, eV):** NIST Atomic Spectra Database (Kramida et al.).

## Files

```
streamlit_app.py      ← the file Streamlit runs (serves docs/index.html)
docs/index.html       ← the finished web app (built, do not edit by hand)
graph_core.py         ← all the chemistry and graph theory (NumPy only)
web/bridge.py         ← connects the page to graph_core
web/index.src.html    ← the page's source (HTML, CSS, JavaScript)
build_web.py          ← builds docs/index.html from the files above
examples/             ← hocl, benzene, cubane, adamantane
tests/                ← checks against networkx and known values (pytest)
legacy/               ← the original PySide6 desktop app
```

After changing `graph_core.py`, `web/bridge.py` or `web/index.src.html`:

```
python build_web.py
pytest
```

then commit `docs/index.html` as well.

© 2026 George Sanjaya & Hafiz Aji Aziz. Made by George and Hafiz.
