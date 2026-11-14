# XYZ Graph Studio

Turns an `.xyz` file into a molecular graph and computes the adjacency matrix, its eigenvalues, graph
energies, spectral moments, and the Wiener, Estrada, Zagreb and Randić indices. Before any of that, it
checks that the bonds make chemical sense, because every number downstream depends on them.

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
| Graph | 2D drawing (spring layout or flattened 3D shape), with or without hydrogens; adjacency list |
| Matrix | W = A + diag(Z) or plain A, as a map or as numbers |
| Spectrum | Σ\|λ\|, centred energy Σ\|λ − tr W/n\|, Gutman graph energy of A; eigenvalues; μ_k = tr(W^k) |
| Indices | Wiener, Estrada, Zagreb M1 and M2, Randić for the full and the H-suppressed graph; subgraph centrality |
| Atoms | Coordinates, number of bonds, neighbours |

The Download menu saves every result as .csv/.txt (or all of them in one .zip) and a picture of the
molecule. The .csv layouts match the old desktop app. Atom numbers can start at 1 (like the file) or 0
(like the old app).

### Bond rule

Default: atoms i and j are bonded when `d_ij ≤ 1.20 × (r_i + r_j)`, with covalent radii from
Cordero et al., *Dalton Trans.* 2008, 2832. The old app's fixed pair table is still available.
Any bond can be added or removed by hand (click an atom).

### A note on Σ|λ| with Z on the diagonal

For W = A + diag(Z), every eigenvalue is ≥ 0 for ordinary organic molecules (Gershgorin: Z_i ≥ degree_i).
Then Σ|λ_i| = Σλ_i = tr W = Σ Z_i, which is the same for any set of bonds. The app reports it, but also
the centred energy and the Gutman energy of A, which do depend on the structure.

## Files

```
streamlit_app.py      ← the file Streamlit runs (serves docs/index.html)
docs/index.html       ← the finished web app (built, do not edit by hand)
graph_core.py         ← all the chemistry and graph theory (NumPy only)
web/bridge.py         ← connects the page to graph_core
web/index.src.html    ← the page's source (HTML, CSS, JavaScript)
build_web.py          ← builds docs/index.html from the three files above
examples/             ← benzene, cubane, adamantane
tests/                ← checks against networkx and known values (pytest)
legacy/               ← the original PySide6 desktop app
```

After changing `graph_core.py`, `web/bridge.py` or `web/index.src.html`:

```
python build_web.py
pytest
```

then commit `docs/index.html` as well.

Made by George and Hafiz.
