# The original desktop app (kept for reference)

`xyz_graph_studio.py` is the PySide6 desktop version, unchanged. `XYZGraphStudio.spec` builds it into a
Windows .exe with PyInstaller (`pyinstaller XYZGraphStudio.spec`).

Known problem: its bond rule uses fixed cutoffs (C–C 1.55 Å, C–H 1.12 Å, …). Real C–C bonds in strained
or crowded molecules are often longer than 1.55 Å (cubane: 1.57 Å), and C–H bonds from semi-empirical or
DFT optimisations can be longer than 1.12 Å. Those bonds are silently dropped, and every matrix and index
after that is wrong. The web app in the folder above uses covalent radii instead, checks every atom's
number of bonds, and can still reproduce this old rule (Bonds tab, "Pair table").

Note: the .exe in the old `dist` folder was built from an earlier version of this file that only had the
adjacency and eigenvalue tabs.
