"""Checks graph_core against networkx and against hand-known answers.  Run:  pytest"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "web"))
import graph_core as gc  # noqa: E402

nx = pytest.importorskip("networkx")


def load(name):
    with open(os.path.join(ROOT, "examples", name + ".xyz"), encoding="utf-8") as fh:
        return gc.parse_xyz(fh.read())[0]


def to_nx(res):
    G = nx.Graph()
    G.add_nodes_from(range(res["n"]))
    G.add_edges_from((b[0], b[1]) for b in res["bonds"])
    return G


@pytest.mark.parametrize("name,bonds", [("benzene", 12), ("cubane", 20), ("adamantane", 28)])
def test_examples_bond_correctly(name, bonds):
    res = gc.analyse(load(name))
    assert len(res["bonds"]) == bonds
    assert res["issues"] == []
    assert len(res["fragments"]) == 1


def test_known_wiener_values():
    # H-suppressed Wiener indices: benzene 27, cubane 48, adamantane 96
    for name, w in (("benzene", 27), ("cubane", 48), ("adamantane", 96)):
        assert gc.analyse(load(name))["graphs"]["heavy"]["wiener"] == w


@pytest.mark.parametrize("name", ["benzene", "cubane", "adamantane"])
def test_matches_networkx(name):
    fr = load(name)
    res = gc.analyse(fr)
    G = to_nx(res)
    Gh = G.subgraph([i for i, s in enumerate(fr.symbols) if s != "H"])
    assert res["graphs"]["full"]["wiener"] == nx.wiener_index(G)
    assert res["graphs"]["heavy"]["wiener"] == nx.wiener_index(Gh)
    assert res["graphs"]["heavy"]["estrada"] == pytest.approx(nx.estrada_index(Gh))
    sc = nx.subgraph_centrality(G)
    assert np.allclose(res["centrality_full"], [sc[i] for i in range(res["n"])])
    lam = np.linalg.eigvalsh(nx.to_numpy_array(G, nodelist=range(res["n"])))
    assert res["energy_A"] == pytest.approx(np.abs(lam).sum())


def test_old_pair_table_misses_cubane_bonds():
    res = gc.analyse(load("cubane"), {"rule": "table"})
    assert len(res["bonds"]) == 8            # only the C–H bonds; 1.57 Å C–C is above 1.55 Å
    assert len(res["issues"]) == 8
    assert any(not q["by_rule"] for q in res["near"])


def test_manual_add_and_remove():
    fr = load("benzene")
    res = gc.analyse(fr, {"remove": [[0, 1]], "add": [[0, 3]]})
    pairs = {(b[0], b[1]) for b in res["bonds"]}
    assert (0, 1) not in pairs and (0, 3) in pairs


def test_energy_with_Z_diagonal_is_just_the_trace():
    res = gc.analyse(load("adamantane"))
    sp = res["spectrum"]
    assert sp["energy_is_trace"]
    assert sp["energy_abs"] == pytest.approx(sum(res["Z"]))


def test_moments_are_traces_of_powers():
    fr = load("cubane")
    res = gc.analyse(fr, {"kmax": 4})
    W = res["_W"]
    for k, m in enumerate(res["spectrum"]["moments"], 1):
        assert m == pytest.approx(np.trace(np.linalg.matrix_power(W, k)))


def test_parse_variants():
    text = "3\nwater\nO 0 0 0\n1 0.96 0 0\nh -0.24 0.93 0\n3\nframe 2\n8 0 0 0\nH 0.97 0 0\nH -0.24 0.93 0\n"
    frames = gc.parse_xyz(text)
    assert len(frames) == 2 and frames[0].symbols == ["O", "H", "H"]
    with pytest.raises(ValueError):
        gc.parse_xyz("5\nshort\nC 0 0 0\n")


def test_exports_keep_old_layout():
    res = gc.analyse(load("benzene"))
    files = gc.export_files(res, "bz", base=0)
    rows = files["bz_adjacency.csv"].strip().splitlines()
    assert len(rows) == 12 and rows[0].split(",")[0] == "6"
    assert files["bz_k_moment.csv"].startswith("k,mu_k\n")
    assert "full_with_H,12,12" in files["bz_wiener.csv"]


def test_bridge_round_trip():
    import bridge
    bridge.HERE = os.path.join(ROOT, "examples")
    assert json.loads(bridge.example("cubane"))["ok"]
    out = json.loads(bridge.analyse(0, json.dumps({"rule": "radii"})))
    assert out["ok"] and len(out["bonds"]) == 20 and out["histogram"]["pairs"] == ["C-C", "C-H", "H-H"]
    files = json.loads(bridge.exports(0, "{}", "cubane", 1))["files"]
    assert "cubane_report.txt" in files


def test_built_page_is_up_to_date():
    import build_web
    with open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8") as fh:
        assert fh.read() == build_web.build(), "run: python build_web.py"
