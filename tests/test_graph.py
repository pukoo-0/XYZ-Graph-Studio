"""Checks graph_core against networkx, Prof Hafiz's HOCl worked examples, and hand-known answers. Run: pytest"""
import json
import math
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


def test_hocl_undirected_and_directed_worked_examples():
    """Prof Hafiz's H-O-Cl worked examples for Z = (1, 8, 17):
    1. Undirected sum matrix W_sum = [[0, 9, 0], [9, 0, 25], [0, 25, 0]] -> eigenvalues +-sqrt(706), 0.
    2. Directed target matrix W_dir = [[0, 8, 0], [1, 0, 17], [0, 8, 0]] -> exact eigenvalues +12, 0, -12.
    """
    res = gc.analyse(load("hocl"), {"weight": "Z"})
    expected_sum = np.array([[0.0, 9.0, 0.0], [9.0, 0.0, 25.0], [0.0, 25.0, 0.0]])
    expected_dir = np.array([[0.0, 8.0, 0.0], [1.0, 0.0, 17.0], [0.0, 8.0, 0.0]])
    assert np.allclose(res["_W"], expected_sum)
    assert np.allclose(res["_W_dir"], expected_dir)

    lam_sum = [math.sqrt(9**2 + 25**2), 0.0, -math.sqrt(9**2 + 25**2)]
    assert np.allclose(res["spectrum"]["eigenvalues"], lam_sum)
    assert res["spectrum"]["energy_abs"] == pytest.approx(2.0 * math.sqrt(9**2 + 25**2))

    assert np.allclose(res["spectrum_directed"]["eigenvalues"], [12.0, 0.0, -12.0])
    assert res["spectrum_directed"]["energy_abs"] == pytest.approx(24.0)


@pytest.mark.parametrize("weight,p_h,p_o,p_cl", [
    ("Z", 1.0, 8.0, 17.0),
    ("M", 1.0080, 15.999, 35.45),
    ("R", 0.31, 0.66, 1.02),
    ("E", 2.300, 3.610, 2.869),
    ("I", 13.5984, 13.6181, 12.9676),
])
def test_hocl_all_five_mirez_properties(weight, p_h, p_o, p_cl):
    res = gc.analyse(load("hocl"), {"weight": weight})
    W = res["_W"]
    Wd = res["_W_dir"]
    assert np.allclose(np.diag(W), 0.0) and np.allclose(np.diag(Wd), 0.0)
    # Undirected sums
    assert W[0, 1] == pytest.approx(p_h + p_o) and W[1, 0] == pytest.approx(p_h + p_o)
    assert W[1, 2] == pytest.approx(p_o + p_cl) and W[2, 1] == pytest.approx(p_o + p_cl)
    # Directed target weights
    assert Wd[0, 1] == pytest.approx(p_o) and Wd[1, 0] == pytest.approx(p_h)
    assert Wd[1, 2] == pytest.approx(p_cl) and Wd[2, 1] == pytest.approx(p_o)
    # Directed spectrum matches raw non-symmetric eigvals
    raw_dir = np.sort(np.linalg.eigvals(Wd).real)[::-1]
    assert np.allclose(res["spectrum_directed"]["eigenvalues"], raw_dir)


def test_supported_elements_and_valueerror():
    assert len(gc.SUPPORTED_ELEMENTS) == 16
    for el in ("S", "C", "H", "N", "O", "P"):
        assert el in gc.SUPPORTED_ELEMENTS
    for bad_el in ("Fe", "I", "Na", "Xe", "Au"):
        bad = gc.Frame("bad", [bad_el, "C"], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        with pytest.raises(ValueError, match="Unsupported element"):
            gc.analyse(bad)


@pytest.mark.parametrize("name,bonds", [("hocl", 2), ("benzene", 12), ("cubane", 20), ("adamantane", 28)])
def test_examples_bond_correctly(name, bonds):
    res = gc.analyse(load(name))
    assert len(res["bonds"]) == bonds
    assert res["issues"] == []
    assert len(res["fragments"]) == 1


def test_known_wiener_values():
    for name, w in (("benzene", 27), ("cubane", 48), ("adamantane", 96)):
        assert gc.analyse(load(name))["graphs"]["heavy"]["wiener"] == w


@pytest.mark.parametrize("name", ["hocl", "benzene", "cubane", "adamantane"])
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
    assert len(res["bonds"]) == 8
    assert len(res["issues"]) == 8
    assert any(not q["by_rule"] for q in res["near"])


def test_manual_add_and_remove():
    fr = load("benzene")
    res = gc.analyse(fr, {"remove": [[0, 1]], "add": [[0, 3]]})
    pairs = {(b[0], b[1]) for b in res["bonds"]}
    assert (0, 1) not in pairs and (0, 3) in pairs


def test_legacy_diagonal_z_reproduced():
    res = gc.analyse(load("adamantane"), {"diagonal": "Z"})
    sp = res["spectrum"]
    assert sp["energy_is_trace"]
    assert sp["energy_abs"] == pytest.approx(sum(res["Z"]))


def test_moments_and_zero_trace_identities():
    fr = load("cubane")
    res = gc.analyse(fr, {"kmax": 4, "weight": "Z"})
    W = res["_W"]
    Wd = res["_W_dir"]
    assert res["spectrum"]["trace"] == pytest.approx(0.0)
    assert res["spectrum_directed"]["trace"] == pytest.approx(0.0)
    for k, m in enumerate(res["spectrum"]["moments"], 1):
        assert m == pytest.approx(np.trace(np.linalg.matrix_power(W, k)), abs=1e-9)
    for k, m in enumerate(res["spectrum_directed"]["moments"], 1):
        assert m == pytest.approx(np.trace(np.linalg.matrix_power(Wd, k)), abs=1e-9)
    edge_sq = sum(b[5] ** 2 for b in res["bonds"])
    edge_prod = sum(b[6] * b[7] for b in res["bonds"])
    assert res["spectrum"]["moments"][1] == pytest.approx(2.0 * edge_sq)
    assert res["spectrum_directed"]["moments"][1] == pytest.approx(2.0 * edge_prod)


def test_parse_variants():
    text = "3\nwater\nO 0 0 0\n1 0.96 0 0\nh -0.24 0.93 0\n3\nframe 2\n8 0 0 0\nH 0.97 0 0\nH -0.24 0.93 0\n"
    frames = gc.parse_xyz(text)
    assert len(frames) == 2 and frames[0].symbols == ["O", "H", "H"]
    with pytest.raises(ValueError):
        gc.parse_xyz("5\nshort\nC 0 0 0\n")


def test_exports_include_undirected_and_directed():
    res = gc.analyse(load("hocl"))
    files = gc.export_files(res, "hocl", base=1)
    assert files["hocl_adjacency.csv"] == "0,9,0\n9,0,25\n0,25,0\n"
    assert files["hocl_directed_adjacency.csv"] == "0,8,0\n1,0,17\n0,8,0\n"
    assert "H1: ->O2 (8)" in files["hocl_directed_adjacency_list.txt"]
    assert "O2: ->H1 (1) ->Cl3 (17)" in files["hocl_directed_adjacency_list.txt"]


def test_bridge_round_trip():
    import bridge
    bridge.HERE = os.path.join(ROOT, "examples")
    assert json.loads(bridge.example("hocl"))["ok"]
    out = json.loads(bridge.analyse(0, json.dumps({"weight": "Z"})))
    assert out["ok"] and out["bonds"][0][5] == 9.0 and out["bonds"][0][6] == 8.0 and out["bonds"][0][7] == 1.0
    files = json.loads(bridge.exports(0, "{}", "hocl", 1))["files"]
    assert "hocl_directed_adjacency.csv" in files


def test_built_page_is_up_to_date():
    import build_web
    with open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8") as fh:
        assert fh.read() == build_web.build(), "run: python build_web.py"
