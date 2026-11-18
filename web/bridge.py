"""
bridge.py — connects the web page to graph_core. Every function takes plain strings and returns JSON text.
"""
import json
import os

import numpy as np

import graph_core as gc

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = {
    "hocl": "hocl.xyz",
    "benzene": "benzene.xyz",
    "cubane": "cubane.xyz",
    "adamantane": "adamantane.xyz",
}
_state = {"frames": None, "name": None, "cache": None}


def _ok(**kw):
    return json.dumps({"ok": True, **kw}, allow_nan=False, default=_default)


def _err(msg):
    return json.dumps({"ok": False, "error": str(msg)})


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def options():
    props = [
        {
            "key": k,
            "available": bool(p["values"]),
            **{f: p[f] for f in ("label", "short", "unit", "about")},
        }
        for k, p in gc.PROPERTIES.items()
    ]
    return _ok(
        defaults=gc.DEFAULT_SETTINGS,
        legacy=gc.LEGACY_PAIR_TABLE,
        radii=gc.COVALENT_RADII,
        default_radius=gc.DEFAULT_RADIUS,
        examples=list(EXAMPLES),
        supported_elements=list(gc.SUPPORTED_ELEMENTS),
        properties=props,
    )


def load(name, text):
    try:
        frames = gc.parse_xyz(text)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    _state.update(frames=frames, name=name, cache=None)
    return _ok(name=name, frames=[{"comment": f.comment, "n": f.n, "formula": gc.formula(f.symbols)} for f in frames])


def example(key):
    path = os.path.join(HERE, EXAMPLES[key])
    with open(path, encoding="utf-8") as fh:
        return load(EXAMPLES[key], fh.read())


def _result(frame, settings_json):
    frames = _state["frames"]
    if not frames:
        raise ValueError("Open an .xyz file first.")
    frame = max(0, min(int(frame), len(frames) - 1))
    settings = gc.merged_settings(json.loads(settings_json or "{}"))
    key = (frame, json.dumps(settings, sort_keys=True))
    if _state["cache"] and _state["cache"][0] == key:
        return _state["cache"][1]
    res = gc.analyse(frames[frame], settings)
    _state["cache"] = (key, res)
    return res


def _clean(v):
    """JSON can't hold inf/nan; send None instead."""
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def analyse(frame, settings_json, pair=""):
    try:
        res = _result(frame, settings_json)
        out = {k: v for k, v in res.items() if not k.startswith("_")}
        out["histogram"] = gc.histogram(res, res["settings"], pair or None)
        return _ok(**_clean(out))
    except Exception as e:  # noqa: BLE001
        return _err(e)


def histogram(frame, settings_json, pair=""):
    try:
        res = _result(frame, settings_json)
        return _ok(**_clean(gc.histogram(res, res["settings"], pair or None)))
    except Exception as e:  # noqa: BLE001
        return _err(e)


def distances(frame, settings_json, i):
    """Distances from atom i to every other atom (for the atom inspector)."""
    try:
        res = _result(frame, settings_json)
        i = int(i)
        return _ok(d=res["_D"][i].tolist(), cut=res["_C"][i].tolist())
    except Exception as e:  # noqa: BLE001
        return _err(e)


def exports(frame, settings_json, stem, base):
    try:
        res = _result(frame, settings_json)
        files = gc.export_files(res, stem or "molecule", int(base))
        return _ok(files=files)
    except Exception as e:  # noqa: BLE001
        return _err(e)
