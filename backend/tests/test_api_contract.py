# -*- coding: utf-8 -*-
import json, pathlib

def test_contract_exists_and_minimal_fields():
    p = pathlib.Path("backend/config/frontend_contract.json")
    assert p.exists(), "frontend_contract.json missing"
    obj = json.loads(p.read_text("utf-8"))
    eps = { (e["method"], e["path"]) for e in obj.get("endpoints", []) }
    expected = {
        ("POST","/env/reset"),
        ("POST","/agent/allocate"),
        ("POST","/agent/observe"),
        ("POST","/agent/encode"),
        ("POST","/agent/decode"),
        ("GET", "/eval/metrics")
    }
    assert expected.issubset(eps), f"missing endpoints: {expected - eps}"
