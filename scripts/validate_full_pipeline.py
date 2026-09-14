# -*- coding: utf-8 -*-
"""
端到端自检（占位）：
- 尝试导入后端关键模块
- 如存在 run_experiment.py，则 dry-run 一次
- 如存在前端契约，则做字段校验
"""
import os, sys, importlib.util, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

def try_import(mod_path):
    if not mod_path.exists():
        return False, f"missing: {mod_path}"
    spec = importlib.util.spec_from_file_location("mod", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
        return True, "ok"
    except Exception as e:
        return False, str(e)

def main():
    ok = True
    paths = [
        ROOT/"backend"/"core"/"memory.py",
        ROOT/"backend"/"core"/"agent_u.py",
        ROOT/"backend"/"scripts"/"run_experiment.py",
        ROOT/"backend"/"config"/"frontend_contract.json",
    ]
    for p in paths:
        exists = p.exists()
        print(f"[check] {p}: {'OK' if exists else 'MISS'}")
        ok &= exists

    # dry-run
    runpy = ROOT/"backend"/"scripts"/"run_experiment.py"
    if runpy.exists():
        try:
            subprocess.check_call([sys.executable, str(runpy), "--dry-run"])
            print("[dry-run] OK")
        except Exception as e:
            ok = False
            print(f"[dry-run] FAIL: {e}")

    # contract basic schema
    contract = ROOT/"backend"/"config"/"frontend_contract.json"
    if contract.exists():
        try:
            obj = json.loads(contract.read_text("utf-8"))
            assert "endpoints" in obj and isinstance(obj["endpoints"], list)
            print("[contract] OK")
        except Exception as e:
            ok = False
            print(f"[contract] FAIL: {e}")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
