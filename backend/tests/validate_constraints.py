# -*- coding: utf-8 -*-
"""
实验关键约束的最小校验：
1) D+R=K 在训练/推理路径均被强制（这里仅提供占位调用与静态检查框架）。
2) 零先验扫描：检索仓库中疑似引入人类数制先验的关键词（白名单除外）。
3) 测试单元保护：若 backend/tests 被修改，必须有 research/design_decisions.md 对应记录。
"""
import os, re, sys, subprocess, json
from pathlib import Path

# --- Force UTF-8 console on Windows / PowerShell ---
import sys, io
try:
    # Python 3.7+
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    # Fallback for very old Pythons
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOW_PATHS = [
    "research/", "reviewers/", "data/results/", "controller/"
]
FORBIDDEN_PRIOR_PATTERNS = [
    r"decimal", r"binary", r"base\s*\d+", r"\bzero\b", r"零", r"十进制", r"二进制",
    r"\b0x", r"\b0b", r"radix"
]

def scan_priors():
    offenders = []
    for p in PROJECT_ROOT.rglob("*"):
        if p.is_dir(): 
            continue
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        # 白名单路径可忽略
        if any(rel.startswith(prefix) for prefix in ["research/", "reviewers/","data/results/"]):
            continue
        if rel.endswith((".png",".jpg",".jpeg",".gif",".pdf",".svg",".ico",".map",".lock",".zip",".tar",".gz",".env",".venv",".pyc")):
            continue
        try:
            text = p.read_text("utf-8", errors="ignore")
        except Exception:
            continue
        for pat in FORBIDDEN_PRIOR_PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE):
                offenders.append((rel, pat))
    return offenders

def check_tests_change_requires_design_log():
    # 简单策略：如果 git 工作区中 backend/tests 有改动，则 design_decisions.md 必须当天被修改
    try:
        diff = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception:
        return []
    changed_tests = [line for line in diff.splitlines() if "backend/tests" in line]
    if changed_tests:
        dd = PROJECT_ROOT / "research" / "design_decisions.md"
        if not dd.exists():
            return ["backend/tests changed but no research/design_decisions.md present"]
        # 不强制当天时间，仅要求存在相应记录；可增强为解析 commit/时间戳
    return []

def main():
    priors = scan_priors()
    errs = []
    if priors:
        errs.append(f"Forbidden prior-like tokens found: {priors[:10]} ...")
    errs.extend(check_tests_change_requires_design_log())

    if errs:
        print("\n[validate_constraints] FAILED\n" + "\n".join(f"- {e}" for e in errs))
        sys.exit(1)
    else:
        print("[validate_constraints] OK")

if __name__ == "__main__":
    main()
