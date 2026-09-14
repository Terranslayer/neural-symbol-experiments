# -*- coding: utf-8 -*-
"""List all run.jsonl files with full architecture description, sorted by mtime."""
import json, os, glob, sys
from datetime import datetime

root = sys.argv[1] if len(sys.argv) > 1 else "logs"
runs = []
for f in glob.glob(os.path.join(root, "**/run.jsonl"), recursive=True):
    try:
        with open(f) as fh:
            lines = fh.readlines()
        if not lines:
            continue
        cfg = json.loads(lines[0])
        a = cfg["agent_cfg"]
        s = cfg["scene_cfg"]
        cur = cfg["curriculum"]
        final_acc = None
        comp = None
        near_ext = None
        far_ext = None
        for ln in lines:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if rec.get("event") == "final_metrics":
                final_acc = rec.get("rollout_accuracy")
                comp = rec.get("composition_index")
                ex = rec.get("extrapolation_full", {})
                near_ext = ex.get("near_extrap")
                far_ext = ex.get("far_extrap")
        mt = os.path.getmtime(f)
        ab = []
        if a.get("read_cnn_kernel", 0) > 0:
            ab.append("CNN_k%d_L%d" % (a["read_cnn_kernel"], a.get("read_cnn_layers", 1)))
        else:
            ab.append("noCNN")
        ab.append("d%d" % a["d_model"])
        if a.get("cell_type", "gru") != "gru":
            ab.append(a["cell_type"])
        if a.get("inner_iter", 1) > 1:
            ab.append("iter%d" % a["inner_iter"])
        if a.get("hidden_noise_std", 0) > 0:
            ab.append("noise%.2f" % a["hidden_noise_std"])
        if a.get("scratch_attn"):
            ab.append("scratchAttn")
        if a.get("gaussian_attn"):
            ab.append("gaussAttn")
        if a.get("pfc_layers", 0) > 0:
            ab.append("PFC_L%d_h%d_iter%d" % (a["pfc_layers"], a["pfc_heads"], a.get("pfc_iter", 1)))
        if a.get("reset_h_at_compare_block"):
            ab.append("resetH")
        if a.get("pfc_split_ab"):
            ab.append("splitAB")
        if a.get("pfc_at_write"):
            ab.append("PFCwrite")
        if a.get("pfc_compare_drop_raw_scratch"):
            ab.append("dropRaw")
        if a.get("compare_mlp_hidden"):
            ab.append("cmpMLP%d" % a["compare_mlp_hidden"])
        if a.get("quantize_levels"):
            ab.append("q%d_%s" % (a["quantize_levels"], a.get("quantize_range", "?")))
        sb = []
        if s.get("complex_world"):
            sb.append("cplx")
        sb.append("L%d" % s["L"])
        sb.append("K%d" % s["K"])
        eq = s.get("equal_weight", 0)
        nw = s.get("near_weight", 0)
        fw = s.get("far_weight", 0)
        nr = s.get("near_range", [0, 0])
        if abs(eq - 1.0 / 3.0) < 0.01 and abs(fw) < 0.01:
            sb.append("trio")
        elif abs(nw - 0.5) < 0.01 and nr[0] == 1 and nr[1] == 1:
            sb.append("discrim")
        else:
            sb.append("eq%.2f_n%.2f_f%.2f" % (eq, nw, fw))
        if s.get("alpha_distribution") == "log_uniform":
            sb.append("logA")
        cur_str = ",".join(str(c["n_max"]) for c in cur)
        cur_ep = "/".join(str(c["max_epochs"]) for c in cur)
        params = cfg.get("num_parameters", "?")
        runs.append((mt, f, " ".join(ab), " ".join(sb), cur_str, cur_ep, params, final_acc, comp, near_ext, far_ext))
    except Exception:
        pass

runs.sort()
for mt, f, a, s, cur, cur_ep, p, fa, c, ne, fe in runs:
    ts = datetime.fromtimestamp(mt).strftime("%m-%d %H:%M")
    fa_s = ("%.3f" % fa) if fa is not None else "  -  "
    c_s = ("%.3f" % c) if c is not None else "  -  "
    ne_s = ("%.2f" % ne) if ne is not None else " - "
    short = f.replace("logs/", "").replace("/run.jsonl", "")
    print("%s  %-50s p=%-6s acc=%s comp=%s ne=%s cur=[%-14s]ep[%s]" % (ts, short, str(p), fa_s, c_s, ne_s, cur, cur_ep))
    print("           arch:  %s" % a)
    print("           scene: %s" % s)
