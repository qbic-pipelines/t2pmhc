#!/usr/bin/env python3
"""
Analyse distance CV experiment logs and produce a summary CSV.

For each model (GAT/GCN) and distance threshold, collects:
  - AUC (mean ± std from 5-fold CV training)
  - Train runtime per CV fold (median of completed folds)
  - Graph generation total runtime
  - Max memory usage (from sacct, separately for train and graph_gen)
  - GPU type (from sacct, train jobs only)

Logs may be partial (train job resumed mid-CV) — the script tolerates this by
computing per-fold runtimes from log timestamps and taking the median.

Logs are copied to the project folder and paths reported in the CSV.
"""

import re
import csv
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from statistics import median, mean, stdev

PROJECT_DIR = Path("/cluster/projects/2026_dego_NF_DB/distance_CV")
LOG_LIST    = PROJECT_DIR / "list_of_logs.txt"
OUT_CSV     = PROJECT_DIR / "results_summary.csv"
LOGS_DEST   = PROJECT_DIR / "logs"
GRAPHS_DIR  = PROJECT_DIR / "generated_graphs"


# ── log-list parsing ──────────────────────────────────────────────────────────

def parse_log_list(path: Path):
    """Return {model -> {"train": {dist: [log_path]}, "graph_gen": {dist: log_path}}}"""
    sections = {"gat": {"train": {}, "graph_gen": {}}, "gcn": {"train": {}, "graph_gen": {}}}
    current_model = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "GAT:":
            current_model = "gat"
            continue
        if line.upper() == "GCN:":
            current_model = "gcn"
            continue
        if current_model is None:
            continue

        p = Path(line)
        name = p.name

        # extract distance threshold:
        # train logs: dist14  /  graph_gen logs: _gat_14 or _gcn_14
        m = re.search(r"dist(\d+)", name) or re.search(r"_(?:gat|gcn)_(\d+)\.", name)
        if not m:
            print(f"  [WARN] cannot parse distance from: {name}")
            continue
        dist = int(m.group(1))

        if "create" in name or "graph" in name:
            sections[current_model]["graph_gen"][dist] = p
        else:
            sections[current_model]["train"].setdefault(dist, []).append(p)

    return sections


# ── log copying ───────────────────────────────────────────────────────────────

def copy_log(src: Path, dest_dir: Path) -> Path:
    """Copy src to dest_dir (skip if already there). Return destination path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


# ── log content parsing ───────────────────────────────────────────────────────

_TS_RE         = re.compile(r"\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]")
_FOLD_START_RE = re.compile(r"---- Fold (\d+)/(\d+) ----")
_FOLD_AUC_RE   = re.compile(r"Fold (\d+) AUROC:\s*([\d.]+)")
_CV_AUC_RE     = re.compile(r"(\d+)-fold CV AUROC:\s*([\d.]+)\s*[±+]\s*([\d.]+)")
_DONE_RE       = re.compile(r"done")


def _parse_ts(line: str) -> datetime | None:
    m = _TS_RE.search(line)
    if m:
        return datetime.strptime(m.group(1), "%m/%d/%y %H:%M:%S")
    return None


def parse_train_logs(log_paths: list[Path]):
    """
    Parse one or more train log files (possibly partial continuations).

    Returns:
        cv_auc, cv_std, fold_aucs, cv_fold_runtimes_s, warnings
    """
    fold_start_ts: dict[int, datetime] = {}
    fold_aucs: list[tuple[int, float]] = []
    cv_fold_runtimes_s: list[float] = []
    cv_auc = None
    cv_std = None
    warnings: list[str] = []
    last_ts: datetime | None = None

    for lp in log_paths:
        if not lp.exists():
            warnings.append(f"Missing log: {lp}")
            continue
        text = lp.read_text(errors="replace")

        for line in text.splitlines():
            ts = _parse_ts(line)
            if ts:
                last_ts = ts

            m = _FOLD_START_RE.search(line)
            if m:
                fold_num = int(m.group(1))
                if last_ts:
                    fold_start_ts[fold_num] = last_ts
                continue

            m = _FOLD_AUC_RE.search(line)
            if m:
                fold_num = int(m.group(1))
                auc = float(m.group(2))
                fold_aucs.append((fold_num, auc))
                if fold_num in fold_start_ts and last_ts:
                    delta = (last_ts - fold_start_ts[fold_num]).total_seconds()
                    cv_fold_runtimes_s.append(delta)
                continue

            m = _CV_AUC_RE.search(line)
            if m:
                cv_auc = float(m.group(2))
                cv_std = float(m.group(3))

    # de-duplicate fold AUCs (keep last occurrence per fold, in case of reruns)
    seen: dict[int, float] = {}
    for fold_num, auc in fold_aucs:
        seen[fold_num] = auc
    fold_aucs = sorted(seen.items())

    # If CV summary line is missing (partial run) compute from available folds
    if cv_auc is None and len(fold_aucs) >= 2:
        aucs = [a for _, a in fold_aucs]
        cv_auc = mean(aucs)
        cv_std = stdev(aucs) if len(aucs) > 1 else 0.0
        warnings.append(
            f"CV summary line missing — computed from {len(aucs)} folds: "
            f"{cv_auc:.4f} ± {cv_std:.4f}"
        )

    if not fold_aucs:
        warnings.append("No fold AUC lines found — training may not have started")

    return cv_auc, cv_std, fold_aucs, cv_fold_runtimes_s, warnings


def parse_graph_gen_log(log_path: Path):
    """Parse a graph generation log and return total runtime in seconds."""
    if not log_path.exists():
        return None, [f"Missing log: {log_path}"]

    warnings: list[str] = []
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    first_ts = last_ts = None
    for line in lines:
        ts = _parse_ts(line)
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

    # tighten end timestamp to the "done" line
    for line in reversed(lines):
        if _DONE_RE.search(line):
            ts = _parse_ts(line)
            if ts:
                last_ts = ts
            break

    if first_ts and last_ts:
        runtime_s = (last_ts - first_ts).total_seconds()
    else:
        runtime_s = None
        warnings.append("Could not determine graph_gen runtime from timestamps")

    return runtime_s, warnings


# ── sacct helpers ─────────────────────────────────────────────────────────────

def sacct_job(job_id: int):
    """Return (max_rss_gb, gpu_type) from sacct — either may be None."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(job_id),
             "--format=JobID,MaxRSS,AllocTRES%200",
             "--noheader", "--parsable2"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None

    max_rss_bytes = None
    gpu_type = None

    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id_field, max_rss_field, alloc_tres_field = parts[0], parts[1], parts[2]

        if job_id_field.endswith(".batch") and max_rss_field:
            val = max_rss_field.strip()
            if val.endswith("K"):
                max_rss_bytes = float(val[:-1]) * 1024
            elif val.endswith("M"):
                max_rss_bytes = float(val[:-1]) * 1024 ** 2
            elif val.endswith("G"):
                max_rss_bytes = float(val[:-1]) * 1024 ** 3
            else:
                try:
                    max_rss_bytes = float(val)
                except ValueError:
                    pass

        if not gpu_type and alloc_tres_field:
            m = re.search(r"gres/gpu:([^,=]+)=", alloc_tres_field)
            if m:
                gpu_type = m.group(1)

    max_rss_gb = round(max_rss_bytes / 1024 ** 3, 2) if max_rss_bytes else None
    return max_rss_gb, gpu_type


def job_id_from_path(p: Path) -> int | None:
    m = re.match(r"(\d+)_", p.name)
    return int(m.group(1)) if m else None


def rel(path: Path) -> str:
    """Return path relative to PROJECT_DIR for portable CSV output."""
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    sections = parse_log_list(LOG_LIST)
    rows = []

    for model in ("gat", "gcn"):
        train_logs    = sections[model]["train"]
        graph_gen_logs = sections[model]["graph_gen"]
        all_dists = sorted(set(list(train_logs.keys()) + list(graph_gen_logs.keys())))

        for dist in all_dists:
            print(f"\n{'='*60}")
            print(f"  Model={model.upper()}  dist={dist}")
            print(f"{'='*60}")

            row = {
                "model": model.upper(),
                "distance_threshold": dist,
                "cv_auc": "",
                "cv_auc_std": "",
                "n_cv_folds_completed": "",
                "train_cv_fold_runtime_max_hms": "",
                "train_cv_fold_runtime_max_s": "",
                "train_max_rss_gb": "",
                "train_gpu_type": "",
                "train_log_paths": "",
                "graph_gen_runtime_hms": "",
                "graph_gen_runtime_s": "",
                "graph_gen_max_rss_gb": "",
                "graph_gen_log_path": "",
                "graph_pt_path": "",
                "notes": "",
            }

            notes = []

            # ── training logs ──────────────────────────────────────────────
            if dist in train_logs:
                orig_paths = train_logs[dist]
                copied_paths = [copy_log(p, LOGS_DEST) for p in orig_paths if p.exists()]
                print(f"  Train logs: {[p.name for p in orig_paths]}")

                cv_auc, cv_std, fold_aucs, cv_fold_runtimes, warnings = parse_train_logs(orig_paths)

                for w in warnings:
                    print(f"    [WARN] {w}")
                    notes.append(w)

                if cv_auc is not None:
                    row["cv_auc"] = f"{cv_auc:.4f}"
                    row["cv_auc_std"] = f"{cv_std:.4f}"
                    print(f"    CV AUROC: {cv_auc:.4f} ± {cv_std:.4f}")
                else:
                    print("    CV AUROC: NOT FOUND")

                row["n_cv_folds_completed"] = len(fold_aucs)
                print(f"    Folds with AUC: {[f'{fn}:{a:.4f}' for fn, a in fold_aucs]}")

                if cv_fold_runtimes:
                    best = max(cv_fold_runtimes)
                    row["train_cv_fold_runtime_max_s"] = round(best)
                    row["train_cv_fold_runtime_max_hms"] = fmt_duration(best)
                    print(f"    CV fold runtime (max): {fmt_duration(best)}  ({len(cv_fold_runtimes)} measured)")
                else:
                    print("    CV fold runtime: NOT AVAILABLE")

                all_rss, all_gpu = [], []
                for lp in orig_paths:
                    jid = job_id_from_path(lp)
                    if jid:
                        rss, gpu = sacct_job(jid)
                        print(f"    sacct {jid}: MaxRSS={rss} GB, GPU={gpu}")
                        if rss:
                            all_rss.append(rss)
                        if gpu:
                            all_gpu.append(gpu)

                if all_rss:
                    row["train_max_rss_gb"] = max(all_rss)
                if all_gpu:
                    row["train_gpu_type"] = ", ".join(dict.fromkeys(all_gpu))

                row["train_log_paths"] = "; ".join(rel(p) for p in copied_paths)

            else:
                print("  No train logs for this distance.")
                notes.append("No train logs")

            # ── graph generation log ───────────────────────────────────────
            if dist in graph_gen_logs:
                orig_lp = graph_gen_logs[dist]
                copied_lp = copy_log(orig_lp, LOGS_DEST) if orig_lp.exists() else orig_lp
                print(f"  Graph gen log: {orig_lp.name}")

                runtime_s, warnings = parse_graph_gen_log(orig_lp)

                for w in warnings:
                    print(f"    [WARN] {w}")
                    notes.append(w)

                if runtime_s is not None:
                    row["graph_gen_runtime_s"] = round(runtime_s)
                    row["graph_gen_runtime_hms"] = fmt_duration(runtime_s)
                    print(f"    Graph gen runtime: {fmt_duration(runtime_s)}")

                jid = job_id_from_path(orig_lp)
                if jid:
                    rss, _ = sacct_job(jid)
                    print(f"    sacct {jid}: MaxRSS={rss} GB")
                    if rss:
                        row["graph_gen_max_rss_gb"] = rss

                row["graph_gen_log_path"] = rel(copied_lp)

            else:
                print("  No graph gen log for this distance.")
                notes.append("No graph gen log")

            # ── graph .pt path ─────────────────────────────────────────────
            pt_name = f"t2pmhc_{model}_dist{dist}.pt"
            pt_path = GRAPHS_DIR / pt_name
            if pt_path.exists():
                row["graph_pt_path"] = rel(pt_path)
            else:
                notes.append(f"Graph .pt not found: {pt_path}")
                print(f"  [WARN] Graph .pt not found: {pt_path}")

            row["notes"] = "; ".join(notes)
            rows.append(row)

    # ── write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "model",
        "distance_threshold",
        "cv_auc",
        "cv_auc_std",
        "n_cv_folds_completed",
        "train_cv_fold_runtime_max_hms",
        "train_cv_fold_runtime_max_s",
        "train_max_rss_gb",
        "train_gpu_type",
        "train_log_paths",
        "graph_gen_runtime_hms",
        "graph_gen_runtime_s",
        "graph_gen_max_rss_gb",
        "graph_gen_log_path",
        "graph_pt_path",
        "notes",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\nResults written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
