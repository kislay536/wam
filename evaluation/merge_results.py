"""
Aggregate eval results per task and output eval_summary.json.
Supports robocasa (eval_results_*.json) and robotwin (_result.txt + mp4 files), auto-detected.

Usage:
    python scripts/merge_results.py <save_root_dir>
"""

import os
import sys
import json
import glob
from collections import defaultdict


def detect_dataset_type(root_dir: str) -> str:
    if glob.glob(os.path.join(root_dir, "eval_results_*.json")):
        return "robocasa"
    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    for d in subdirs:
        if os.path.isfile(os.path.join(root_dir, d, "_result.txt")):
            return "robotwin"
    return "robocasa"


def aggregate_robocasa(root_dir: str) -> dict:
    json_files = sorted(glob.glob(os.path.join(root_dir, "eval_results_*.json")))
    if not json_files:
        print(f"[merge_results] No eval_results_*.json found in {root_dir}")
        return {}

    print(f"[merge_results] Found {len(json_files)} result files, aggregating...")

    task_stats: dict = defaultdict(lambda: {"num_success_rollouts": 0, "num_rollouts": 0})

    for jf in json_files:
        with open(jf) as f:
            data: dict = json.load(f)
        for task_name, stats in data.items():
            task_stats[task_name]["num_success_rollouts"] += stats["num_success_rollouts"]
            task_stats[task_name]["num_rollouts"] += stats["num_rollouts"]

    summary: dict = {}
    total_success = 0
    total_rollouts = 0

    for task_name in sorted(task_stats.keys()):
        s = task_stats[task_name]
        rate = s["num_success_rollouts"] / s["num_rollouts"] if s["num_rollouts"] > 0 else 0.0
        summary[task_name] = {
            "num_success_rollouts": s["num_success_rollouts"],
            "num_rollouts": s["num_rollouts"],
            "success_rate": round(rate, 4),
        }
        total_success += s["num_success_rollouts"]
        total_rollouts += s["num_rollouts"]

    overall_rate = total_success / total_rollouts if total_rollouts > 0 else 0.0
    summary["__overall__"] = {
        "num_success_rollouts": total_success,
        "num_rollouts": total_rollouts,
        "success_rate": round(overall_rate, 4),
    }
    return summary


def aggregate_robotwin(root_dir: str) -> dict:
    subdirs = sorted([d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
    task_dirs = [d for d in subdirs if os.path.isfile(os.path.join(root_dir, d, "_result.txt"))]

    if not task_dirs:
        print(f"[merge_results] No task directories with _result.txt found in {root_dir}")
        return {}

    print(f"[merge_results] Found {len(task_dirs)} task directories, aggregating...")

    summary: dict = {}
    total_success = 0
    total_rollouts = 0

    for task_name in task_dirs:
        task_path = os.path.join(root_dir, task_name)
        mp4_files = glob.glob(os.path.join(task_path, "*.mp4"))
        num_rollouts = len(mp4_files)
        num_success = sum(1 for f in mp4_files if "_success.mp4" in f)

        with open(os.path.join(task_path, "_result.txt")) as f:
            lines = f.read().strip().splitlines()
        success_rate = float(lines[-1])

        summary[task_name] = {
            "num_success_rollouts": num_success,
            "num_rollouts": num_rollouts,
            "success_rate": round(success_rate, 4),
        }
        total_success += num_success
        total_rollouts += num_rollouts

    overall_rate = total_success / total_rollouts if total_rollouts > 0 else 0.0
    summary["__overall__"] = {
        "num_success_rollouts": total_success,
        "num_rollouts": total_rollouts,
        "success_rate": round(overall_rate, 4),
    }
    return summary


def aggregate_jsons(root_dir: str, output_name: str = "eval_summary.json") -> None:
    dataset_type = detect_dataset_type(root_dir)
    print(f"[merge_results] Detected dataset type: {dataset_type}")

    if dataset_type == "robocasa":
        summary = aggregate_robocasa(root_dir)
    else:
        summary = aggregate_robotwin(root_dir)

    if not summary:
        return

    output_path = os.path.join(root_dir, output_name)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    overall = summary.get("__overall__", {})
    overall_rate = overall.get("success_rate", 0.0)
    total_success = overall.get("num_success_rollouts", 0)
    total_rollouts = overall.get("num_rollouts", 0)

    print(f"[merge_results] Summary saved to: {output_path}")
    print(f"[merge_results] Overall success rate: {overall_rate:.4f}  ({total_success}/{total_rollouts})")
    print()
    for task, s in summary.items():
        if task == "__overall__":
            continue
        print(f"  {task:<35s}  SR={s['success_rate']:.4f}  ({s['num_success_rollouts']}/{s['num_rollouts']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_results.py <save_root_dir>")
        sys.exit(1)
    aggregate_jsons(sys.argv[1])
