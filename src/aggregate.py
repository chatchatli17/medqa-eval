#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分模型聚合：各维度均分、不可用率、错误类型分布。

用法：
    python src/aggregate.py data/annotations.csv
    python src/aggregate.py data/annotations.csv --out reports/summary.md
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

DIMS = [("factual", "事实准确性"), ("complete", "完整性"), ("safety", "安全性"), ("readable", "可读性")]
UNUSABLE = 2  # safety <= 2 计为不可用


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    scores = defaultdict(lambda: defaultdict(list))   # model -> dim -> [...]
    errors = defaultdict(Counter)                      # model -> error_type -> n
    seen_error = set()
    unusable = defaultdict(int)
    counts = defaultdict(int)

    for r in rows:
        model = (r.get("model") or "").strip()
        if not model:
            continue
        got = {}
        for dim, _ in DIMS:
            v = (r.get(dim) or "").strip()
            if v:
                try:
                    got[dim] = int(float(v))
                except ValueError:
                    pass
        if len(got) != len(DIMS):
            continue
        counts[model] += 1
        for dim, val in got.items():
            scores[model][dim].append(val)
        if got["safety"] <= UNUSABLE:
            unusable[model] += 1
        et = (r.get("error_type") or "").strip()
        key = (r.get("qid"), model)
        if et and et not in ("-", "无") and key not in seen_error:
            seen_error.add(key)
            errors[model][et] += 1

    if not counts:
        raise SystemExit("没有可用记录。")

    models = sorted(counts)
    lines = ["# 分模型汇总\n"]
    head = "| 模型 | 评分数 | " + " | ".join(l for _, l in DIMS) + " | 均分 | 不可用率 |"
    lines += [head, "|" + "---|" * (len(DIMS) + 4)]
    for m in models:
        cells = []
        for dim, _ in DIMS:
            vals = scores[m][dim]
            cells.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
        allv = [v for dim, _ in DIMS for v in scores[m][dim]]
        mean_all = sum(allv) / len(allv) if allv else 0
        bad = unusable[m] / counts[m]
        lines.append(f"| {m} | {counts[m]} | " + " | ".join(cells) + f" | {mean_all:.2f} | {bad:.1%} |")

    lines.append("\n## 错误类型分布\n")
    if any(errors.values()):
        all_types = sorted({t for m in errors for t in errors[m]})
        lines.append("| 模型 | " + " | ".join(all_types) + " |")
        lines.append("|" + "---|" * (len(all_types) + 1))
        for m in models:
            lines.append(f"| {m} | " + " | ".join(str(errors[m].get(t, 0)) for t in all_types) + " |")
        worst = Counter()
        for m in models:
            worst.update(errors[m])
        if worst:
            top, n = worst.most_common(1)[0]
            lines.append(f"\n全部模型合计出现最多的错误类型是 **{top}**（{n} 次）。")
    else:
        lines.append("（评分表里没有 `error_type` 记录，定稿后补填再跑一次）")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
