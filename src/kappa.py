#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""盲评一致性检验：逐维度计算加权 Cohen's kappa。

用法：
    python src/kappa.py data/annotations.csv
    python src/kappa.py data/annotations.csv --out reports/agreement.md

多标注员时，按两两组合取均值。
"""

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

DIMS = [("factual", "事实准确性"), ("complete", "完整性"), ("safety", "安全性"), ("readable", "可读性")]
K = 5  # 量表级数 1..5


def load(path):
    """读入评分表，只保留四个维度齐全的记录。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    records, skipped = [], 0
    for r in rows:
        qid = (r.get("qid") or "").strip()
        model = (r.get("model") or "").strip()
        who = (r.get("annotator") or "").strip()
        if not (qid and model and who):
            skipped += 1
            continue
        rec = {"qid": qid, "model": model, "annotator": who}
        ok = True
        for dim, _ in DIMS:
            v = (r.get(dim) or "").strip()
            if not v:
                ok = False
                break
            try:
                score = int(float(v))
            except ValueError:
                ok = False
                break
            if not 1 <= score <= K:
                ok = False
                break
            rec[dim] = score
        if ok:
            records.append(rec)
        else:
            skipped += 1
    return records, skipped


def weighted_kappa(a, b):
    """线性加权的 Cohen's kappa（序数量表，5 级）。"""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0] * K for _ in range(K)]
    for x, y in zip(a, b):
        obs[x - 1][y - 1] += 1
    w = [[abs(i - j) / (K - 1) for j in range(K)] for i in range(K)]
    row = [sum(obs[i]) for i in range(K)]
    col = [sum(obs[i][j] for i in range(K)) for j in range(K)]
    po = sum(w[i][j] * obs[i][j] for i in range(K) for j in range(K)) / n
    pe = sum(w[i][j] * row[i] * col[j] for i in range(K) for j in range(K)) / (n * n)
    if pe == 0:
        return float("nan")
    return 1 - po / pe


def interpret(k):
    """Landis & Koch 解释标准。"""
    if k != k:  # NaN
        return "无法计算"
    if k < 0:
        return "差于随机"
    if k < 0.20:
        return "极弱"
    if k < 0.40:
        return "一般"
    if k < 0.60:
        return "中等"
    if k < 0.80:
        return "较强"
    return "强"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default=None, help="把结果写成 markdown 表格")
    args = ap.parse_args()

    records, skipped = load(args.csv_path)
    if not records:
        raise SystemExit("没有有效记录：检查列名 qid/model/annotator/factual/complete/safety/readable。")

    by_item = defaultdict(dict)
    for r in records:
        by_item[(r["qid"], r["model"])][r["annotator"]] = r

    annotators = sorted({r["annotator"] for r in records})
    paired = {k: v for k, v in by_item.items() if len(v) >= 2}
    both = {k: v for k, v in by_item.items() if len(v) == len(annotators)}

    lines = []
    lines.append("# 一致性检验报告\n")
    lines.append(f"- 评分文件：`{args.csv_path}`")
    lines.append(f"- 标注员：{', '.join(annotators)}（{len(annotators)} 人）")
    lines.append(f"- 有效评分记录：{len(records)} 条，跳过 {skipped} 条不完整记录")
    lines.append(f"- 被两名以上标注员共同评过的条目：{len(paired)} 条")
    if len(both) != len(by_item):
        lines.append(f"- 注意：{len(by_item) - len(both)} 条未做到全员覆盖，kappa 只用两两重叠部分")
    lines.append("")
    lines.append("| 维度 | 对比条目 | 完全一致率 | 加权 κ | 解释 |")
    lines.append("|---|---|---|---|---|")

    summary = {}
    for dim, label in DIMS:
        ks, agree, n_pairs = [], [], []
        # 按标注员两两配对，批量算 kappa（小样本逐条算没有意义）
        per_pair = defaultdict(lambda: {"x": [], "y": []})
        for item, per in paired.items():
            for a, b in itertools.combinations(sorted(per), 2):
                per_pair[(a, b)]["x"].append(per[a][dim])
                per_pair[(a, b)]["y"].append(per[b][dim])
        for (a, b), d in per_pair.items():
            ks.append(weighted_kappa(d["x"], d["y"]))
            agree.append(sum(1 for x, y in zip(d["x"], d["y"]) if x == y) / len(d["x"]))
            n_pairs.append(len(d["x"]))
        if not ks:
            lines.append(f"| {label} | 0 | - | - | 无重叠评分 |")
            continue
        k_mean = sum(ks) / len(ks)
        a_mean = sum(agree) / len(agree)
        summary[dim] = (k_mean, a_mean)
        lines.append(f"| {label} | {sum(n_pairs)} | {a_mean:.1%} | {k_mean:.3f} | {interpret(k_mean)} |")

    if summary:
        overall = sum(v[0] for v in summary.values()) / len(summary)
        lines.append(f"\n**四维平均 κ = {overall:.3f}**（{interpret(overall)}）")
        worst = min(summary.items(), key=lambda kv: kv[1][0])
        lines.append(
            f"\n口径最弱的是 **{dict(DIMS)[worst[0]]}**（κ = {worst[1][0]:.3f}），"
            "建议回到 `docs/标注手册.md` 补充该维度的边界样例后重标分歧条目。"
        )

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
