#!/usr/bin/env python3
"""python3 compare.py results_seq.csv results_omp.csv [results_cuda.csv]  -> speed-up + quality table"""
import csv, sys
rows = {}
for path in sys.argv[1:]:
    tag = path.split('results_')[-1].replace('.csv', '')
    for r in csv.DictReader(open(path)):
        rows.setdefault(int(r['fid']), {})[tag] = r
tags = [p.split('results_')[-1].replace('.csv', '') for p in sys.argv[1:]]
base = tags[0]
print(f"{'F':>3} {'name':<11}" + "".join(f"{t+' mean err':>18}{t+' t_opt':>12}{'speedup':>9}" for t in tags))
for fid, d in sorted(rows.items()):
    line = f"{fid:>3} {d[base]['name']:<11}"
    for t in tags:
        sp = float(d[base]['t_opt']) / max(float(d[t]['t_opt']), 1e-12)
        line += f"{float(d[t]['mean_err']):>18.4e}{float(d[t]['t_opt']):>12.3f}{sp:>8.2f}x"
    print(line)
