# MABS v3 (SLEM) results summary

## Honest goal

Not beat Higgott–Gidney absolute µs in pure Python.
Goal: **Pareto-dominate full PyMatching Sparse Blossom** on mean stage time
for sparse Katoch workloads, with **LER matching batch MWPM**.

## Before / after (stage vs stream_w3d)

| d | p | stream_w3d | mabs_v3 | speedup | escalate | LER v3 | LER batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.001 | 21610 | 14539 | **1.49×** | 0.880 | 0.0005 | 0.0005 |
| 7 | 0.001 | 42846 | 27124 | **1.58×** | 0.998 | 0 | 0 |
| 5 | 0.002 | 25991 | 19132 | **1.36×** | 0.986 | 0.011 | 0.011 |
| 7 | 0.002 | 59118 | 40391 | **1.46×** | 1.000 | 0.00125 | 0.00125 |

Full private repo: https://github.com/HellWithCode0/mabs
