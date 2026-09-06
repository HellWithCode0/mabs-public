# MABS v2 benchmark summary

**Streaming fix (v2.1):** truncated per-window DEMs with past-drop /
future-truncate + carry-forward. Closes d=7, p=2e-3 LER gap
(~0.02 → ~0.00125, matching batch).

## Before / after (d=7, p=2e-3)

| method | before LER | after LER |
|--------|-----------:|----------:|
| batch | 0.00125 | 0.00125 |
| stream_w3d | **0.02** | **0.00125** |
| stream_w2d | 0.02125 | 0.00125 |
| mabs_adaptive | 0.02125 | 0.00125 |

At d=7, p=0.002, 800 shots: stream_w3d LER = batch LER = 0.00125.
