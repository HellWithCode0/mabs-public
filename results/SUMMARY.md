# MABS benchmark summary

| method | d | p | shots | LER | errors | mean_stage_ns | mean_shot_ns | escalate/retry | N_disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 1 | 4718.6 | 4718.6 | 0.000 | 0 |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | 1 | 14483.1 | 195083.0 | 0.000 | 0 |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | 1 | 22584.0 | 286405.0 | 0.741 | 0 |
| mabs_v4 | 5 | 0.001 | 2000 | 0.0005 | 1 | 14758.6 | 219802.3 | 0.741 | 0 |
| batch | 7 | 0.001 | 800 | 0 | 0 | 15570.7 | 15570.7 | 0.000 | 0 |
| stream_w3d | 7 | 0.001 | 800 | 0 | 0 | 25402.7 | 293933.1 | 0.000 | 0 |
| mabs_v3 | 7 | 0.001 | 800 | 0 | 0 | 29782.5 | 346361.9 | 0.991 | 0 |
| mabs_v4 | 7 | 0.001 | 800 | 0 | 0 | 25607.7 | 310495.2 | 0.991 | 0 |
| batch | 5 | 0.002 | 2000 | 0.011 | 22 | 10129.0 | 10129.0 | 0.000 | 0 |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | 22 | 18381.3 | 230133.4 | 0.000 | 0 |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | 22 | 21885.9 | 274541.7 | 0.957 | 0 |
| mabs_v4 | 5 | 0.002 | 2000 | 0.011 | 22 | 18319.4 | 241754.1 | 0.957 | 0 |
| batch | 7 | 0.002 | 800 | 0.00125 | 1 | 32406.9 | 32406.9 | 0.000 | 0 |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | 1 | 38986.4 | 423496.3 | 0.000 | 0 |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | 1 | 41346.8 | 455852.0 | 1.000 | 0 |
| mabs_v4 | 7 | 0.002 | 800 | 0.00125 | 1 | 38987.9 | 441731.3 | 1.000 | 0 |

## Method notes

- **mabs_v3 / SLEM**: empty skip + local 1-2 + blossom escalate.
- **mabs_v4 / CASCADE FLASH** (default): empty / K=1 boundary CommitAction LUT / K=2 pair CommitAction LUT / K>=3 blossom; FULL mode opt-in.
- **stream_w3d**: single `decode_to_edges_array` per window (fair baseline; no double blossom).
