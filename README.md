# MABS v4.1 — CASCADE

Author: **Aryaman Katoch**.

## Honest goal

Implement CASCADE priority list; keep LER = batch; **do not claim stage speedup** vs fair `stream_w3d` unless benches show it.

## Colab

```python
!rm -rf mabs-public
!git clone https://github.com/HellWithCode0/mabs-public.git
%cd mabs-public
!pip install -e . -q
!pytest -q
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5 --fixed-order
```

Version: `4.1.0a1`
