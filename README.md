# MABS v3 — Sparse Local Escalation Matching (SLEM)

**Honest goal:** Not beat Higgott–Gidney absolute µs in pure Python. **Pareto-dominate full PyMatching Sparse Blossom** on mean stage time for sparse Katoch workloads, with **LER matching batch MWPM**.

See full docs: https://github.com/HellWithCode0/mabs

## Colab

```python
!rm -rf mabs
!git clone https://github.com/HellWithCode0/mabs.git
%cd mabs
!pip install -e . -q
!pytest -q
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_adaptive,mabs_v3 --distances 5,7 --noise 0.001
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_adaptive,mabs_v3 --distances 5,7 --noise 0.002
```

Version `3.0.0a1` — MIT
