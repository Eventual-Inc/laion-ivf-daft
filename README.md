# Distributed IVF search & dedup over LAION embeddings

A from-scratch implementation of **IVF (inverted-file) nearest-neighbor search**,
built to deduplicate and search over hundreds of millions of CLIP embeddings that
don't fit in RAM. The vector math and data-engineering layer run on
[Daft](https://github.com/Eventual-Inc/Daft); the IVF index itself is written from
scratch in NumPy and validated against FAISS.

## The problem

LAION-400M ships ~400 million CLIP image embeddings (512-dim, fp16) as 400 `.npy`
files, ~410 GB total. You can't load that into memory, and brute-force all-pairs
similarity is O(n²). The standard answer is IVF: partition the vector space into
cells, then only search the few cells near a query.

## The method (three small pieces)

1. **k-means** - learn `nlist` centroids on a *sample* (a few GB), defining the cells.
   The full dataset is never loaded; centroids are summary statistics.
2. **assign** - stream every vector through, putting it in its nearest cell. Bounded
   memory.
3. **search** - for a query, scan only the `nprobe` nearest cells instead of all data.

`nprobe` is the speed/quality dial: `nprobe=1` is fast and approximate, `nprobe=nlist`
is exact brute force.

## Result (5M real LAION vectors, 2048 cells, M4 Max)

Queries are real LAION vectors perturbed with Gaussian noise to cosine ≈ 0.95
(simulating a near-duplicate), then searched for. We report recall against two
baselines and the cost.

| nprobe | recall@1 vs original | recall@1 vs exact NN | candidates scanned | query time |
|-------:|--------------------:|--------------------:|-------------------:|-----------:|
| 1      | 0.886 | 0.900 | 2,855 (0.06%)   | 1.3 ms   |
| 4      | 0.982 | 0.998 | 11,285 (0.23%)  | 5.0 ms   |
| 16     | 0.984 | **1.000** | 43,836 (0.88%) | 20 ms |
| 64     | 0.984 | 1.000 | 170,090 (3.4%)  | 78 ms    |
| 256    | 0.984 | 1.000 | 668,757 (13%)   | 334 ms   |
| exact  | 0.984 | 1.000 | 5,000,000 (100%)| 92 ms    |

At `nprobe=16`, IVF matches exact search's answers (**recall 1.000**) while scanning
**under 1% of the data**.

> **Why isn't "recall vs original" 1.0?** It's capped by the task, not by IVF. Exact
> brute-force search also scores 0.984: for ~1.6% of queries the noise makes a
> *different* real LAION image genuinely closer than the original - because LAION
> contains many near-duplicates. Measured against exact search's own answer (the true
> nearest neighbor), IVF reaches a clean **1.000**, confirming it loses nothing.

(Query times are for the from-scratch NumPy loop; FAISS is faster in absolute ms but
algorithmically identical - the "candidates scanned" column is the real comparison.)

## Architecture

| Layer | Tool |
|---|---|
| Stream embeddings, assign to cells, partition by cell, within-cell scan, exact baseline | **Daft** (`cosine_distance`, `groupby`) |
| k-means + nprobe search ("the IVF brain") | **from scratch** (`ivf_scratch.py`) |
| Correctness oracle | **FAISS** IVF (cross-check only, not a runtime dep) |

This mirrors LAION's own production index (`autofaiss`: `OPQ…,IVF131072_HNSW32,PQ…`) - same IVF + PQ skeleton, rebuilt from first principles.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python faiss-cpu numpy daft tqdm matplotlib

# download LAION-400M embeddings (default 5 files ~5 GB; pass N for more)
./download_embeddings.sh 5
```

## License

Apache License 2.0 (same as Daft).
