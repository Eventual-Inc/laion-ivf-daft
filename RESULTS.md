# Results

- 5,000,000 real LAION-400M vectors (512-dim), 2048 cells
- queries: 500 real vectors perturbed to cosine ~0.95
- cell sizes: min 1, median 2329, max 8008

| nprobe | recall@1 (vs exact NN) | candidates scanned | query time |
|---:|---:|---:|---:|
| 1 | 0.900 | 2,855 (0.06%) | 1.28 ms |
| 4 | 0.998 | 11,285 (0.23%) | 5.00 ms |
| 16 | 1.000 | 43,836 (0.88%) | 20.35 ms |
| 64 | 1.000 | 170,090 (3.40%) | 78.12 ms |
| 256 | 1.000 | 668,757 (13.38%) | 334.50 ms |
| 1024 | 1.000 | 2,631,722 (52.63%) | 1315.22 ms |
| exact | 1.000 | 5,000,000 (100%) | 92.3 ms |

## FAISS oracle (shared centroids)

| nprobe | from-scratch vs FAISS top-1 agreement |
|---:|---:|
| 1 | 1.000 |
| 16 | 1.000 |
| 256 | 1.000 |
