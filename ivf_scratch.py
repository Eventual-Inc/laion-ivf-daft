"""
IVF (inverted-file) nearest-neighbor search, implemented from scratch in NumPy.

The whole method is three small pieces:
  1. k-means  -> learn `nlist` centroids on a SAMPLE (defines the cells)
  2. assign   -> put every vector in its nearest cell (the inverted lists)
  3. search   -> for a query, scan only the `nprobe` nearest cells

No FAISS here. We validate this against FAISS-IVF separately (the "oracle").

Convention: all vectors are L2-normalized, so cosine similarity == dot product,
and "nearest" == largest dot product. We work with similarities (higher = closer).
"""

from __future__ import annotations
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


# ---------------------------------------------------------------------------
# 1. k-means (from scratch) -- this is the only "training" step
# ---------------------------------------------------------------------------
def kmeans(x: np.ndarray, k: int, n_iter: int = 20, seed: int = 0,
           verbose: bool = False) -> np.ndarray:
    """
    Lloyd's algorithm on L2-normalized vectors (so assignment uses dot product).
    Returns k centroids. Trains on whatever you pass in -- at scale you pass a
    SAMPLE, not the whole dataset (centroids are summary statistics).
    """
    x = l2_normalize(x)
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    # init: k distinct random points (k-means++ would be fancier; random is fine here)
    centroids = x[rng.choice(n, k, replace=False)].copy()

    for it in range(n_iter):
        # assign: each point -> centroid with highest dot product
        # (done in row-blocks to keep the similarity matrix small in memory)
        assign = np.empty(n, dtype=np.int32)
        block = 100_000
        for s in range(0, n, block):
            sims = x[s:s + block] @ centroids.T          # (block, k)
            assign[s:s + block] = np.argmax(sims, axis=1)

        # update: new centroid = mean of its members, then renormalize
        new_c = np.zeros_like(centroids)
        counts = np.bincount(assign, minlength=k)
        np.add.at(new_c, assign, x)
        empty = counts == 0
        new_c[~empty] /= counts[~empty, None]
        # re-seed any empty cluster onto a random point so we keep k cells
        if empty.any():
            new_c[empty] = x[rng.choice(n, int(empty.sum()), replace=False)]
        new_c = l2_normalize(new_c)

        shift = float(np.linalg.norm(new_c - centroids))
        centroids = new_c
        if verbose:
            print(f"  kmeans iter {it+1}/{n_iter}  shift={shift:.4f}  "
                  f"empty_cells={int(empty.sum())}")
        if shift < 1e-4:
            break
    return centroids


# ---------------------------------------------------------------------------
# 2 + 3. The IVF index: assign vectors to cells, then search nprobe cells
# ---------------------------------------------------------------------------
class IVFIndex:
    def __init__(self, centroids: np.ndarray):
        self.centroids = l2_normalize(centroids)         # (nlist, d)
        self.nlist = self.centroids.shape[0]
        self.lists: list[np.ndarray] = [np.empty(0, dtype=np.int64)
                                        for _ in range(self.nlist)]
        self.x: np.ndarray | None = None                 # the stored vectors

    def assign(self, x: np.ndarray, block: int = 100_000,
               normalize: bool = True) -> np.ndarray:
        """Nearest centroid for each row of x (the cell id)."""
        if normalize:
            x = l2_normalize(x)
        out = np.empty(x.shape[0], dtype=np.int32)
        for s in range(0, x.shape[0], block):
            out[s:s + block] = np.argmax(x[s:s + block] @ self.centroids.T, axis=1)
        return out

    def add(self, x: np.ndarray, normalize: bool = True):
        """Store vectors and bucket their ids into inverted lists by cell.

        Pass normalize=False if x is already L2-normalized (e.g. LAION embeddings)
        to avoid an extra full-size copy -- important at multi-GB scale.
        """
        if normalize:
            x = l2_normalize(x)
        self.x = x
        cell = self.assign(x, normalize=False)
        order = np.argsort(cell, kind="stable")          # group ids by cell
        sorted_cell = cell[order]
        # split the sorted id array at cell boundaries
        bounds = np.searchsorted(sorted_cell, np.arange(self.nlist + 1))
        for c in range(self.nlist):
            self.lists[c] = order[bounds[c]:bounds[c + 1]]

    def cell_sizes(self) -> np.ndarray:
        return np.array([len(l) for l in self.lists])

    def search(self, queries: np.ndarray, k: int = 10, nprobe: int = 1):
        """
        For each query: pick the `nprobe` nearest centroids, gather the vectors
        in those cells, and return the top-k by exact similarity.
        Returns (ids [nq,k], sims [nq,k]).
        """
        queries = l2_normalize(queries)
        nq = queries.shape[0]
        # nearest nprobe centroids per query
        cent_sims = queries @ self.centroids.T           # (nq, nlist)
        probe = np.argpartition(-cent_sims, nprobe - 1, axis=1)[:, :nprobe]

        ids_out = np.full((nq, k), -1, dtype=np.int64)
        sims_out = np.full((nq, k), -np.inf, dtype=np.float32)
        for q in range(nq):
            cand = np.concatenate([self.lists[c] for c in probe[q]]) \
                if nprobe > 1 else self.lists[probe[q][0]]
            if cand.size == 0:
                continue
            sims = self.x[cand] @ queries[q]             # exact sim to candidates
            topn = min(k, cand.size)
            top = np.argpartition(-sims, topn - 1)[:topn]
            top = top[np.argsort(-sims[top])]
            ids_out[q, :topn] = cand[top]
            sims_out[q, :topn] = sims[top]
        return ids_out, sims_out


# ---------------------------------------------------------------------------
# Exact brute-force search -- the ground-truth baseline (and the nprobe=nlist limit)
# ---------------------------------------------------------------------------
def exact_search(x: np.ndarray, queries: np.ndarray, k: int = 10,
                 normalize: bool = True):
    if normalize:
        x = l2_normalize(x); queries = l2_normalize(queries)
    ids = np.empty((queries.shape[0], k), dtype=np.int64)
    for i in range(queries.shape[0]):
        sims = x @ queries[i]
        top = np.argpartition(-sims, k - 1)[:k]
        ids[i] = top[np.argsort(-sims[top])]
    return ids
