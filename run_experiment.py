"""
Experiment driver: build IVF over real LAION-400M embeddings, sweep the knobs,
validate against FAISS, and chart the speed/quality tradeoff.

Runs entirely on the from-scratch IVF (ivf_scratch.py). FAISS is used only as a
correctness oracle on a subset (shared centroids -> results should match exactly).

Usage:
  python run_experiment.py --n-vectors 5000000 --nlist 2048
"""

from __future__ import annotations
import argparse, glob, time, os
import numpy as np
from ivf_scratch import kmeans, IVFIndex, exact_search, l2_normalize


def load_vectors(n_vectors: int) -> np.ndarray:
    files = sorted(glob.glob("data/laion/img_emb_*.npy"),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    chunks, total = [], 0
    for f in files:
        a = np.load(f, mmap_mode="r")
        take = min(len(a), n_vectors - total)
        chunks.append(np.asarray(a[:take], dtype=np.float32))
        total += take
        if total >= n_vectors:
            break
    # LAION embeddings ship already L2-normalized, so we skip re-normalizing
    # (avoids an extra full-size copy at multi-GB scale).
    return np.concatenate(chunks)


def perturb(v, target_cos, rng):
    """Add Gaussian noise sized to land at ~target_cos, simulating a near-dup."""
    d = v.shape[1]
    sigma = np.sqrt((1 / target_cos**2 - 1) / d)
    return l2_normalize(v + rng.normal(0, sigma, v.shape).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vectors", type=int, default=5_000_000)
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--target-cos", type=float, default=0.95)
    ap.add_argument("--nprobe", type=int, nargs="+",
                    default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--oracle-n", type=int, default=1_000_000,
                    help="subset size for the FAISS correctness check")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs("charts", exist_ok=True)

    print(f"loading {args.n_vectors:,} LAION vectors ...")
    t0 = time.time()
    x = load_vectors(args.n_vectors)
    N = x.shape[0]
    print(f"  loaded {N:,} x {x.shape[1]}  ({x.nbytes/1e9:.1f} GB)  {time.time()-t0:.0f}s")

    # --- train k-means on a sample -> centroids (the only "training") ---
    n_train = min(N, max(256 * args.nlist, 100_000))
    print(f"training k-means: {args.nlist} cells on {n_train:,}-vector sample ...")
    t0 = time.time()
    cent = kmeans(x[rng.choice(N, n_train, replace=False)], k=args.nlist,
                  n_iter=15, verbose=True)
    print(f"  k-means done {time.time()-t0:.0f}s")

    # --- build the from-scratch index ---
    t0 = time.time()
    idx = IVFIndex(cent); idx.add(x, normalize=False)
    cs = idx.cell_sizes()
    print(f"index built {time.time()-t0:.0f}s  cells: min={cs.min()} "
          f"median={int(np.median(cs))} max={cs.max()}")

    # --- queries: real vectors perturbed to ~target_cos ---
    qi = rng.choice(N, args.n_queries, replace=False)
    queries = perturb(x[qi], args.target_cos, rng)
    truth_orig = qi                                   # the vector each query came from

    # exact NN (ground truth for the clean IVF-quality metric)
    print("computing exact baseline ...")
    t0 = time.time()
    exact_nn = exact_search(x, queries, k=1, normalize=False)[:, 0]
    exact_t = (time.time() - t0) / args.n_queries * 1000
    rec_orig_exact = (exact_nn == truth_orig).mean()
    print(f"  exact: {exact_t:.1f} ms/query   recall@1 vs original = {rec_orig_exact:.3f}")

    # --- sweep nprobe ---
    rows = []
    cent_sims = queries @ idx.centroids.T
    for nprobe in args.nprobe:
        if nprobe > args.nlist:
            continue
        t0 = time.time()
        ids, _ = idx.search(queries, k=1, nprobe=nprobe)
        qt = (time.time() - t0) / args.n_queries * 1000
        rec_vs_orig = (ids[:, 0] == truth_orig).mean()
        rec_vs_exact = (ids[:, 0] == exact_nn).mean()
        probe = np.argpartition(-cent_sims, nprobe - 1, axis=1)[:, :nprobe]
        avg_cand = cs[probe].sum(1).mean()
        rows.append((nprobe, rec_vs_orig, rec_vs_exact, avg_cand, qt))
        print(f"  nprobe={nprobe:5d}  rec_vs_orig={rec_vs_orig:.3f}  "
              f"rec_vs_exact={rec_vs_exact:.3f}  cand={avg_cand:,.0f}  {qt:.2f} ms")

    # --- FAISS oracle on a subset (shared centroids) ---
    import faiss
    on = min(args.oracle_n, N)
    print(f"FAISS oracle check on {on:,} vectors (shared centroids) ...")
    quant = faiss.IndexFlatIP(x.shape[1]); quant.add(cent)
    fi = faiss.IndexIVFFlat(quant, x.shape[1], args.nlist, faiss.METRIC_INNER_PRODUCT)
    fi.is_trained = True; fi.add(x[:on])
    sub = IVFIndex(cent); sub.add(x[:on], normalize=False)
    oracle = []
    for nprobe in [1, 16, 256]:
        if nprobe > args.nlist:
            continue
        fi.nprobe = nprobe
        _, fid = fi.search(queries, 1)
        sid, _ = sub.search(queries, k=1, nprobe=nprobe)
        oracle.append((nprobe, (fid[:, 0] == sid[:, 0]).mean()))
        print(f"  nprobe={nprobe:4d}  from-scratch vs FAISS agreement = {oracle[-1][1]:.3f}")

    _make_charts(rows, cs, N, args)
    _write_results(rows, oracle, cs, N, exact_t, rec_orig_exact, args)
    print("wrote charts/ and RESULTS.md")


def _make_charts(rows, cs, N, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nprobe = [r[0] for r in rows]
    rec_exact = [r[2] for r in rows]
    qt = [r[4] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(nprobe, rec_exact, "o-")
    a1.set_xscale("log"); a1.set_xlabel("nprobe"); a1.set_ylabel("recall@1 vs exact NN")
    a1.set_title(f"Quality vs nprobe ({N:,} vectors, {args.nlist} cells)")
    a1.grid(True, alpha=0.3)
    a2.plot(nprobe, qt, "o-", color="C3")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("nprobe"); a2.set_ylabel("query time (ms)")
    a2.set_title("Speed vs nprobe"); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("charts/tradeoff.png", dpi=120)

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cs, bins=50)
    ax.set_xlabel("vectors per cell"); ax.set_ylabel("number of cells")
    ax.set_title(f"Cell-size distribution ({args.nlist} cells)")
    fig2.tight_layout(); fig2.savefig("charts/cell_sizes.png", dpi=120)


def _write_results(rows, oracle, cs, N, exact_t, rec_orig_exact, args):
    with open("RESULTS.md", "w") as f:
        f.write(f"# Results\n\n")
        f.write(f"- {N:,} real LAION-400M vectors (512-dim), {args.nlist} cells\n")
        f.write(f"- queries: {args.n_queries} real vectors perturbed to cosine "
                f"~{args.target_cos}\n")
        f.write(f"- cell sizes: min {cs.min()}, median {int(np.median(cs))}, "
                f"max {cs.max()}\n\n")
        f.write("| nprobe | recall@1 (vs exact NN) | candidates scanned | "
                "query time |\n")
        f.write("|---:|---:|---:|---:|\n")
        for nprobe, ro, re, cand, qt in rows:
            f.write(f"| {nprobe} | {re:.3f} | {cand:,.0f} "
                    f"({100*cand/N:.2f}%) | {qt:.2f} ms |\n")
        f.write(f"| exact | 1.000 | {N:,} (100%) | {exact_t:.1f} ms |\n\n")
        f.write("## FAISS oracle (shared centroids)\n\n")
        f.write("| nprobe | from-scratch vs FAISS top-1 agreement |\n|---:|---:|\n")
        for nprobe, ag in oracle:
            f.write(f"| {nprobe} | {ag:.3f} |\n")


if __name__ == "__main__":
    main()
