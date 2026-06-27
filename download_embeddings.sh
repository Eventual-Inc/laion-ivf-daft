#!/usr/bin/env bash
# Download LAION-400M CLIP image embeddings (512-dim, fp16).
# Each file is ~1 GB = ~1M vectors. There are 400 files total (img_emb_0 .. img_emb_399).
#
# Usage:
#   ./download_embeddings.sh [N]      # download the first N files (default: 5)
#
# Example: ./download_embeddings.sh 40   -> ~40 GB, 40M vectors

set -euo pipefail

N="${1:-5}"
BASE="https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb"
OUT="data/laion"
mkdir -p "$OUT"

echo "Downloading $N LAION-400M embedding file(s) into $OUT/ ..."
for ((i=0; i<N; i++)); do
  dest="$OUT/img_emb_$i.npy"
  if [[ -f "$dest" ]]; then
    echo "  img_emb_$i.npy already present, skipping"
    continue
  fi
  echo "  img_emb_$i.npy ..."
  curl -fL --retry 3 -o "$dest" "$BASE/img_emb_$i.npy"
done
echo "Done. $(ls "$OUT"/*.npy | wc -l) file(s), $(du -sh "$OUT" | cut -f1) total."
