"""Batch pipeline script to extract features from fma_small and evaluate against genre_top."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from music_clusterer import (
    cluster_tracks,
    evaluate_with_ground_truth,
    extract_features_batch,
    load_fma_tracks_metadata,
    scan_audio_files,
)


def main() -> None:
    t0 = time.perf_counter()
    fma_dir = Path("fma_small")
    tracks_csv = Path("fma_metadata/tracks.csv")
    ckpt_csv = Path("fma_small_features.csv")

    files = scan_audio_files(str(fma_dir), recursive=True)
    print(f"Found {len(files)} audio files in {fma_dir}")
    if not files:
        raise FileNotFoundError(f"No audio files found in {fma_dir}")

    def _prog(idx: int, total: int, name: str) -> None:
        if idx % 500 == 0 or idx == total:
            elapsed = time.perf_counter() - t0
            print(f"[{idx}/{total}] ({elapsed:.1f}s) latest={name}", flush=True)

    df_feat, errors = extract_features_batch(
        files,
        duration=30.0,
        n_jobs=-1,
        checkpoint_path=ckpt_csv,
        checkpoint_every=200,
        progress_callback=_prog,
    )
    print(f"Extraction finished: {len(df_feat)} tracks, {len(errors)} errors in {time.perf_counter() - t0:.1f}s")

    if tracks_csv.exists():
        meta = load_fma_tracks_metadata(tracks_csv)
        df_feat = df_feat.drop(columns=[c for c in ["genre_top", "subset", "title", "artist"] if c in df_feat.columns])
        df_feat = df_feat.merge(meta, on="track_id", how="left")
        df_feat.to_csv(ckpt_csv, index=False, encoding="utf-8-sig")

    for algo, use_pca in [("kmeans", False), ("kmeans", True), ("agglomerative", True), ("dbscan", True)]:
        res = cluster_tracks(
            df_feat,
            algorithm=algo,
            n_clusters=8,
            dbscan_eps=None,
            dbscan_min_samples=5,
            use_pca_for_clustering=use_pca,
        )
        print(f"\n=== {algo.upper()} (PCA={use_pca}) ===")
        print("Internal metrics:", res.metrics)
        if "genre_top" in res.df.columns and res.df["genre_top"].notna().any():
            gt, ct = evaluate_with_ground_truth(res.df, label_col="genre_top")
            print("External metrics (genre_top):", gt)
            print("Crosstab (genre_top x cluster):\n", ct)


if __name__ == "__main__":
    main()
