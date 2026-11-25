"""Core module for unsupervised clustering of MP3/audio music tracks."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")


@dataclass
class ClusteringResult:
    """Container for clustering outputs, metrics, and PCA projection."""

    df: pd.DataFrame
    feature_columns: List[str]
    scaled_features: np.ndarray
    pca_variance_ratio: Tuple[float, float]
    metrics: Dict[str, Optional[float]]
    cluster_profiles: pd.DataFrame


def scan_audio_files(folder_path: str, recursive: bool = True) -> List[Path]:
    """Scan a directory for supported audio files (MP3, WAV, OGG, FLAC, M4A)."""
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        return []

    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def extract_audio_features(
    file_path: str | Path,
    sr: int = 22050,
    duration: Optional[float] = 30.0,
    offset: float = 0.0,
    n_mfcc: int = 13,
) -> Dict[str, float | str]:
    """Extract acoustic, timbral, rhythmic, and spectral features from an audio file."""
    path = Path(file_path)
    y, sample_rate = librosa.load(
        str(path), sr=sr, mono=True, duration=duration, offset=offset
    )

    if len(y) == 0:
        raise ValueError(f"Tệp âm thanh rỗng: {path.name}")

    # Trim leading/trailing silence to avoid skewing spectral averages
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) >= sample_rate // 2:
        y = y_trimmed

    actual_duration = float(len(y) / sample_rate)

    # Rhythm: Tempo (BPM)
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
    tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)
    tempo = float(np.atleast_1d(tempo_arr)[0])

    # Energy & Percussiveness
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    # Spectral shape
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate)

    # Harmony (Chroma) & Timbre (MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)

    features: Dict[str, float | str] = {
        "filename": path.name,
        "filepath": str(path.resolve()),
        "duration_sec": round(actual_duration, 2),
        "tempo_bpm": round(tempo, 2),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_contrast_std": float(np.std(contrast)),
    }

    for i in range(chroma.shape[0]):
        features[f"chroma_{i + 1}_mean"] = float(np.mean(chroma[i]))

    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    return features


def extract_features_batch(
    file_paths: List[str | Path],
    duration: Optional[float] = 30.0,
    offset: float = 0.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Extract features for a list of audio files, returning a DataFrame and any errors."""
    records: List[Dict[str, float | str]] = []
    errors: List[Tuple[str, str]] = []
    total = len(file_paths)

    for idx, fpath in enumerate(file_paths, start=1):
        name = Path(fpath).name
        if progress_callback is not None:
            progress_callback(idx, total, name)
        try:
            feat = extract_audio_features(fpath, duration=duration, offset=offset)
            records.append(feat)
        except Exception as exc:  # noqa: BLE001
            errors.append((str(fpath), str(exc)))

    df = pd.DataFrame(records)
    return df, errors


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns excluding metadata and cluster outputs."""
    excluded = {"filename", "filepath", "duration_sec", "cluster", "pca_1", "pca_2"}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def evaluate_k_range(
    df: pd.DataFrame, k_min: int = 2, k_max: int = 10
) -> pd.DataFrame:
    """Compute Inertia (Elbow) and Silhouette Score across candidate cluster counts."""
    feature_cols = get_feature_columns(df)
    if len(df) < 3 or not feature_cols:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])

    x_scaled = StandardScaler().fit_transform(df[feature_cols].fillna(0.0))
    upper_k = min(k_max, len(df) - 1)
    rows = []

    for k in range(max(2, k_min), upper_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
        sil = float(silhouette_score(x_scaled, labels))
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": sil,
            }
        )

    return pd.DataFrame(rows)


def _describe_cluster(row: pd.Series, global_means: pd.Series) -> str:
    """Create a concise Vietnamese musical profile label for a cluster."""
    tags: List[str] = []

    if row["tempo_bpm"] >= global_means["tempo_bpm"] * 1.1:
        tags.append("Nhịp nhanh")
    elif row["tempo_bpm"] <= global_means["tempo_bpm"] * 0.9:
        tags.append("Nhịp chậm")
    else:
        tags.append("Nhịp vừa")

    if row["rms_mean"] >= global_means["rms_mean"] * 1.15:
        tags.append("Năng lượng cao")
    elif row["rms_mean"] <= global_means["rms_mean"] * 0.85:
        tags.append("Êm dịu")

    if row["spectral_centroid_mean"] >= global_means["spectral_centroid_mean"] * 1.1:
        tags.append("Âm sắc sáng")
    elif row["spectral_centroid_mean"] <= global_means["spectral_centroid_mean"] * 0.9:
        tags.append("Âm sắc trầm/ấm")

    return " • ".join(tags)


def cluster_tracks(
    df: pd.DataFrame,
    algorithm: str = "kmeans",
    n_clusters: int = 4,
    dbscan_eps: float = 3.0,
    dbscan_min_samples: int = 2,
) -> ClusteringResult:
    """Scale features, run clustering, project to 2D via PCA, and compute metrics."""
    if df.empty:
        raise ValueError("Không có dữ liệu bài hát để gom cụm.")

    feature_cols = get_feature_columns(df)
    x_raw = df[feature_cols].fillna(0.0).to_numpy()
    x_scaled = StandardScaler().fit_transform(x_raw)

    algo = algorithm.lower().strip()
    if algo == "kmeans":
        k = max(1, min(n_clusters, len(df)))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
    elif algo == "agglomerative":
        k = max(1, min(n_clusters, len(df)))
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(x_scaled)
    elif algo == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(x_scaled)
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")

    result_df = df.copy()
    result_df["cluster"] = labels

    # 2D PCA projection for visualization
    n_components = min(2, x_scaled.shape[0], x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(x_scaled)
    result_df["pca_1"] = coords[:, 0]
    result_df["pca_2"] = coords[:, 1] if n_components == 2 else 0.0
    var_ratio = (
        float(pca.explained_variance_ratio_[0]),
        float(pca.explained_variance_ratio_[1]) if n_components == 2 else 0.0,
    )

    # Evaluation metrics (requires >= 2 distinct clusters and < n_samples)
    unique_labels = set(labels)
    valid_labels = [lbl for lbl in unique_labels if lbl != -1]
    metrics: Dict[str, Optional[float]] = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "n_clusters_found": float(len(valid_labels)),
        "n_noise": float(int(np.sum(labels == -1))),
    }

    if len(unique_labels) >= 2 and len(unique_labels) < len(df):
        metrics["silhouette"] = float(silhouette_score(x_scaled, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(x_scaled, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x_scaled, labels))

    # Build cluster summary profiles
    summary_cols = [
        "tempo_bpm",
        "rms_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "duration_sec",
    ]
    global_means = result_df[summary_cols].mean()
    profiles = (
        result_df.groupby("cluster")[summary_cols]
        .mean()
        .round(3)
        .reset_index()
    )
    counts = result_df.groupby("cluster")["filename"].count().rename("track_count")
    profiles = profiles.merge(counts, on="cluster")
    profiles["profile"] = profiles.apply(
        lambda r: "Nhiễu (Outlier)" if r["cluster"] == -1 else _describe_cluster(r, global_means),
        axis=1,
    )

    return ClusteringResult(
        df=result_df,
        feature_columns=feature_cols,
        scaled_features=x_scaled,
        pca_variance_ratio=var_ratio,
        metrics=metrics,
        cluster_profiles=profiles,
    )


def export_clusters_to_folders(
    clustered_df: pd.DataFrame, output_dir: str | Path, copy_files: bool = True
) -> Path:
    """Export clustered tracks into subfolders, M3U playlists, and a CSV report."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "clusters_summary.csv"
    clustered_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    for cluster_id, group in clustered_df.groupby("cluster"):
        folder_name = "Noise" if cluster_id == -1 else f"Cluster_{cluster_id}"
        cluster_dir = out_root / folder_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        playlist_lines = ["#EXTM3U"]
        for _, row in group.iterrows():
            src = Path(str(row["filepath"]))
            if src.exists():
                if copy_files:
                    dst = cluster_dir / src.name
                    if src.resolve() != dst.resolve():
                        shutil.copy2(src, dst)
                playlist_lines.append(str(src))

        m3u_path = out_root / f"{folder_name}.m3u8"
        m3u_path.write_text("\n".join(playlist_lines), encoding="utf-8")

    return out_root
