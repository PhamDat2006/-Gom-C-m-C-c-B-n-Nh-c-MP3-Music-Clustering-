"""Core module for scalable, unsupervised clustering of MP3/audio music tracks."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
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


def _safe_stats(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute (mean, std, skew, kurtosis) safely without NaNs on constant arrays."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = float(np.mean(arr))
    s = float(np.std(arr))
    if s < 1e-12:
        return m, s, 0.0, 0.0
    sk = float(skew(arr, bias=False))
    ku = float(kurtosis(arr, bias=False))
    return (
        m,
        s,
        0.0 if np.isnan(sk) else sk,
        0.0 if np.isnan(ku) else ku,
    )


def _add_matrix_stats(
    features: Dict[str, float | str | int], prefix: str, matrix: np.ndarray
) -> None:
    """Add mean, std, skew, kurtosis for each row of a 2D feature matrix."""
    for i in range(matrix.shape[0]):
        m, s, sk, ku = _safe_stats(matrix[i])
        features[f"{prefix}_{i + 1}_mean"] = m
        features[f"{prefix}_{i + 1}_std"] = s
        features[f"{prefix}_{i + 1}_skew"] = sk
        features[f"{prefix}_{i + 1}_kurt"] = ku


def extract_audio_features(
    file_path: str | Path,
    sr: int = 22050,
    duration: Optional[float] = 30.0,
    offset: float = 0.0,
    n_mfcc: int = 13,
) -> Dict[str, float | str | int]:
    """Extract rich acoustic features (MFCC, Chroma, Tonnetz, Spectral, Tempo, RMS, ZCR)."""
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

    # Harmony (Chroma, Tonnetz) & Timbre (MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)

    stem = path.stem
    track_id = int(stem) if stem.isdigit() else -1

    features: Dict[str, float | str | int] = {
        "track_id": track_id,
        "filename": path.name,
        "filepath": str(path.resolve()),
        "duration_sec": round(actual_duration, 2),
        "tempo_bpm": round(tempo, 2),
    }

    for name, vec in [
        ("rms", rms),
        ("zcr", zcr),
        ("spectral_centroid", centroid),
        ("spectral_bandwidth", bandwidth),
        ("spectral_rolloff", rolloff),
        ("spectral_contrast", contrast),
    ]:
        m, s, sk, ku = _safe_stats(vec)
        features[f"{name}_mean"] = m
        features[f"{name}_std"] = s
        features[f"{name}_skew"] = sk
        features[f"{name}_kurt"] = ku

    _add_matrix_stats(features, "chroma", chroma)
    _add_matrix_stats(features, "tonnetz", tonnetz)
    _add_matrix_stats(features, "mfcc", mfcc)

    return features


def _extract_worker(
    args: Tuple[str, float | None, float],
) -> Tuple[str, Optional[Dict[str, float | str | int]], Optional[str]]:
    """Top-level worker function for ProcessPoolExecutor."""
    fpath, duration, offset = args
    try:
        feat = extract_audio_features(fpath, duration=duration, offset=offset)
        return fpath, feat, None
    except Exception as exc:  # noqa: BLE001
        return fpath, None, str(exc)


def extract_features_batch(
    file_paths: List[str | Path],
    duration: Optional[float] = 30.0,
    offset: float = 0.0,
    n_jobs: int = -1,
    checkpoint_path: Optional[str | Path] = None,
    checkpoint_every: int = 50,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Extract features in parallel with optional CSV checkpoint/resume support."""
    existing_df = pd.DataFrame()
    done_paths: set[str] = set()
    ckpt_file = Path(checkpoint_path) if checkpoint_path else None

    if ckpt_file and ckpt_file.exists():
        existing_df = pd.read_csv(ckpt_file)
        if "filepath" in existing_df.columns:
            done_paths = set(existing_df["filepath"].astype(str).tolist())

    pending = [
        str(Path(p).resolve())
        for p in file_paths
        if str(Path(p).resolve()) not in done_paths
    ]

    records: List[Dict[str, float | str | int]] = (
        existing_df.to_dict("records") if not existing_df.empty else []
    )
    errors: List[Tuple[str, str]] = []
    total = len(file_paths)
    completed_count = len(done_paths)

    if not pending:
        return pd.DataFrame(records), errors

    workers = os.cpu_count() or 1 if n_jobs == -1 else max(1, n_jobs)
    tasks = [(fp, duration, offset) for fp in pending]

    def _flush_checkpoint() -> None:
        if ckpt_file and records:
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(ckpt_file, index=False, encoding="utf-8-sig")

    if workers == 1 or len(tasks) <= 3:
        for task in tasks:
            fp, feat, err = _extract_worker(task)
            completed_count += 1
            if progress_callback is not None:
                progress_callback(min(completed_count, total), total, Path(fp).name)
            if feat is not None:
                records.append(feat)
            elif err is not None:
                errors.append((fp, err))
            if ckpt_file and len(records) % checkpoint_every == 0:
                _flush_checkpoint()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_extract_worker, t) for t in tasks]
            for fut in as_completed(futures):
                fp, feat, err = fut.result()
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(min(completed_count, total), total, Path(fp).name)
                if feat is not None:
                    records.append(feat)
                elif err is not None:
                    errors.append((fp, err))
                if ckpt_file and len(records) % checkpoint_every == 0:
                    _flush_checkpoint()

    _flush_checkpoint()
    df = pd.DataFrame(records)
    if not df.empty and "filename" in df.columns:
        df = df.sort_values("filename").reset_index(drop=True)
    return df, errors


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric acoustic feature columns excluding metadata and outputs."""
    excluded = {
        "track_id",
        "filename",
        "filepath",
        "duration_sec",
        "cluster",
        "pca_1",
        "pca_2",
        "genre_top",
        "subset",
        "title",
        "artist",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def evaluate_k_range(
    df: pd.DataFrame, k_min: int = 2, k_max: int = 12
) -> pd.DataFrame:
    """Compute Inertia (Elbow) and Silhouette Score across candidate cluster counts."""
    feature_cols = get_feature_columns(df)
    if len(df) < 3 or not feature_cols:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])

    x_scaled = StandardScaler().fit_transform(
        df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
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


def estimate_dbscan_eps(x_scaled: np.ndarray, min_samples: int = 5) -> float:
    """Estimate DBSCAN eps automatically using the 90th percentile of k-NN distances."""
    k = max(2, min(min_samples, len(x_scaled)))
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_scaled)
    distances, _ = nbrs.kneighbors(x_scaled)
    k_dists = np.sort(distances[:, -1])
    eps = float(np.percentile(k_dists, 90))
    return max(0.5, round(eps, 3))


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
    n_clusters: int = 8,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 5,
    use_pca_for_clustering: bool = False,
    pca_variance: float = 0.90,
) -> ClusteringResult:
    """Scale features, run clustering, project to 2D via PCA, and compute metrics."""
    if df.empty:
        raise ValueError("Không có dữ liệu bài hát để gom cụm.")

    feature_cols = get_feature_columns(df)
    x_raw = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()
    x_scaled = StandardScaler().fit_transform(x_raw)

    x_cluster = x_scaled
    if use_pca_for_clustering and x_scaled.shape[0] > 10 and x_scaled.shape[1] > 10:
        pca_red = PCA(n_components=pca_variance, random_state=42)
        x_cluster = pca_red.fit_transform(x_scaled)

    algo = algorithm.lower().strip()
    used_eps: Optional[float] = None
    if algo == "kmeans":
        k = max(1, min(n_clusters, len(df)))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_cluster)
    elif algo == "agglomerative":
        k = max(1, min(n_clusters, len(df)))
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(x_cluster)
    elif algo == "dbscan":
        used_eps = (
            float(dbscan_eps)
            if dbscan_eps is not None and dbscan_eps > 0
            else estimate_dbscan_eps(x_cluster, dbscan_min_samples)
        )
        model = DBSCAN(eps=used_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(x_cluster)
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

    unique_labels = set(labels)
    valid_labels = [lbl for lbl in unique_labels if lbl != -1]
    metrics: Dict[str, Optional[float]] = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "n_clusters_found": float(len(valid_labels)),
        "n_noise": float(int(np.sum(labels == -1))),
        "dbscan_eps_used": used_eps,
    }

    if len(unique_labels) >= 2 and len(unique_labels) < len(df):
        metrics["silhouette"] = float(silhouette_score(x_cluster, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(x_cluster, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x_cluster, labels))

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


def evaluate_with_ground_truth(
    clustered_df: pd.DataFrame, label_col: str = "genre_top"
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate clustering against ground-truth labels using NMI, ARI, Purity, and crosstab."""
    valid = clustered_df.dropna(subset=[label_col]).copy()
    if valid.empty:
        raise ValueError(f"Không có nhãn thật hợp lệ trong cột '{label_col}'.")

    y_true = valid[label_col].astype(str)
    y_pred = valid["cluster"].astype(int)

    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))

    crosstab = pd.crosstab(y_true, y_pred)
    purity = float(crosstab.max(axis=0).sum() / len(valid))

    scores = {
        "nmi": nmi,
        "ari": ari,
        "purity": purity,
        "evaluated_tracks": float(len(valid)),
    }
    return scores, crosstab


def load_fma_tracks_metadata(tracks_csv_path: str | Path) -> pd.DataFrame:
    """Load FMA tracks.csv and return flat track_id, genre_top, subset, title, artist."""
    tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])
    out = pd.DataFrame(
        {
            "track_id": tracks.index.astype(int),
            "genre_top": tracks[("track", "genre_top")],
            "subset": tracks[("set", "subset")],
            "title": tracks[("track", "title")],
            "artist": tracks[("artist", "name")],
        }
    ).reset_index(drop=True)
    return out


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
