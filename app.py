"""Interactive Streamlit application for clustering MP3/audio music tracks."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from music_clusterer import (
    cluster_tracks,
    evaluate_k_range,
    evaluate_with_ground_truth,
    export_clusters_to_folders,
    extract_features_batch,
    load_fma_tracks_metadata,
    scan_audio_files,
)

st.set_page_config(
    page_title="Gom Cụm Nhạc MP3 (Music Clustering)",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Ứng Dụng Gom Cụm Các Bản Nhạc MP3")
st.caption(
    "Trích xuất đặc trưng âm thanh mở rộng (MFCC, Chroma, Tonnetz, Spectral, Tempo, RMS, ZCR + mean/std/skew/kurtosis), "
    "hỗ trợ xử lý song song, lưu checkpoint, và đánh giá với nhãn thật (NMI/ARI)."
)

with st.sidebar:
    st.header("1. Nguồn dữ liệu âm thanh")
    source_mode = st.radio(
        "Chọn cách nhập dữ liệu:",
        [
            "Chọn thư mục trên máy",
            "Tải lên tệp MP3/WAV",
            "Nạp từ tệp đặc trưng đã lưu (CSV)",
        ],
    )

    audio_paths: List[Path] = []
    loaded_csv_df: pd.DataFrame | None = None

    if source_mode == "Chọn thư mục trên máy":
        folder_input = st.text_input(
            "Đường dẫn thư mục chứa nhạc:",
            value=str(Path.cwd() / "sample_music"),
        )
        recursive = st.checkbox("Quét cả thư mục con", value=True)
        if folder_input:
            audio_paths = scan_audio_files(folder_input, recursive=recursive)
            st.info(f"Tìm thấy **{len(audio_paths)}** tệp âm thanh hợp lệ.")
    elif source_mode == "Tải lên tệp MP3/WAV":
        uploaded_files = st.file_uploader(
            "Tải lên các bản nhạc (.mp3, .wav, .ogg, .flac):",
            type=["mp3", "wav", "ogg", "flac", "m4a"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            upload_dir = Path(tempfile.gettempdir()) / "mp3_clustering_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for up in uploaded_files:
                target = upload_dir / up.name
                target.write_bytes(up.getbuffer())
                audio_paths.append(target)
            st.info(f"Đã nhận **{len(audio_paths)}** tệp âm thanh.")
    else:
        csv_path_input = st.text_input(
            "Đường dẫn tệp CSV đặc trưng:",
            value=str(Path.cwd() / "fma_small_features.csv"),
        )
        if csv_path_input and Path(csv_path_input).exists():
            loaded_csv_df = pd.read_csv(csv_path_input)
            st.success(f"Đã đọc **{len(loaded_csv_df)}** bài hát từ CSV.")
        else:
            st.warning("Không tìm thấy tệp CSV tại đường dẫn trên.")

    st.divider()
    st.header("2. Trích xuất & Bộ nhớ đệm")
    duration_limit = st.slider(
        "Thời lượng phân tích mỗi bài (giây):",
        min_value=10,
        max_value=120,
        value=30,
        step=5,
    )
    use_parallel = st.checkbox("Trích xuất song song (đa nhân CPU)", value=True)
    cache_file = st.text_input(
        "Tệp checkpoint/cache (.csv):",
        value="features_cache.csv",
        help="Lưu tiến trình trích xuất để tiếp tục khi bị ngắt và tránh trích xuất lại.",
    )

    st.divider()
    st.header("3. Nhãn thật FMA (tùy chọn)")
    tracks_csv_input = st.text_input(
        "Đường dẫn fma_metadata/tracks.csv:",
        value=str(Path.cwd() / "fma_metadata" / "tracks.csv"),
    )

    st.divider()
    st.header("4. Thuật toán gom cụm")
    algo = st.selectbox("Chọn thuật toán:", ["KMeans", "Agglomerative", "DBSCAN"])
    use_pca_cluster = st.checkbox(
        "Giảm nhiễu bằng PCA (giữ 90% phương sai) trước khi gom cụm",
        value=False,
    )

    if algo in ("KMeans", "Agglomerative"):
        n_clusters = st.slider("Số cụm (k):", min_value=2, max_value=16, value=8)
        dbscan_eps: float | None = None
        dbscan_min = 5
    else:
        n_clusters = 8
        auto_eps = st.checkbox("Tự động ước lượng eps (k-distance)", value=True)
        dbscan_min = st.slider("DBSCAN min_samples:", min_value=2, max_value=20, value=5)
        dbscan_eps = (
            None
            if auto_eps
            else st.slider("DBSCAN eps:", min_value=0.5, max_value=25.0, value=8.0, step=0.5)
        )


if "features_df" not in st.session_state:
    st.session_state["features_df"] = None
if "extract_errors" not in st.session_state:
    st.session_state["extract_errors"] = []

if source_mode == "Nạp từ tệp đặc trưng đã lưu (CSV)" and loaded_csv_df is not None:
    st.session_state["features_df"] = loaded_csv_df
    st.session_state["extract_errors"] = []
else:
    run_extract = st.button(
        "🚀 Trích xuất đặc trưng & Gom cụm",
        type="primary",
        disabled=len(audio_paths) < 2,
    )
    if len(audio_paths) < 2:
        st.warning("Vui lòng cung cấp ít nhất **2 tệp âm thanh** để thực hiện gom cụm.")

    if run_extract and len(audio_paths) >= 2:
        progress_bar = st.progress(0, text="Đang trích xuất đặc trưng âm thanh...")

        def _update_progress(idx: int, total: int, name: str) -> None:
            progress_bar.progress(idx / total, text=f"Đang xử lý ({idx}/{total}): {name}")

        df_feat, errs = extract_features_batch(
            audio_paths,
            duration=float(duration_limit),
            n_jobs=-1 if use_parallel else 1,
            checkpoint_path=cache_file if cache_file.strip() else None,
            progress_callback=_update_progress,
        )
        progress_bar.empty()
        st.session_state["features_df"] = df_feat
        st.session_state["extract_errors"] = errs

features_df: pd.DataFrame | None = st.session_state["features_df"]
extract_errors = st.session_state["extract_errors"]

if extract_errors:
    with st.expander(f"⚠️ Có {len(extract_errors)} tệp không đọc được", expanded=False):
        for fpath, msg in extract_errors[:50]:
            st.text(f"{fpath}: {msg}")

if features_df is not None and len(features_df) >= 2:
    # Attach genre_top from FMA tracks.csv if available and not already present
    work_df = features_df.copy()
    if (
        "genre_top" not in work_df.columns
        and tracks_csv_input
        and Path(tracks_csv_input).exists()
        and "track_id" in work_df.columns
    ):
        meta_df = load_fma_tracks_metadata(tracks_csv_input)
        work_df = work_df.merge(meta_df, on="track_id", how="left")

    res = cluster_tracks(
        work_df,
        algorithm=algo,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min,
        use_pca_for_clustering=use_pca_cluster,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng số bài hát", len(res.df))
    m2.metric("Số cụm tìm được", int(res.metrics["n_clusters_found"] or 0))
    sil = res.metrics["silhouette"]
    m3.metric("Silhouette Score", f"{sil:.3f}" if sil is not None else "N/A")
    dbi = res.metrics["davies_bouldin"]
    m4.metric("Davies-Bouldin Index", f"{dbi:.3f}" if dbi is not None else "N/A")

    if res.metrics.get("dbscan_eps_used") is not None:
        st.info(
            f"DBSCAN sử dụng `eps = {res.metrics['dbscan_eps_used']}` | "
            f"Số bài nhiễu (Outlier): `{int(res.metrics['n_noise'] or 0)}`"
        )

    tab_viz, tab_eval, tab_k, tab_listen, tab_export = st.tabs(
        [
            "📊 Trực quan hóa cụm",
            "🏷️ Đánh giá với nhãn thật (FMA)",
            "📈 Tìm số cụm tối ưu",
            "🎧 Nghe thử theo cụm",
            "💾 Xuất kết quả & Playlist",
        ]
    )

    with tab_viz:
        c1, c2 = st.columns([3, 2])
        with c1:
            plot_df = res.df.sample(min(2000, len(res.df)), random_state=42)
            fig, ax = plt.subplots(figsize=(8, 5))
            palette = sns.color_palette("tab10", n_colors=max(1, plot_df["cluster"].nunique()))
            sns.scatterplot(
                data=plot_df,
                x="pca_1",
                y="pca_2",
                hue="cluster",
                palette=palette,
                s=65 if len(plot_df) > 100 else 110,
                alpha=0.8,
                ax=ax,
            )
            if len(plot_df) <= 30:
                for _, row in plot_df.iterrows():
                    ax.text(
                        row["pca_1"] + 0.05,
                        row["pca_2"] + 0.05,
                        str(row["filename"])[:18],
                        fontsize=8,
                    )
            v1, v2 = res.pca_variance_ratio
            ax.set_title("Biểu đồ phân cụm bài hát (PCA 2D)")
            ax.set_xlabel(f"PCA 1 ({v1 * 100:.1f}% phương sai)")
            ax.set_ylabel(f"PCA 2 ({v2 * 100:.1f}% phương sai)")
            ax.legend(title="Cụm")
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.subheader("Hồ sơ đặc trưng từng cụm")
            st.dataframe(res.cluster_profiles, use_container_width=True)

        display_cols = [
            col
            for col in [
                "filename",
                "cluster",
                "genre_top",
                "duration_sec",
                "tempo_bpm",
                "rms_mean",
                "spectral_centroid_mean",
            ]
            if col in res.df.columns
        ]
        st.dataframe(res.df[display_cols].head(500), use_container_width=True)

    with tab_eval:
        if "genre_top" in res.df.columns and res.df["genre_top"].notna().any():
            gt_scores, crosstab = evaluate_with_ground_truth(res.df, label_col="genre_top")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("NMI", f"{gt_scores['nmi']:.4f}")
            g2.metric("ARI", f"{gt_scores['ari']:.4f}")
            g3.metric("Purity", f"{gt_scores['purity']:.4f}")
            g4.metric("Số bài có nhãn", int(gt_scores["evaluated_tracks"]))

            st.subheader("Bảng phân bố Thể loại (genre_top) theo Cụm")
            fig_ct, ax_ct = plt.subplots(figsize=(10, 5))
            sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax_ct)
            ax_ct.set_xlabel("Cụm dự đoán (Cluster)")
            ax_ct.set_ylabel("Thể loại thật (genre_top)")
            st.pyplot(fig_ct)
            plt.close(fig_ct)
        else:
            st.info(
                "Chưa có cột nhãn thật `genre_top`. Hãy cung cấp đường dẫn tới `fma_metadata/tracks.csv` "
                "khi chạy trên bộ dữ liệu FMA."
            )

    with tab_k:
        if len(work_df) > 1500:
            st.info("Dữ liệu lớn: đang lấy mẫu ngẫu nhiên 1.500 bài để vẽ biểu đồ Elbow & Silhouette nhanh.")
            k_sample = work_df.sample(1500, random_state=42)
        else:
            k_sample = work_df

        k_df = evaluate_k_range(k_sample, k_min=2, k_max=min(12, len(k_sample) - 1))
        if not k_df.empty:
            fig_k, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(k_df["k"], k_df["inertia"], marker="o")
            ax1.set_title("Elbow Method (Inertia)")
            ax1.set_xlabel("Số cụm (k)")

            ax2.plot(k_df["k"], k_df["silhouette"], marker="s", color="green")
            ax2.set_title("Silhouette Score")
            ax2.set_xlabel("Số cụm (k)")
            st.pyplot(fig_k)
            plt.close(fig_k)
        else:
            st.info("Cần ít nhất 4 bài hát để vẽ biểu đồ đánh giá dải k.")

    with tab_listen:
        selected_cluster = st.selectbox(
            "Chọn cụm muốn nghe thử:",
            sorted(res.df["cluster"].unique().tolist()),
        )
        sub_df = res.df[res.df["cluster"] == selected_cluster].head(15)
        profile_text = res.cluster_profiles.loc[
            res.cluster_profiles["cluster"] == selected_cluster, "profile"
        ].values[0]
        st.markdown(f"**Đặc trưng cụm {selected_cluster}:** {profile_text}")

        for _, row in sub_df.iterrows():
            genre_info = f" | Thể loại: `{row['genre_top']}`" if "genre_top" in row and pd.notna(row["genre_top"]) else ""
            st.write(
                f"🎵 **{row['filename']}** — Tempo: `{row['tempo_bpm']} BPM` | "
                f"Thời lượng: `{row['duration_sec']}s`{genre_info}"
            )
            fpath = Path(str(row["filepath"]))
            if fpath.exists():
                st.audio(str(fpath))

    with tab_export:
        out_folder = st.text_input(
            "Thư mục xuất kết quả:",
            value=str(Path.cwd() / "clustered_output"),
        )
        copy_audio = st.checkbox("Sao chép tệp nhạc vào các thư mục cụm", value=False)
        if st.button("💾 Xuất thư mục & Playlist (.m3u8, .csv)"):
            out_path = export_clusters_to_folders(res.df, out_folder, copy_files=copy_audio)
            st.success(f"Đã xuất kết quả thành công vào: `{out_path}`")
