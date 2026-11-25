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
    export_clusters_to_folders,
    extract_features_batch,
    scan_audio_files,
)

st.set_page_config(
    page_title="Gom Cụm Nhạc MP3 (Music Clustering)",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Ứng Dụng Gom Cụm Các Bản Nhạc MP3")
st.caption(
    "Trích xuất đặc trưng âm thanh (MFCC, Chroma, Spectral, Tempo, RMS) "
    "và phân cụm bài hát tự động bằng Machine Learning không giám sát."
)

# Sidebar configuration
with st.sidebar:
    st.header("1. Nguồn dữ liệu âm thanh")
    source_mode = st.radio(
        "Chọn cách nhập nhạc:",
        ["Chọn thư mục trên máy", "Tải lên tệp MP3/WAV"],
    )

    audio_paths: List[Path] = []
    if source_mode == "Chọn thư mục trên máy":
        folder_input = st.text_input(
            "Đường dẫn thư mục chứa nhạc:",
            value=str(Path.cwd() / "sample_music"),
        )
        recursive = st.checkbox("Quét cả thư mục con", value=True)
        if folder_input:
            audio_paths = scan_audio_files(folder_input, recursive=recursive)
            st.info(f"Tìm thấy **{len(audio_paths)}** tệp âm thanh hợp lệ.")
    else:
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

    st.divider()
    st.header("2. Trích xuất đặc trưng")
    duration_limit = st.slider(
        "Thời lượng phân tích mỗi bài (giây):",
        min_value=10,
        max_value=120,
        value=30,
        step=5,
        help="Phân tích 30 giây đầu giúp xử lý nhanh hơn với bộ sưu tập lớn.",
    )
    offset_sec = st.number_input(
        "Bỏ qua phần đầu (giây):",
        min_value=0,
        max_value=60,
        value=0,
        step=5,
    )

    st.divider()
    st.header("3. Thuật toán gom cụm")
    algo = st.selectbox(
        "Chọn thuật toán:",
        ["KMeans", "Agglomerative", "DBSCAN"],
    )
    if algo in ("KMeans", "Agglomerative"):
        n_clusters = st.slider("Số cụm (k):", min_value=2, max_value=12, value=3)
        dbscan_eps = 3.0
        dbscan_min = 2
    else:
        n_clusters = 3
        dbscan_eps = st.slider("DBSCAN eps:", min_value=0.5, max_value=10.0, value=3.5, step=0.5)
        dbscan_min = st.slider("DBSCAN min_samples:", min_value=2, max_value=10, value=2)


# Session state for extracted features
if "features_df" not in st.session_state:
    st.session_state["features_df"] = None
if "extract_errors" not in st.session_state:
    st.session_state["extract_errors"] = []

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
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
        offset=float(offset_sec),
        progress_callback=_update_progress,
    )
    progress_bar.empty()
    st.session_state["features_df"] = df_feat
    st.session_state["extract_errors"] = errs

features_df: pd.DataFrame | None = st.session_state["features_df"]
extract_errors = st.session_state["extract_errors"]

if extract_errors:
    with st.expander(f"⚠️ Có {len(extract_errors)} tệp không đọc được", expanded=False):
        for fpath, msg in extract_errors:
            st.text(f"{fpath}: {msg}")

if features_df is not None and len(features_df) >= 2:
    res = cluster_tracks(
        features_df,
        algorithm=algo,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min,
    )

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng số bài hát", len(res.df))
    m2.metric("Số cụm tìm được", int(res.metrics["n_clusters_found"] or 0))
    sil = res.metrics["silhouette"]
    m3.metric("Silhouette Score", f"{sil:.3f}" if sil is not None else "N/A")
    dbi = res.metrics["davies_bouldin"]
    m4.metric("Davies-Bouldin Index", f"{dbi:.3f}" if dbi is not None else "N/A")

    tab_viz, tab_k, tab_listen, tab_export = st.tabs(
        [
            "📊 Trực quan hóa cụm",
            "📈 Tìm số cụm tối ưu (Elbow & Silhouette)",
            "🎧 Nghe thử theo cụm",
            "💾 Xuất kết quả & Playlist",
        ]
    )

    with tab_viz:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            palette = sns.color_palette("tab10", n_colors=max(1, res.df["cluster"].nunique()))
            sns.scatterplot(
                data=res.df,
                x="pca_1",
                y="pca_2",
                hue="cluster",
                palette=palette,
                s=110,
                alpha=0.85,
                ax=ax,
            )
            for _, row in res.df.iterrows():
                ax.text(
                    row["pca_1"] + 0.05,
                    row["pca_2"] + 0.05,
                    str(row["filename"])[:18],
                    fontsize=8,
                    alpha=0.8,
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

        st.subheader("Bảng chi tiết bài hát & cụm")
        display_cols = [
            "filename",
            "cluster",
            "duration_sec",
            "tempo_bpm",
            "rms_mean",
            "spectral_centroid_mean",
        ]
        st.dataframe(res.df[display_cols], use_container_width=True)

    with tab_k:
        st.write("Đánh giá K-Means với các giá trị $k$ khác nhau để chọn số cụm hợp lý:")
        k_df = evaluate_k_range(features_df, k_min=2, k_max=min(10, len(features_df) - 1))
        if not k_df.empty:
            fig_k, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(k_df["k"], k_df["inertia"], marker="o")
            ax1.set_title("Elbow Method (Inertia)")
            ax1.set_xlabel("Số cụm (k)")
            ax1.set_ylabel("Inertia (SSE)")

            ax2.plot(k_df["k"], k_df["silhouette"], marker="s", color="green")
            ax2.set_title("Silhouette Score (Càng cao càng tốt)")
            ax2.set_xlabel("Số cụm (k)")
            ax2.set_ylabel("Silhouette Score")
            st.pyplot(fig_k)
            plt.close(fig_k)
        else:
            st.info("Cần ít nhất 4 bài hát để vẽ biểu đồ đánh giá dải k.")

    with tab_listen:
        selected_cluster = st.selectbox(
            "Chọn cụm muốn nghe thử:",
            sorted(res.df["cluster"].unique().tolist()),
        )
        sub_df = res.df[res.df["cluster"] == selected_cluster]
        profile_text = res.cluster_profiles.loc[
            res.cluster_profiles["cluster"] == selected_cluster, "profile"
        ].values[0]
        st.markdown(f"**Đặc trưng cụm {selected_cluster}:** {profile_text}")

        for _, row in sub_df.iterrows():
            st.write(
                f"🎵 **{row['filename']}** — Tempo: `{row['tempo_bpm']} BPM` | "
                f"Thời lượng: `{row['duration_sec']}s`"
            )
            fpath = Path(str(row["filepath"]))
            if fpath.exists():
                st.audio(str(fpath))

    with tab_export:
        out_folder = st.text_input(
            "Thư mục xuất kết quả:",
            value=str(Path.cwd() / "clustered_output"),
        )
        copy_audio = st.checkbox("Sao chép tệp nhạc vào các thư mục cụm", value=True)
        if st.button("💾 Xuất thư mục & Playlist (.m3u8, .csv)"):
            out_path = export_clusters_to_folders(
                res.df, out_folder, copy_files=copy_audio
            )
            st.success(f"Đã xuất kết quả thành công vào: `{out_path}`")
