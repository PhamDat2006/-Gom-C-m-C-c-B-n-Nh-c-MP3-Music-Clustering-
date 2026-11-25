# 🎵 Ứng Dụng Gom Cụm Các Bản Nhạc MP3 (Music Clustering)

Ứng dụng phân tích và gom cụm tự động các bản nhạc (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`) theo sự tương đồng về âm thanh bằng Machine Learning không giám sát (*Unsupervised Learning*), kèm giao diện web tương tác **Streamlit** và **Jupyter Notebook**.

---

## 1. Giải pháp & Quy trình xử lý

Hệ thống thực hiện gom cụm dựa trên nội dung âm thanh (*Content-based Audio Clustering*) theo 5 bước:

1. **Đọc & Tiền xử lý âm thanh:**
   - Tải tệp âm thanh bằng `librosa` ở tần số lấy mẫu `22050 Hz`, kênh đơn (`mono`).
   - Tự động cắt khoảng lặng đầu/cuối (`librosa.effects.trim`) để tránh sai lệch giá trị trung bình phổ tần.
2. **Trích xuất 51 đặc trưng âm thanh:**
   - **Nhịp điệu & Năng lượng:** `Tempo (BPM)`, `RMS Energy` (mean/std), `Zero-Crossing Rate` (mean/std).
   - **Đặc trưng phổ tần (Spectral):** `Spectral Centroid`, `Spectral Bandwidth`, `Spectral Rolloff`, `Spectral Contrast` (mean/std).
   - **Hòa âm & Âm sắc:** `Chroma STFT` (12 bậc cao độ) và `MFCC` (13 hệ số `mean` & `std`).
3. **Chuẩn hóa & Giảm chiều:**
   - Chuẩn hóa đặc trưng bằng `StandardScaler`.
   - Giảm chiều xuống 2D bằng `PCA` để trực quan hóa phân bố cụm.
4. **Gom cụm không giám sát:**
   - **K-Means:** Hỗ trợ tìm số cụm tối ưu $k$ qua **Elbow Method** và **Silhouette Score**.
   - **Agglomerative Clustering:** Gom cụm phân cấp theo liên kết Ward.
   - **DBSCAN:** Gom cụm theo mật độ, tự động phát hiện các bản nhạc ngoại lai (`Noise / Outlier`).
5. **Đánh giá & Xuất kết quả:**
   - Đánh giá chất lượng cụm bằng `Silhouette Score`, `Davies-Bouldin Index`, `Calinski-Harabasz Index`.
   - Tự động mô tả đặc trưng từng cụm và xuất ra thư mục con, playlist `.m3u8`, báo cáo `.csv`.

---

## 2. Cấu trúc thư mục

```text
.
├── app.py                          # Giao diện web tương tác Streamlit
├── music_clusterer.py              # Module cốt lõi: trích xuất đặc trưng, gom cụm, xuất file
├── 079206008013-PhamTanDat.ipynb   # Jupyter Notebook trình bày và chạy thử nghiệm từng bước
├── sample_music/                   # Thư mục chứa nhạc đầu vào (có sẵn 9 file MP3 mẫu)
├── clustered_output/               # Thư mục xuất kết quả sau khi gom cụm
│   ├── Cluster_0/                  # Các bài hát thuộc cụm 0
│   ├── Cluster_1/                  # Các bài hát thuộc cụm 1
│   ├── Cluster_2/                  # Các bài hát thuộc cụm 2
│   ├── Cluster_*.m3u8              # Playlist phát nhạc theo từng cụm
│   └── clusters_summary.csv        # Bảng tổng hợp đặc trưng và nhãn cụm
└── README.md                       # Tài liệu hướng dẫn dự án
```

---

## 3. Yêu cầu & Cài đặt

Yêu cầu Python 3.10+ (đã kiểm thử trên Python 3.13). Cài đặt các thư viện cần thiết:

```bash
pip install librosa soundfile streamlit scikit-learn pandas numpy matplotlib seaborn
```

---

## 4. Hướng dẫn sử dụng

### Cách 1: Chạy ứng dụng Web (Streamlit)

Mở terminal tại thư mục dự án và chạy:

```bash
streamlit run app.py
```

Trên giao diện web, bạn có thể:
- Chọn đường dẫn thư mục chứa nhạc trên máy (mặc định là `sample_music/`) hoặc tải lên trực tiếp các tệp `.mp3`.
- Tùy chỉnh thời lượng trích xuất mỗi bài (mặc định 30 giây đầu) và chọn thuật toán (`KMeans`, `Agglomerative`, `DBSCAN`).
- Xem biểu đồ phân cụm PCA 2D, biểu đồ Elbow & Silhouette, nghe thử nhạc trực tiếp theo từng cụm, và xuất kết quả ra thư mục/playlist.

### Cách 2: Chạy trên Jupyter Notebook

Mở tệp `079206008013-PhamTanDat.ipynb` bằng Jupyter hoặc VS Code và chạy lần lượt các ô mã nguồn để xem chi tiết từng bước xử lý và trực quan hóa.
