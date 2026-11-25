# 🎵 Ứng Dụng Gom Cụm Các Bản Nhạc MP3 (Music Clustering)

Hệ thống phân tích và gom cụm tự động các bản nhạc (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`) theo đặc trưng âm thanh bằng Machine Learning không giám sát (*Unsupervised Learning*), được kiểm chứng trên bộ dữ liệu thực tế **FMA Small** (`8.000` bài hát MP3, thời lượng `30 giây`/bài) và tích hợp ứng dụng web **Streamlit** để demo trên bất kỳ tệp MP3 nào.

---

## 1. Giải pháp & Quy trình xử lý

1. **Đọc & Tiền xử lý âm thanh:**
   - Tải tệp âm thanh bằng `librosa` (`sr=22050 Hz`, `mono`, mặc định `30 giây`), tự động cắt khoảng lặng đầu/cuối (`librosa.effects.trim`).
2. **Trích xuất đặc trưng âm thanh mở rộng (149 đặc trưng số):**
   - **Nhịp điệu:** `Tempo (BPM)`.
   - **Năng lượng & Độ gắt:** `RMS Energy`, `Zero-Crossing Rate (ZCR)`.
   - **Phổ tần (Spectral):** `Spectral Centroid`, `Bandwidth`, `Rolloff`, `Contrast`.
   - **Hòa âm & Âm sắc:** `Chroma STFT` (12 bậc), `Tonnetz` (6 chiều hòa âm), `MFCC` (13 hệ số).
   - Mỗi nhóm đặc trưng trên đều tính đủ 4 thống kê: `mean`, `std`, `skew` (độ lệch) và `kurtosis` (độ nhọn).
3. **Xử lý dữ liệu lớn (Scalability):**
   - Trích xuất song song đa nhân CPU (`ProcessPoolExecutor`).
   - Hỗ trợ lưu **checkpoint CSV** định kỳ để tiếp tục khi bị gián đoạn và nạp lại từ cache trên giao diện Streamlit mà không cần trích xuất lại từ đầu.
4. **Thuật toán gom cụm & Giảm chiều:**
   - **K-Means** (tối ưu $k$ qua Elbow Method & Silhouette Score).
   - **Agglomerative Clustering** (Ward linkage).
   - **DBSCAN** với cơ chế tự động ước lượng `eps` theo phân vị khoảng cách $k$-láng giềng gần nhất (`estimate_dbscan_eps`).
   - Hỗ trợ giảm nhiễu bằng **PCA** (giữ 90% phương sai) trước khi gom cụm và chiếu 2D để trực quan hóa.
5. **Đánh giá cụm:**
   - **Nội tại (Internal):** `Silhouette Score`, `Davies-Bouldin Index`, `Calinski-Harabasz Index`.
   - **Đối chiếu nhãn thật (External với `genre_top` của FMA):** `NMI`, `ARI`, `Purity` và bảng phân bố thể loại theo cụm (*Contingency Table*).

---

## 2. Kết quả thực nghiệm trên bộ dữ liệu FMA Small

- **Dữ liệu:** `fma_small` (`8.000` tệp MP3) + `fma_metadata/tracks.csv` (`8` thể loại cân bằng: *Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock*).
- **Trích xuất thực tế:** Thành công **7.994 / 8.000** bản nhạc (6 tệp trong bộ FMA gốc bị lỗi mã hóa MP3 được bộ xử lý lỗi bỏ qua an toàn), tạo bảng đặc trưng `153` cột (`149` đặc trưng âm thanh + metadata).

| Thuật toán | Silhouette | Davies-Bouldin | Calinski-Harabasz | NMI | ARI | Purity | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means (k=8, PCA=False)** | 0.0237 | 3.0588 | 319.02 | 0.1493 | **0.0964** | 31.05% | Tách rõ Hip-Hop (638), Rock (546), Folk (405) |
| **K-Means (k=8, PCA=True)** | 0.0243 | **2.8800** | **366.02** | **0.1505** | 0.0939 | **32.52%** | Kết quả cân bằng tốt nhất (vượt xa mức ngẫu nhiên 12.5%) |
| **Agglomerative (k=8, PCA=True)** | 0.0164 | 2.9179 | 301.92 | 0.1262 | 0.0848 | 30.34% | Tách tốt Hip-Hop (642), Rock (535), Folk (437) |
| **DBSCAN (PCA=True, auto eps=11.334)** | 0.3416 | 5.6362 | 104.89 | 0.0149 | 0.0014 | 14.25% | Tìm 1 cụm đặc chính + 538 bài ngoại lai (*Experimental/Instrumental/Electronic*) |

> **Nhận xét:** Không gian đặc trưng âm nhạc của 8 thể loại giao thoa liên tục nên các thuật toán phân hoạch (`K-Means`, `Agglomerative`) đạt `Purity` ~`31–32.5%` (gấp ~2.6 lần ngẫu nhiên `12.5%`), gom rõ các thể loại có đặc trưng nhịp/âm sắc riêng biệt như **Hip-Hop**, **Rock**, **Folk/Instrumental**, và **Electronic** vào các cụm riêng; trong khi **DBSCAN** hữu ích để phát hiện các bản nhạc ngoại lai (`538` bài).

---

## 3. Cấu trúc thư mục

```text
.
├── app.py                          # Giao diện web tương tác Streamlit (demo trên mọi MP3 hoặc nạp CSV cache)
├── music_clusterer.py              # Module cốt lõi: trích xuất 149 đặc trưng song song, checkpoint, gom cụm, NMI/ARI
├── run_fma_clustering.py           # Script chạy hàng loạt trên toàn bộ fma_small và đánh giá với tracks.csv
├── 079206008013-PhamTanDat.ipynb   # Jupyter Notebook trình bày và chạy thử nghiệm từng bước
├── sample_music/                   # 9 tệp MP3 mẫu (30 giây/bài) để kiểm thử nhanh
└── README.md                       # Tài liệu hướng dẫn dự án
```

*(Các thư mục dữ liệu lớn `fma_small/`, `fma_metadata/`, `clustered_output/` và tệp cache `*.csv` được loại trừ khỏi Git trong `.gitignore`.)*

---

## 4. Cài đặt & Sử dụng

### Cài đặt thư viện
```bash
pip install librosa soundfile streamlit scikit-learn pandas numpy matplotlib seaborn scipy
```

### Chạy ứng dụng Web (Streamlit) để demo trên MP3 bất kỳ
```bash
streamlit run app.py
```
- Hỗ trợ chọn thư mục MP3 trên máy, tải lên tệp MP3 trực tiếp, hoặc nạp nhanh tệp đặc trưng `fma_small_features.csv` đã lưu.
- Cho phép xem biểu đồ PCA 2D, đánh giá NMI/ARI và Heatmap thể loại theo cụm (khi có `fma_metadata/tracks.csv`), nghe thử nhạc từng cụm, và xuất playlist `.m3u8`.

### Chạy đánh giá toàn bộ FMA Small
1. Tải và giải nén `fma_small.zip` và `fma_metadata.zip` vào thư mục dự án (`fma_small/` và `fma_metadata/`).
2. Chạy:
```bash
python run_fma_clustering.py
```
Hoặc mở `079206008013-PhamTanDat.ipynb` và chạy từng ô mã nguồn.
