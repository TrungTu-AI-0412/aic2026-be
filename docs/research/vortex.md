# Vortex — Multi-Modal Fusion System for Intelligent Video Retrieval

Ghi chép từ [arXiv 2606.19682](https://arxiv.org/html/2606.19682). Hệ thống dự
AI Challenge HCMC 2025, **79.6/88 (90.5%)** vòng loại. Vòng chung kết được ban
giám khảo chấm: Overall *Excellent*, Q&A *Outstanding*, TKIS *Excellent*,
VKIS và TRAKE *Very Good*.

**Không có code công khai.** Đây là note đọc paper, không phải thứ clone về được.

---

## 1. Đường ống tiền xử lý

Hai tầng, không phải một:

1. **AutoShot** cắt video thành shot.
2. Trong mỗi shot, chọn keyframe thích ứng bằng embedding **CLIP
   ViT-L-14-quickgelu (pretrain DFN2B)**, lấy mẫu **mỗi 8 frame**, giữ lại frame
   khi độ lệch tương đối vượt ngưỡng:

   ```
   rel_diff = ‖e_current − e_prev‖ / ‖e_prev‖ > 0.4
   ```

Paper tự nhận hạn chế của cách này: sự kiện diễn ra **lọt giữa hai mốc 8 frame
sẽ bị bỏ sót**. Đây là hạn chế duy nhất họ nêu rõ.

## 2. Model từng tầng

| Tầng | Model | Ghi chú |
| --- | --- | --- |
| Đặc trưng để lọc keyframe | CLIP ViT-L-14-quickgelu | pretrain DFN2B |
| Truy hồi ngữ nghĩa tổng thể | CLIP | biến thể **DFN5B** |
| Nhận dạng chi tiết | **SigLIP2** | |
| OCR **và** captioning | **Qwen2.5-VL-3B-Instruct** | một model làm cả hai |
| ASR | Whisper | transcript có timestamp |

Điểm đáng chú ý nhất: **một model 3B làm cả OCR lẫn caption**. Không phải hai
pipeline riêng. Paper gọi đó là "đánh đổi tốt giữa độ chính xác và hiệu năng".

Với ASR, khoảng lặng được **kéo dài text nói cuối cùng về phía trước** để mọi
keyframe đều có text phủ — không để trống.

## 3. Tầng lưu trữ

Ba thành phần, không phải một:

- **Milvus** — vector CLIP + SigLIP2, index HNSW
- **Elasticsearch** — index text và lọc metadata cho OCR/ASR
- **Redis** — cache độ trễ thấp

## 4. RRF

```
RRF_Score(d) = Σ(i=1..N) 1 / (k + rank_i(d))
```

với **N = 2** (CLIP và SigLIP2) và **k = 60**.

Chú ý: họ fuse **hai model dense với nhau**, không phải dense với lexical. Phần
text (OCR/ASR) nằm ở Elasticsearch, đi đường riêng.

## 5. Thuật toán tìm kiếm theo thời gian

Nhận ba thành phần: *Previous*, *Current*, *Next*. Chạy **ba lần tìm vector độc
lập**, rồi chấm lại kết quả *Current*:

```
S_final(r_c) = S(r_c) + max S(r_p ∈ cùng video) + max S(r_n ∈ cùng video)
```

Độ phức tạp **O(K log K)**, chi phối bởi bước sắp xếp cuối. Paper nói rõ họ
**cố tình tránh quy hoạch động** vì nó không hợp với truy hồi tương tác.

Hạn chế: người dùng phải **tự tay tách câu hỏi thành ba phần**. Không có bước
phân tích câu hỏi tự động.

## 6. Phản hồi người dùng — Rocchio

Người vận hành đánh dấu "Prefer" / "Not prefer" trên keyframe trả về, vector
truy vấn được cập nhật:

```
q_m = α·q_0 + β·(1/|C_r|)·Σ d_j∈C_r − γ·(1/|C_nr|)·Σ d_j∈C_nr
```

**Paper không công bố α, β, γ.**

## 7. ⚠ Điểm số — đọc đúng cách

Con số thô:

| Vòng | Số câu | Điểm | Cấu hình |
| --- | ---: | ---: | --- |
| 1 | 24 | 20.6 | chỉ CLIP |
| 2 | 30 | 27.8 | + RRF hybrid |
| 3 | 35 | 31.2 | + Temporal + Feedback |
| | | **79.6/88** | |

**Đây không phải ablation.** Ba vòng dùng **ba bộ câu hỏi khác nhau**, nên mức
tăng lẫn lộn giữa "hệ thống tốt lên" và "câu hỏi dễ hơn".

Tệ hơn: chuẩn hoá theo số câu thì bức tranh đảo chiều:

| Vòng | Tỉ lệ |
| --- | ---: |
| 1 | 20.6/24 = **85.8%** |
| 2 | 27.8/30 = **92.7%** |
| 3 | 31.2/35 = **89.1%** |

Vòng 3 — vòng có **nhiều tính năng nhất** — lại có tỉ lệ **thấp hơn** vòng 2.
Nói cách khác, con số này **không chứng minh** được RRF hybrid hay temporal
search đóng góp bao nhiêu.

(Lưu ý: 23+30+35 = 88, khớp tổng; paper ghi vòng 1 có 24 câu nên có một chỗ
lệch 1 — mẫu số vòng 1 nên coi là 23 hoặc 24, không đổi kết luận.)

Cách chấm chính thức: **Mean of Top-k R-Score**, k ∈ {1, 5, 20, 50, 100}, nộp
tối đa **100 kết quả xếp hạng** cho mỗi câu.

## 8. Những chỗ paper để trống

- Prompt cho Qwen2.5-VL (cả OCR lẫn caption)
- Giá trị α, β, γ của Rocchio
- **Toàn bộ phần cứng** — không một dòng nào
- Độ trễ, throughput
- Thống kê kích thước dữ liệu

## 9. Đối chiếu với repo này

| Vortex | Ta |
| --- | --- |
| AutoShot | shot của BTC (đã kiểm: drift trung vị 0.24s) |
| Keyframe thích ứng, ngưỡng 0.4 | keyframe BTC ~1/giây |
| CLIP DFN5B + SigLIP2, RRF k=60 | SigLIP2 đơn + fusion frame/clip |
| Milvus + Elasticsearch + Redis | Qdrant, sparse vector IDF ngay trong Qdrant |
| Qwen2.5-VL-3B cho OCR | EasyOCR trên 2×T4 |
| Rocchio feedback | **chưa có** |
| Temporal 3 vế thủ công | `tracks.py` TRAKE tự chọn chuỗi tăng dần |

**Đáng lấy:** cơ chế Rocchio. Đó là thứ duy nhất trong Vortex mà ta không có
và không tốn GPU — chỉ là số học trên vector đã có. Nó cũng khớp với nhận định
rằng điểm nằm ở vòng lặp người vận hành.

**Không đáng lấy:** thêm một model dense thứ hai để RRF. Tốn một lần embed lại
toàn bộ, mà chính số liệu của họ không chứng minh được nó đáng.

**Ta đã làm khác và có lý do:** ta nhét sparse vector vào thẳng Qdrant thay vì
dựng Elasticsearch riêng. Ít một hệ thống phải vận hành trong phòng thi.
