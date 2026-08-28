# MERVIN — Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos

Ghi chép từ [arXiv 2605.16120](https://arxiv.org/html/2605.16120). Hệ thống dự
AI Challenge HCMC 2025, **79/88** vòng loại, thuộc nhóm đội đại học dẫn đầu.
Vòng chung kết: "truy hồi được toàn bộ cảnh cần tìm" (không có số).

**Không có code, không có weight công khai.**

---

## 1. Hai pha

**Tiền xử lý:** TransNetV2 cắt shot → lấy keyframe → lấy transcript (YouTube
Transcript API, Whisper là dự phòng) → LLM làm sạch → trích đặc trưng → nạp
Milvus.

**Truy vấn:** người dùng nhập text tiếng Việt → hai luồng truy hồi (thị giác và
văn bản) chạy độc lập → tinh chỉnh lặp lại nếu cần.

## 2. Keyframe

TransNetV2 cắt shot, rồi từ **mỗi shot lấy đúng 3 keyframe** tại vị trí chuẩn
hoá **0.15, 0.50, 0.85**.

Đây là lựa chọn khác hẳn ta: cố định 3 frame/shot, phân bố đều trong shot,
không phụ thuộc độ dài shot.

## 3. Model

| Tầng | Model | Lý do họ nêu |
| --- | --- | --- |
| Thị giác | **PE-Core-bigG-14-448** (Meta Perception Encoder) | 85.4% top-1 ImageNet-1K, 76.9% Kinetics-400; hơn CLIP ViT-H/14 và OpenCLIP ở zero-shot |
| Văn bản | **dangvantuan/vietnamese-embedding** | Vietnamese STS: Pearson 0.8502, Spearman 0.8499 |
| ASR | YouTube Transcript API, Whisper dự phòng | |
| OCR | **không đề cập** | |
| Object detection | **không đề cập** | |

PE-Core hỗ trợ ngữ cảnh text 72 token và được pretrain trên video tổng hợp để
biểu diễn nhất quán theo chuyển động.

## 4. ⚠ Không dịch câu hỏi

**Câu hỏi tiếng Việt được đưa thẳng vào model embedding tiếng Việt, không qua
bước dịch sang tiếng Anh.**

Đây là điểm quan trọng nhất của paper này với ta. Mục 5 trong `task.md` định
dùng LLM dịch Việt→Anh rồi mở rộng 4–5 câu. MERVIN đạt 79/88 mà bỏ qua toàn bộ
bước đó, bằng cách chọn model hiểu tiếng Việt sẵn.

Bốn chế độ tìm kiếm chạy trên bốn index riêng, độc lập nhau: theo frame, theo
transcript, theo thời gian, theo tóm tắt video.

## 5. Làm sạch transcript bằng Gemini 1.5 Flash

Hai giai đoạn:

1. **Cleaning** — bỏ token không nhận dạng được, chuẩn hoá biến thể dấu, xử lý
   cụm từ nhập nhằng theo ngữ cảnh. **~8.000 token/video.**
2. **Summarization** — sinh biểu diễn cô đọng ở mức sự kiện. **~3.000–4.000
   token/video.**

Prompt không được công bố.

Đây là dịch vụ đám mây, nhưng nằm ở **tiền xử lý** nên không vi phạm ràng buộc
"không phụ thuộc cloud trong query path". Ta đã có thứ tương đương: cột
`asr_text_corrected` trong `frames.parquet` đến từ dataset `zzzlazy/aic-asr`.

## 6. Lưu trữ

**Milvus**, với embedding của transcript và của summary để ở **hai database
tách rời**. Không dùng Elasticsearch hay bất kỳ text search engine truyền thống
nào — truy hồi thuần bằng embedding.

Đây là khác biệt lớn với Vortex và với ta: MERVIN **không có nhánh lexical**.

## 7. TRAKE / truy vấn thời gian

```
S_video = (10.0 × S_pair) + (5.0 × (S̄₁ + S̄₂))
```

- `S_pair` — điểm của cặp tuần tự tốt nhất
- `S̄₁, S̄₂` — điểm trung bình top-10 của từng sự kiện riêng lẻ

Ràng buộc thời gian: **T₂ > T₁** và **T₂ − T₁ ≤ 5 phút**.

Trọng số 10:5 nghĩa là một cặp khớp đúng thứ tự đáng gấp đôi tổng chất lượng
trung bình của hai sự kiện rời.

## 8. Điểm số

| Vòng | Điểm | Tỉ lệ |
| --- | ---: | ---: |
| 1 | 18/23 | 78.3% |
| 2 | 27/30 | 90.0% |
| 3 | 34/35 | **97.1%** |
| | **79/88** | 89.8% |

Khác Vortex, tiến triển của MERVIN **đơn điệu tăng** kể cả sau khi chuẩn hoá
theo số câu. Nhưng vẫn là ba bộ câu hỏi khác nhau, nên vẫn không phải ablation
có kiểm soát — chỉ là nó không tự mâu thuẫn như Vortex.

Cách chấm: **Mean of Top-k R-Scores**, k ∈ {1, 5, 20, 50, 100}.

## 9. ⚠ Phần cứng

> AMD Ryzen 5 5600G (12 nhân, 3.9 GHz), **32 GB RAM**, **NVIDIA RTX 3060
> 12 GB VRAM**. Mạng 309 Mbps xuống / 317 Mbps lên, độ trễ 2 ms.

Đây là thông tin `task.md` đang thiếu. Mục 2 ghi "GPU tối thiểu **4 GB** VRAM".
Một đội đạt 79/88 chạy trên **12 GB**. Với profile SigLIP2-giant (1536 chiều)
cộng reranker BLIP ITM của ta, 4 GB là không đủ.

## 10. Giao diện vận hành

React, bốn module: tìm theo frame, tìm theo transcript (trả về mức đoạn kèm
keyframe), tìm theo thời gian (hai sự kiện, có ràng buộc thời gian), tìm theo
tóm tắt (mức video, không chỉ frame).

Trang nộp bài cho phép **kiểm chứng frame bằng FFmpeg** trước khi sinh file
nộp. Phát video qua YouTube Player API, số frame tính tại chỗ từ timestamp và
FPS.

## 11. Hạn chế họ tự nêu

- Sửa lỗi ASR theo miền chuyên biệt còn bỏ ngỏ
- Chưa có LLM agent tự động hoá quy trình — hiện **phụ thuộc chỉnh tay**

## 12. Đối chiếu với repo này

| MERVIN | Ta |
| --- | --- |
| TransNetV2 → 3 keyframe/shot @ 0.15/0.50/0.85 | shot BTC, keyframe BTC ~1/giây |
| PE-Core-bigG-14-448 | SigLIP2 (giant hoặc so400m) |
| Vietnamese embedding, **không dịch** | SigLIP2, câu hỏi cần tiếng Anh |
| Không có nhánh lexical | sparse vector IDF trong Qdrant |
| Gemini làm sạch transcript | `asr_text_corrected` có sẵn từ dataset |
| Milvus, nhiều DB tách rời | Qdrant một collection, nhiều named vector |
| TRAKE: 10·S_pair + 5·(S̄₁+S̄₂), ≤5 phút | `tracks.py` chọn chuỗi frame tăng dần |
| Kiểm chứng frame bằng FFmpeg trước khi nộp | `LocalSubmissionService` chặn theo `video_bounds.parquet` |

**Đáng lấy ngay:** ràng buộc **T₂ − T₁ ≤ 5 phút** cho TRAKE. Rẻ, và loại được
những chuỗi trải dài vô lý mà `tracks.py` hiện vẫn chấp nhận.

**Đáng cân nhắc sau kỳ thi:** hướng embedding tiếng Việt trực tiếp. Nó thay thế
cả mục 5 của `task.md` chứ không bổ sung.

**Đã có, khỏi làm:** làm sạch transcript bằng LLM.

---

## Đọc chung hai paper

| | MERVIN | Vortex |
| --- | --- | --- |
| Điểm | 79/88 | 79.6/88 |
| Thị giác | PE-Core-bigG | CLIP DFN5B + SigLIP2 |
| Tiếng Việt | model tiếng Việt, không dịch | không nói rõ |
| Lexical | **không có** | Elasticsearch |
| OCR | không đề cập | Qwen2.5-VL-3B |
| Vector DB | Milvus | Milvus + ES + Redis |
| Phần cứng | RTX 3060 12 GB | không công bố |

Hai stack **khác nhau gần như hoàn toàn**, điểm chênh **0.6/88**. MERVIN thậm
chí không có nhánh lexical lẫn OCR mà vẫn ngang Vortex.

Kết luận rút ra: **chọn model embedding không phải chỗ quyết định thắng thua.**
Nó củng cố việc không nên đổi SigLIP2 sang Jina-CLIP trong 3 ngày còn lại — chi
phí là embed lại toàn bộ, mà bằng chứng cho phần thưởng thì không có.

Chỗ hai đội đầu tư mà ta chưa có, đều nằm ở **vòng lặp người vận hành**:
Rocchio feedback (Vortex), kiểm chứng frame trước khi nộp (MERVIN), ràng buộc
thời gian cho TRAKE (cả hai).
