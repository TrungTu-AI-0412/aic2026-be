# Dữ liệu đã dựng — hiện trạng

Cập nhật 15/08/2026. Tài liệu này trả lời ba câu: **đã bổ sung gì vào parquet**,
**mượn được gì từ bên ngoài**, và **còn thiếu gì**.

Mô tả từng cột và các cảnh báo khi dùng nằm ở [`data/ARTIFACTS.md`](../data/ARTIFACTS.md).
Hai bài báo tham chiếu ở [`research/mervin.md`](research/mervin.md) và
[`research/vortex.md`](research/vortex.md).

---

## 1. Bốn file dữ liệu

| File | Dòng | Cột | Vai trò |
| --- | ---: | ---: | --- |
| `frames.parquet` | 177.321 | 19 | manifest chính — mỗi dòng một keyframe |
| `clips.parquet` | 97.811 | 12 | mỗi dòng một shot |
| `video_bounds.parquet` | 873 | 4 | cận trên frame hợp lệ khi nộp bài |
| `eval_set.jsonl` | 300 | — | câu hỏi đánh giá **sinh tự động** |

Tất cả đã đẩy lên `r2:aicc26/artifacts/`, MD5 đã đối chiếu. Output OCR thô nằm ở
`r2:aicc26/artifacts/ocr/` (23 MB) — giữ lại vì đó là 7 giờ GPU.

## 2. Độ phủ

| Tín hiệu | Phủ | Nguồn |
| --- | ---: | --- |
| `shot_id` | 100% | mượn — HF `tanp21/aic-hcmc-2025-videos` |
| `original_frame_id` | 100% | BTC (`map-keyframes`) |
| `objects` / `object_counts` | 95,0% | BTC (Faster R-CNN / OpenImages) |
| `asr_text` | 94,1% | mượn — Kaggle `zzzlazy/aic-asr` (Apache-2.0) |
| `asr_entities` | 57,4% | cùng nguồn ASR |
| **`ocr_text`** | **90,7%** | **tự chạy** — EasyOCR, 2×T4, 7 giờ |
| metadata video | 100% | BTC (`media-info`) |
| caption VLM | **0%** | không làm — xem mục 6 |

10.169/97.811 shot (10,4%) không chứa keyframe nào: shot ngắn hơn lưới
~1 keyframe/giây của BTC. Không phải lỗi; chỉ tìm được qua clip vector.

---

## 3. Mượn được gì

Đây là phần tiết kiệm nhiều thời gian nhất.

### Shot boundaries — bỏ được bước GPU từng deadlock

HF dataset `tanp21/aic-hcmc-2025-videos`, thư mục `annotations/shot_json/`,
phủ đúng **873/873** video.

Không tin ngay mà kiểm chứng bằng `scripts/verify_shots.py`, đối chiếu với
`map-keyframes` + `media-info` **không cần tải 77 GiB video**:

```
873/873 video khớp
drift thời lượng: trung vị 0.24s, tối đa 0.57s
keyframe nằm đúng shot: 99.914%
```

Nhờ vậy bỏ hẳn bước chạy TransNetV2, vốn đã từng treo vì callback logging của
PyAV.

### ASR — bỏ được vấn đề YouTube chặn IP

Ban đầu tự cào bằng `scripts/scrape_transcripts.py`. YouTube chặn IP vĩnh viễn
sau **64/873** video; thử lại 3 lần cách 15 phút rồi 2 lần cách 60 phút đều
về 0, yt-dlp cũng dính 429.

Kaggle `zzzlazy/aic-asr` (**Apache-2.0**) phủ đúng **873/873**, kèm cả bản đã
thêm dấu câu và NER. Vấn đề biến mất hoàn toàn.

64 transcript tự cào được giữ lại làm **đối chứng chéo** — chỗ nào hai nguồn
bất đồng là chỗ ASR không đáng tin.

### Objects — đã có sẵn, không chạy lại YOLO

Cột `objects` + `object_counts` lấy từ artefact BTC. U-CESE xác nhận đó là
Faster R-CNN trên OpenImages. Chạy lại YOLOv8 lên 133k keyframe chỉ để có thứ
đang có là phí GPU.

### Không mượn được: code của các đội

Đã tra kỹ. `chancholat/Lameframes` có code Flask chạy được nhưng **không có
giấy phép** (mặc định giữ toàn quyền). `BaryuH/PIKA` là MIT nhưng cây repo chỉ
có 4 file — không có code. U-CESE, MERVIN, Vortex đều **không công khai code**.

**Không có ground truth công khai** cho cuộc thi này, kể cả trên trang chính
thức. Đó là lý do mục 5 phải tự dựng.

---

## 4. Tự làm gì

### OCR — chữ trên màn hình

EasyOCR trên 2×Tesla T4, **133.757 ảnh** = keyframe đầu và cuối mỗi shot.
Chọn hai đầu vì thanh tin chạy ngang, nên đầu và cuối shot mang hai nửa khác
nhau của một dòng tin; 47% shot chỉ có 1 keyframe nên phần tăng thêm chỉ rơi
vào chỗ thật sự có nhiều chữ để đọc.

**Quyết định quan trọng nhất: không lọc theo confidence.** Vùng chữ thật, nguyên
văn: `'Tam DUnG LuU Thong'` 0.15, `'NGuoi Dan Di Lai CHu Y Quan Sat'` 0.20 —
đọc đúng hết, điểm thấp chỉ vì in hoa không dấu. Cắt ở 0.3 sẽ vứt **29,2% tổng
ký tự**, vứt trúng phần thanh tin. Lọc theo **độ dài** thay thế.

Chúng vẫn tìm được nhờ `app.features.sparse` bỏ dấu — đã kiểm 4/4 và 8/8 token
khớp với câu hỏi gõ có dấu.

**Chất lượng, đo bằng cách lấy từ vựng ASR làm đối chứng chéo:**

| Độ dài text/shot | Số shot | % token đúng | % tổng lượng chữ |
| --- | ---: | ---: | ---: |
| 1–3 token | 36.974 | 62,9% | 5,9% |
| 4–9 | 18.609 | 52,5% | 11,7% |
| 10–24 | 12.777 | 75,4% | 23,5% |
| **25+** | 9.977 | **83,0%** | **59,0%** |

82,5% khối lượng chữ nằm ở hai nhóm dài với độ chính xác 75–83%. Rác dồn vào
shot ít chữ, mà nhóm đó gần như không đóng góp khối lượng.

**80,7% loại token rác chỉ xuất hiện một lần** — không ai gõ trúng, nên trong
index sparse có IDF chúng không tạo kết quả sai, chỉ tốn dung lượng.

Hai giới hạn: phép đo này là **proxy chứ không phải ground truth**, nên không
bắt được lỗi OCR tạo ra một từ có thật khác; và **chưa ai đọc tay xác nhận**.

### OCR bù đúng chỗ ASR sai nặng nhất

Cùng shot `L21_V001 kf6`:

```
ASR : "Đường băng sông Cửu Long của tình trạng sục lún..."
OCR : "TIN CHÍNH TÌNH TRẠNG SỤT LÚN Ở ĐBSCL ĐANG DIỄN RA RẤT NHANH"
```

Mức token: ASR có `sục` (sai), OCR có `sụt` (đúng) và thêm `đbscl`. Đây là lý
do hai kênh phải để **riêng hai sparse vector**, không gộp.

### `video_bounds.parquet` — chặn dòng nộp không thể ghi điểm

`app.submissions.service` đọc file này để từ chối video ngoài corpus hoặc frame
vượt độ dài video, trước khi tiêu một lượt nộp.

Cận trên **cố tình nới**, không siết. `media-info` ghi độ dài bằng **giây
nguyên**, nên `length × fps` luôn ngắn hơn video thật: ở **43/873** video,
keyframe cuối đã vượt qua tích đó tới 10 frame — chính BTC lấy mẫu chúng. Nên
lấy max của hai bằng chứng. Đã kiểm: **0/177.321 keyframe bị từ chối oan.**

### `eval_set.jsonl` — 300 câu hỏi đánh giá

Sinh từ ASR bằng `scripts/build_eval_set.py`, chấm bằng `app.eval.metrics`.

⚠ **Vòng lặp:** câu hỏi lấy từ ASR, mà ASR nằm trong index lexical, nên nhánh
speech **thắng theo thiết kế**. Phải chạy `--no-hybrid` mới đọc trung thực.
Dùng tốt để **bắt hồi quy** và so hai cấu hình cùng thiết lập kênh; không dùng
để nói độ chính xác tuyệt đối.

Bốn sai lầm đã phải sửa khi dựng — mỗi cái đều tạo ra bộ số đẹp mà vô nghĩa:

| Sai lầm | Hậu quả quan sát được | Cách sửa |
| --- | --- | --- |
| Tổng IDF toàn token | Top toàn bài giảng camera tĩnh **157 keyframe/shot** | Chỉ tính 8 token hiếm nhất |
| Cắt 32 từ đầu | Câu hỏi thành *"Xin chào tất cả các bạn"* | Chọn cửa sổ 32 từ nhiều IDF nhất |
| Không lọc token hiếm | IDF **thưởng lỗi ASR**: *"Bi tơ ri Cu nốp"* thắng *"Angelina Jolie"* | Bỏ token có df < 3 |
| Một shot một câu hỏi | Shot kề nhau chung đoạn ASR → hỏi trùng, một đáp án chắc chắn sai | Gom lại, **một câu nhiều shot đúng** |

---

## 5. Hai lỗi âm thầm đã chặn được

Cả hai đều không báo lỗi ra ngoài — đó là chỗ nguy hiểm.

### 614 keyframe lẽ ra đã ghi đè lên nhau

`frame_idx` suy từ presentation timestamp đã làm tròn, nên **hai keyframe liên
tiếp có thể trùng `frame_idx`** — xảy ra ở **192/873 video, tổng 614 keyframe**.
Dùng `(video_id, original_frame_id)` làm khoá point id thì 614 keyframe đó ghi
đè lẫn nhau lúc upsert vào Qdrant, **không có exception nào**.

Khoá đúng là `(video_id, keyframe_n)`. `original_frame_id` là **giá trị đem
nộp**, không phải khoá.

### `batch_builder keyframes` sẽ nộp sai số frame

Nó lấy `original_frame_id` từ **tên file**, mà BTC đặt tên keyframe theo cột
`n` (số thứ tự), không phải `frame_idx`. Dùng nó thì mọi bài nộp đều mang số
sai. Đó là lý do `scripts/build_frames_manifest.py` tồn tại.

### Ngoài lề: ingest chưa bao giờ chạy được

Phát hiện khi kiểm tra vòng đời dữ liệu sau bước nối OCR.
`RecordBatch.to_pylist()` trả cột `map<string,int32>` thành list các cặp tuple,
Pydantic khai `dict[str, int]` nên từ chối — chết ở **dòng đầu tiên**. Đã xác
nhận bản sao lưu trước khi nối cũng hỏng y hệt, nên là bug có sẵn. Đã sửa bằng
validator, kèm test hồi quy.

---

## 6. Còn thiếu gì

### Chặn đường ngay lúc này

| Việc | Vì sao chặn |
| --- | --- |
| **Chưa ingest collection nào** | Qdrant không chạy trên máy này (`exit=7`) |
| **Chưa tải 28,7 GiB ảnh keyframe** | Cột `path` trỏ tới file không tồn tại → không embed được |
| **`app.eval.runner` chưa chạy lần nào** | Cần cả hai thứ trên. Metrics và builder thì đã chạy thật |

Hệ quả: **chưa có một con số đo lường nào** về chất lượng truy hồi. Mọi so sánh
cấu hình vẫn là giả thuyết.

### Chưa làm, có chủ đích

| Việc | Lý do |
| --- | --- |
| Caption VLM | 133k ảnh qua VLM là hàng chục giờ. Vortex dùng Qwen2.5-VL-3B làm cả OCR lẫn caption — rẻ hơn tôi ước tính ban đầu, nhưng vẫn không vừa 3 ngày |
| Đổi sang Jina-CLIP v2 | Bắt embed lại toàn bộ. MERVIN và Vortex đạt 79 và 79.6/88 với hai stack embedding khác hẳn nhau → chọn model không phải chỗ quyết định |
| Chạy lại YOLO | Đã có objects của BTC |
| Duyệt tay `eval_set.jsonl` | Cả 300 dòng đang `reviewed: false`. Không tự động hoá được |
| Làm sạch thêm OCR | Bỏ hết nhóm 1–3 token chỉ cắt 5,9% khối lượng; stoplist watermark gỡ ~1,6%. Không xứng công |

### Đáng lấy từ hai paper, chưa làm

- **Rocchio relevance feedback** (Vortex) — chỉ là số học trên vector đã có,
  không tốn GPU. Thứ duy nhất trong Vortex ta không có mà đáng lấy.
- **Ràng buộc T₂ − T₁ ≤ 5 phút cho TRAKE** (MERVIN) — rẻ, loại được chuỗi trải
  dài vô lý mà `tracks.py` hiện vẫn chấp nhận.

### Ghi nhận cho lần sau

Chia lô OCR theo nhóm `L*` là sai. L26 một mình chiếm **73.303/133.757 ảnh
(55%)**, nên một GPU ôm trọn nó còn GPU kia ngồi chờ; đến lô cuối chỉ còn một
GPU chạy. Tốc độ thực tế 3,5–3,8 ảnh/s **cao hơn** benchmark 2,74 mà tổng thời
gian vẫn vượt ước tính. Shard theo **ảnh** thay vì theo lô có lẽ tiết kiệm được
khoảng một giờ.

---

## 7. Dựng lại từ đầu

```bash
# 1. shots + manifest gốc
python scripts/verify_shots.py --shots DIR --map-keyframes DIR --media-info DIR
python scripts/build_frames_manifest.py --map-keyframes DIR --shots DIR \
    --media-info DIR --objects DIR --asr-csv DIR \
    --out-frames data/frames.parquet --out-clips data/clips.parquet \
    --out-videos data/video_bounds.parquet

# 2. nối OCR (từ JSONL trên R2, không cần chạy lại GPU)
python scripts/join_ocr.py --ocr data/ocr_raw/ocr

# 3. sinh tập đánh giá
python scripts/build_eval_set.py --limit 300
```

Parquet vẫn là nguồn sự thật để dựng lại và đối soát.
