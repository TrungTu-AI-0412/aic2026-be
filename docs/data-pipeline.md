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

---

## 8. Dựng lại từ video thô (đường đi hiện tại)

Mục 7 ở trên cần `map-keyframes` và ảnh keyframe của ban tổ chức. **Cả hai đều
không có trên máy này**: 177.321 đường dẫn trong `frames.parquet` cũ đều trỏ vào
file không tồn tại. Nguồn ảnh duy nhất dùng được là `data/videos` (873 file,
78 GB), nên toàn bộ pipeline chạy lại từ video thô.

```bash
./scripts/ingest_all.sh          # chạy trong tmux, ~9 giờ
```

Thời gian **đã đo** trên 1× L40S / 4 vCPU, không phải phỏng đoán:

| Bước | Thông lượng | Toàn bộ corpus |
| --- | ---: | ---: |
| probe | — | ~1 phút |
| shot detection (TransNetV2, 3 worker) | 737 fps | **~4,6 h** |
| sampling 3 keyframe/shot (3 worker) | 1.094 fps | **~3,1 h** |
| build_asr_manifest | — | ~30 s |
| embed frame (SigLIP2 so400m) | 83 điểm/s | ~1 h |
| embed ASR (Qwen3-0.6B) | 85 điểm/s | ~7 phút |

Đầu ra: ~293k keyframe (**26 GB** JPEG, không phải 50 GB như dự tính ban đầu) và
35.202 segment ASR.

### Vì sao dùng process, không dùng thread

`app/features/media.py` ghi lại một deadlock thật: PyAV đẩy log của FFmpeg vào
logging của Python từ chính thread đang decode, giành GIL, trong khi thread chính
đang giữ GIL bên trong `avcodec_free_context()`. `transformers` cài lại callback
đó mỗi lần import.

Đã thử `av.logging.restore_default_callback()`: hết treo trong một lần chạy
3.000 frame và nhanh hơn 1,65× (705 fps so với 428 fps). Nhưng đây là race, một
lần chạy đúng không chứng minh được gì. Ba process, mỗi process decode
đơn luồng, cho **1.284 fps** — vừa an toàn hơn vừa nhanh hơn. Giữ
`thread_type = "NONE"` ở mọi nơi.

Đo thêm: 4 worker chỉ được 763 fps so với 737 fps của 3 worker, tức CPU đã bão
hoà ở 3. Mặc định là 3, chừa một core cho process cha.

### Đổi model ảnh — quy trình

`FEATURE_PROFILE` gắn chặt với collection: số chiều dense cố định lúc tạo
collection. Nên **đổi model luôn có nghĩa là tạo collection mới**, không bao giờ
cập nhật tại chỗ. Nếu profile và collection lệch nhau, mọi truy vấn rơi vào
không gian vector sai và trả về kết quả trông hợp lý nhưng vô nghĩa — không có
lỗi nào được nêu.

Slot dense được ghi có số chiều lấy từ profile của **chính job đó**, nên so sánh
hai model là chuyện làm được:

```bash
# cùng một manifest, hai collection, hai profile
curl -sX POST localhost:8000/api/v1/ingestions -H 'content-type: application/json' \
  -d '{"entity":"frames","manifest_path":"manifests/frames.parquet",
       "collection_name":"probe-giant",
       "feature_profile":"siglip2-giant-opt-patch16-384-v1"}'
curl -sX POST localhost:8000/api/v1/ingestions -H 'content-type: application/json' \
  -d '{"entity":"frames","manifest_path":"manifests/frames.parquet",
       "collection_name":"probe-so400m",
       "feature_profile":"siglip2-so400m-patch14-384-v1"}'
```

Rồi trỏ `QDRANT_FRAMES_COLLECTION` + `FEATURE_PROFILE` vào từng cái và so cùng
một bộ câu hỏi. **Phải đổi cả hai cùng lúc.** Dùng `--limit` ở bước sampling để
so trên vài chục video thay vì chạy lại 9 giờ.

Thêm model chưa có trong `FEATURE_PROFILES`: khai báo
`FeatureProfile(model_id=..., dimension=..., kind="image"|"text")` và kiểm số
chiều với `hidden_size` trong config của model. Sai số chiều chỉ lộ ra ở bước
upsert, sau khi đã trả xong toàn bộ chi phí embed.

`siglip2-giant-opt-patch16-384` **không có trong HF cache** — phải tải ~4 GB.
`scripts/ingest_all.sh` tải ở bước 0 rồi kiểm lại bằng `HF_HUB_OFFLINE=1`, vì
đường truy vấn lúc thi không được phép ra mạng.

### Hai collection, không phải một

Segment ASR là **khoảng thời gian**, keyframe là **một thời điểm**; hai thứ không
có khoá chung. Nên tách hai collection và nối lại theo thời gian lúc truy vấn
(`app/ranking/asr.py`).

| Collection | Vector | Có dữ liệu |
| --- | --- | --- |
| `frames` | `dense_video` (ảnh) | có |
| | `dense_text` | chưa — dành cho caption VLM |
| | `ocr` (sparse) | chưa — chờ `join_ocr` re-upsert |
| `asr` | `dense_text` (Qwen3-0.6B) | có |
| | `speech` (sparse BM25 + IDF) | có |

Frames **không khai báo slot `speech`**. Nếu có, cùng một đoạn lời nói sẽ được
tính điểm hai lần: một lần qua RRF trong collection frames, một lần qua overlap
bonus. Hệ quả cần biết: lúc này frames không có sparse vector nào được ghi, nên
tìm frame là thuần dense, và `sparse_names` phải để rỗng — truy vấn một slot
chưa có điểm nào sẽ lỗi thẳng.

### ASR: chỉ giữ `text_corrected`

Luật cũ của dự án là giữ cả hai cột. Đã đo lại trước khi bỏ: trong
40.023 segment, số segment **có `text` thô mà không có `text_corrected` là 0**
(36.003 so với 35.997). Bỏ cột thô không mất segment nào tra được, và BM25 vốn
đã lowercase + bỏ dấu câu.

Mất mát thật là về mặt định tính: `text_corrected` trôi chảy nhưng vẫn sai — LLM
chỉ thêm dấu câu, từ nghe sai vẫn sai và giờ trông như câu đúng. Cột thô là dấu
hiệu duy nhất cho biết đoạn đó không đáng tin. Đó là thứ cho người soi lại kết
quả, không phải tín hiệu truy xuất, và `data/transcripts/` vẫn giữ cả hai cột.

**Bỏ segment dưới 2 từ.** 801 segment một từ gần như đều là tiếng đệm — "Ừ", "À",
"thì", "Ờ", "và", "Dạ". Không chỉ vô dụng mà còn có hại: một transcript một chữ
vẫn sinh ra vector dense, và vì điểm được chuẩn hoá với hit tốt nhất bằng 1.0,
segment đó **đã thực sự** vượt lên trên lời nói đúng chủ đề và trao cho frame
toàn bộ overlap bonus. Ngưỡng dừng ở 1 từ có chủ đích: segment hai từ có thể là
tên người ("Xuân Sơn"), đúng thứ một câu truy vấn hay hỏi.

Entity tách thành ba trường lọc riêng, không gộp một danh sách:

| Trường | Segment | Lần xuất hiện |
| --- | ---: | ---: |
| `asr_locations` | 5.024 | 10.941 |
| `asr_persons` | 3.349 | 5.060 |
| `asr_orgs` | 2.311 | 2.903 |

Nhóm `others` bị bỏ: 900 segment, không có nghĩa xác định.

Bỏ `publish_date` và `keywords` khỏi mọi payload. `keywords` chỉ từng tồn tại để
nhồi thêm cho sparse vector `speech` của frame — thứ giờ không còn; `publish_date`
là chuỗi `dd/mm/yyyy`, sắp xếp sai và lọc kém.

### ASR overlap bonus

Truy vấn tìm cả hai collection. Mỗi frame được cộng
`asr_weight × điểm segment tốt nhất phủ nó về thời gian`:

- **Cộng, không nhân.** 4,5% thời lượng video không có segment nào và 22/873
  video không có transcript, nên "không có lời nói" không bao giờ được coi là
  bằng chứng chống lại một frame.
- **Segment tốt nhất, không phải tổng.** Một shot dài phủ nhiều segment; cộng dồn
  sẽ thưởng cho độ dài shot thay vì độ liên quan.
- **Trước dedupe.** Nhờ vậy lời nói quyết định cả *frame nào* đại diện cho shot,
  chứ không chỉ shot đó xếp thứ mấy.
- **Dense nặng hơn sparse** (0,7 / 0,3). RRF của Qdrant gộp theo *hạng* và không
  có chỗ đặt trọng số, nên hai nhánh được truy vấn riêng rồi chuẩn hoá min-max và
  cộng theo trọng số trong `ranking/asr.py`.

Bật/tắt và chỉnh trọng số theo từng request: `asr_enabled`, `asr_weight`,
`asr_dense_weight`, `asr_sparse_weight` trên cả ba endpoint tìm kiếm.

**Đừng chỉnh `ASR_WEIGHT` dựa trên `data/eval_set.jsonl`.** Tập đó sinh ra *từ*
chính ASR, nên nó sẽ luôn ưu ái mọi tín hiệu dựa trên ASR — tăng trọng số sẽ
trông như cải thiện bất kể thực tế. Dùng câu hỏi tự viết, hoặc chạy ablation với
`--no-hybrid`.
