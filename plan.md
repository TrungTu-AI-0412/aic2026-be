# Implementation plan — AIC 2026 Backend

Kế hoạch cải thiện backend theo thứ tự ảnh hưởng tới điểm thi. Mỗi item được
scope để ship độc lập, đúng review rule "keep changes scoped to one
implementation-plan item".

Trạng thái tại thời điểm lập plan (2026-08-13), lấy từ code và `ingestion.db`:

| Sự kiện | Bằng chứng |
| --- | --- |
| Frame collection đã build xong | `ing-b38d6f05c7`, 5314/5314, profile `siglip2-so400m-patch14-384-v1` |
| Clip collection **chưa có data** | `ing-be427841f5` failed: `runner deadlocked in PyAV log callback during clip decode` |
| Bug gây fail đã được vá | commit `af47014` tắt multi-threaded decode |
| Export bài nộp **chưa chạy được** | `services/submissions.py:21` là Protocol rỗng; `runtime/container.py:18` để `None` |
| Không có OCR/ASR | grep toàn repo: không có |
| Không có bộ đo retrieval | grep toàn repo: không có Recall@K/MRR |

Ký hiệu: mỗi item có **Acceptance** là điều kiện coi như xong.

---

## Dataset thật (khảo sát 2026-08-14)

Đã kiểm tra trực tiếp artifact của BTC. Những con số này thay đổi nhiều item bên dưới.

| Chỉ số | Giá trị |
| --- | ---: |
| Video | **873** |
| Keyframe | **177,321** |
| Raw data | 106.73 GiB (video 77.3 + keyframe 28.7 + meta 0.8) |
| Vector ước tính | 177,321 × 1152 × 4B ≈ **817 MB** |
| Snapshot ước tính | **1.5–2 GB** (gồm HNSW) |

### BTC cho sẵn ba thứ, dùng được ngay

**`map-keyframes/*.csv`** — 873 file, cột `n,pts_time,fps,frame_idx`. Khớp
**chính xác** `KeyframeManifestRow`:

| CSV của BTC | Field hệ thống |
| --- | --- |
| `frame_idx` | `original_frame_id` |
| `pts_time` | `pts_sec` |
| `fps` | verify VFR |

→ **Bỏ được hoàn toàn bước `sampling`**, tức không phải decode 77 GB video.
Dùng `batch_builder keyframes` nạp thẳng từ CSV.

**⚠ Bẫy chết người: tên file keyframe KHÔNG phải `original_frame_id`.**

Zip giải nén ra `keyframes/L23_V001/020.jpg`. Số `020` là cột **`n`** — "keyframe
thứ 20 của video này" — **không phải** `frame_idx`. Trong ví dụ trên, `n=2` ứng
với `frame_idx=135`.

Mà `docs/ingestion.md` mô tả `batch_builder.scan_keyframes` quét theo quy ước
`{video_id}_{frame_id}.jpg` — quy ước đó **không khớp** cấu trúc BTC. Nạp thẳng
theo tên file là ghi `n` vào `original_frame_id`, và **mọi bài nộp sẽ sai** mà
không có gì báo lỗi: id vẫn là số nguyên hợp lệ, vẫn qua validate, chỉ là trỏ
sai frame.

Bắt buộc join qua `map-keyframes/*.csv` để lấy `frame_idx`. Cần sửa hoặc thay
`scan_keyframes` cho khớp layout thật (`keyframes/<video_id>/<n>.jpg`).

File trong `objects/` cũng đánh số theo `n`, nên join cùng một cách.

**`objects/<video>/<frame>.json`** — 177,321 file, khớp **1:1** với keyframe.
Format OpenImages: `detection_class_entities` (nhãn tiếng Anh),
`detection_scores`, `detection_boxes`. 100 detection/frame nhưng phần lớn score
< 0.1 — cần threshold (≥0.5 là hợp lý).

**`media-info/*.json`** — 873 file, có `watch_url` (YouTube), `length` (giây),
`title`, `description`, `keywords`, `publish_date`, `author`.

### Thứ BTC **không** cho

**`shot_id`.** `map-keyframes` không có cột shot, mà `dedupe_by_shot` và
`fusion` đều lấy `(video_id, shot_id)` làm khoá gộp.

**Đã tìm được nguồn công khai thay cho việc tự chạy TransNetV2** (khảo sát
2026-08-14). Dataset HF `tanp21/aic-hcmc-2025-videos`, thư mục
`annotations/shot_json/`, có **đúng 873 file** phủ khít L21–L30. Định dạng là
list `[start_frame, end_frame]` — đầu ra TransNetV2, tổng 97.811 shot,
trung vị 101 shot/video. Đã tải về `data/shots_hf/` (7.7 MB).

**Đã kiểm chứng, kết quả đạt** — `scripts/verify_shots.py`, đối chiếu với
`map-keyframes` + `media-info` nên không cần tải 77 GiB video:

| Phép kiểm | Kết quả | Ý nghĩa |
| --- | --- | --- |
| Khớp tên video | 873/873, không thiếu, không thừa | phủ khít |
| Drift thời lượng (`last_shot_end/fps` vs `media-info.length`) | trung vị 0.24s, **max 0.57s** | BTC **không** encode lại; frame numbering trùng khớp |
| Keyframe nằm trong shot | 177.169/177.321 = **99.914%** | ranh giới shot đúng |

152 keyframe lọt ngoài, đã truy nguyên hết, **không phải lỗi dữ liệu**:

- **132 rơi vào khe 1–2 frame giữa hai shot liền kề** — TransNetV2 để trống
  đúng frame chuyển cảnh. Gán vào shot kế tiếp là xong.
- **20 là `frame_idx=0`** ở những video có `shot[0]` bắt đầu từ frame 1. Kẹp
  về shot đầu tiên.

→ **Bỏ hẳn bước shot detection.** Đây là bước GPU đắt nhất còn lại, và cũng
chính là bước đã chết vì deadlock PyAV ở job clip trước đó.

Cần tự soát quy chế BTC xem có cấm dùng annotation của đội khác không.

### ⚠ `frame_idx` KHÔNG duy nhất trong một video

Phát hiện khi kiểm chứng shot, độc lập với dữ liệu shot. Ngay trong
`map-keyframes` của BTC:

```csv
n,pts_time,fps,frame_idx
1,0.0,30.0,0        ← hai keyframe khác nhau...
2,0.0333333,30.0,0  ← ...cùng frame_idx
3,4.03333,30.0,120
```

Quy mô: **192/873 video**, **614 keyframe trùng** (0.346%).

Hệ quả: `(video_id, frame_idx)` **không dùng làm khoá định danh được**. Point
id sinh xác định từ cặp này sẽ va nhau và **âm thầm nuốt mất 614 keyframe**
lúc upsert vào Qdrant — không có lỗi nào báo ra.

Khoá đúng là `(video_id, n)`. `frame_idx` chỉ là **giá trị đem đi nộp**, không
phải danh tính. Xem thêm bẫy `n` vs `frame_idx` ở trên — hai bẫy này ngược
chiều nhau và dính cả hai thì vừa sai bài nộp vừa mất dữ liệu.

**Đã sửa trong code** (2026-08-14). `KeyframeManifestRow` có thêm trường
`keyframe_n`, `point_parts()` đổi sang `(video_id, f"kf{n}")`, cột thêm vào
`KEYFRAME_ARROW_SCHEMA` và payload. Kèm test hồi quy
`test_keyframes_sharing_a_frame_id_stay_distinct_points`. Đo trên dữ liệu thật:
scheme cũ sinh 176.707 point id cho 177.321 hàng (**mất 614**), scheme mới sinh
đủ 177.321 (**mất 0**).

⚠ Đây là thay đổi phá vỡ tương thích: manifest cũ thiếu cột `keyframe_n` sẽ
không qua `validate_columns`, và mọi point id đều đổi → phải dựng collection
mới. Đúng hướng, vì ingestion vốn đã build collection có version.

### Manifest đã dựng xong

`scripts/build_frames_manifest.py` join map-keyframes + shot_json + objects +
media-info + transcripts. Không dùng `batch_builder keyframes` được, vì nó đọc
`original_frame_id` từ **tên file** — đúng với keyframe do repo tự sample, sai
với keyframe của BTC.

| Artifact | Nội dung |
| --- | --- |
| `data/frames.parquet` | **177.321 hàng**, 873 video, 15 cột, 4.9 MB |
| `data/clips.parquet` | **97.811 shot**, 2.3 MB |

Cột: 6 cột bắt buộc của pipeline (`video_id`, `shot_id`, `keyframe_n`,
`original_frame_id`, `pts_sec`, `path`) + enrichment (`objects`,
`object_counts`, `asr_text`, `title`, `author`, `channel_id`, `publish_date`,
`keywords`, `watch_url`). Pydantic bỏ qua cột lạ nên pipeline hiện tại đọc
được ngay, không phải sửa gì.

Kiểm tra đã chạy: `validate_columns` đạt, `(video_id, keyframe_n)` duy nhất
177.321/177.321, không có giá trị âm, không có path rỗng, không shot nào
`end < start`. Phủ: **95.0%** frame có object (ngưỡng 0.3 — trung vị 7
object/frame; 5% còn lại là ảnh chữ/cảnh chuyển, file objects vẫn tồn tại
nhưng không detection nào vượt ngưỡng), **9.7%** frame có ASR (do mới cào được
64/873 video).

10.169/97.811 shot (10.4%) không chứa keyframe nào — shot quá ngắn để rơi vào
lưới ~1 keyframe/giây của BTC. Không phải lỗi; chỉ là những shot đó chỉ tìm
được qua clip vector, không qua frame vector.

---

## P0 — Blocker: không sửa thì không quy đổi được thành điểm

### P0-1. Implement `SubmissionService`

**Vấn đề.** `POST /submissions/export` có contract trong `docs/openapi.json`
nhưng `deps.get_submission_service` trả `None` → request nổ ở runtime.
Đây là bước duy nhất biến kết quả search thành điểm.

**Việc cần làm**
- Implement class trong `backend/app/services/` (hoặc module mới
  `app/submissions/`), wire vào `runtime/container.py`.
- Sinh CSV đúng format BTC. Giữ nguyên `original_frame_id` từ payload, tuyệt đối
  không tự đánh lại index.
- Xử lý `VideoNotFoundError` / `FrameOutOfBoundsError` đã khai báo sẵn ở
  `services/submissions.py:11-18` thành HTTP error tương ứng ở endpoint.

**Rủi ro.** Format nộp bài của BTC chưa được xác nhận trong repo → xem Open
questions Q1. Không đoán format; hỏi trước khi code.

**Acceptance.** Unit test sinh file từ một `ExportRequest` cố định và so byte
với fixture; endpoint trả `200` + `Content-Disposition` đúng tên file.

---

### P0-2. Chốt hướng đi cho track QA

**Vấn đề.** `retrieval/tracks.py:56` hardcode `answer=None`, có chủ ý (comment
giải thích: không đoán bừa). Nhưng QA cần một chuỗi answer để chấm.

**Hai lựa chọn — phải chọn một, không để lửng**
1. **Operator nhập tay** (khuyến nghị cho vòng đầu): backend chỉ lo retrieval,
   FE cho phép gõ answer trước khi export. Chi phí gần bằng 0.
2. **Wire VQA model**: thêm một VLM local vào query path. Chi phí latency và
   VRAM lớn, cần đo bằng P1-3 trước khi tin.

**Acceptance.** Nếu chọn (1): `SearchResponse` giữ `answer=None` và P0-1 chấp
nhận answer từ request. Nếu chọn (2): có số Recall/accuracy trên tập QA có nhãn.

---

### P0-3. Chạy lại clip ingestion

**Vấn đề.** Toàn bộ `ranking/fusion.py` và `CLIP_FUSION_WEIGHT` đang là code
chết vì clip collection rỗng. Bug PyAV deadlock đã vá ở `af47014`.

**Việc cần làm**
- Chạy lại job cho `clips.parquet` với **tên collection versioned mới**
  (`store.collection_name_exists` sẽ chặn tên cũ — đúng thiết kế).
- Bật `QDRANT_CLIPS_COLLECTION` trong `.env`, restart API.
- Đo lại latency trước/sau: fusion thêm một round trip Qdrant và kéo theo
  clip hit vào rerank (xem P2-2, làm trước sẽ đỡ đau).

**Acceptance.** Job `succeeded` với `progress_completed == progress_total ==
1311`; một query KIS trả về ít nhất một hit có `start_sec` khác `None`.

---

### P0-4. CORS — FE khác origin sẽ bị browser chặn hoàn toàn

**Vấn đề.** `grep -rn "middleware\|CORS" backend/app/` trả về **rỗng**. App
không đăng ký một middleware nào. FE là repository riêng (`../aic2026-fe`),
chạy trong browser, gần như chắc chắn ở origin khác (Vite dev server, hoặc máy
khác trong LAN). Mọi request từ FE sẽ bị preflight chặn.

Đây là loại lỗi không xuất hiện khi test bằng `curl` hay Swagger UI (cùng
origin) nhưng làm chết toàn bộ hệ thống lúc ghép FE.

**Việc cần làm.** Thêm `CORSMiddleware` trong `main.py:create_app`, origin lấy
từ Settings (không hardcode `*` — LAN thi đấu vẫn nên whitelist).

**Acceptance.** Request từ origin khác kèm preflight `OPTIONS` trả `200` với
đúng `Access-Control-Allow-Origin`.

---

## P1 — Nơi thắng thua thật sự (accuracy)

### P1-1. OCR index — lỗ hổng accuracy lớn nhất

**Vấn đề.** Không một dòng code nào trong repo đụng tới chữ trên màn hình.
Query kiểu AIC liên tục nhắc bản tin chạy chữ, biển hiệu, tên riêng. Data là
tin tức HTV — đồng nhất về thị giác đến mức khắc nghiệt: hàng nghìn frame
"người dẫn ngồi bàn trong trường quay" có cosine ~0.95 với nhau. Chữ dưới màn
hình là tín hiệu **duy nhất** phân biệt được chúng.

**Kiến trúc** (bám architecture rule: Qdrant là nguồn filter duy nhất, không
thêm search engine thứ hai)
- CLI mới `app/ingestion/video/ocr.py`, cùng dạng `probe`/`shot_detect`, ghi
  `ocr.parquet` (`video_id`, `original_frame_id`, `text`, `confidence`).
- Nạp vào **cùng point** của frame collection: payload + sparse vector, không
  tạo collection riêng → hybrid search trong một `query_points`.
- Fuse dense + sparse bằng RRF (P1-4).

**Chọn engine — phải benchmark, không chọn theo cảm tính**

177,321 frame khiến throughput quyết định tất cả. Ước lượng thô, **cần đo lại
trên GPU thật của bạn**:

| Engine | Ước tính | Ghi chú |
| --- | --- | --- |
| EasyOCR | ~2–5 giờ | Chạy trên torch sẵn có, không thêm framework |
| PaddleOCR PP-OCRv4 | ~2–4 giờ | Chính xác hơn trên scene text dày, kéo theo `paddlepaddle` |
| Qwen2-VL-2B | ~5–8 giờ | Đọc được layout phức tạp, ra text có ngữ cảnh |
| Qwen2-VL-7B | ~15–25 giờ | Chất lượng cao nhất, chi phí cao nhất |

**Việc đầu tiên phải làm: benchmark 500 frame** trên cả 4, so cả tốc độ lẫn
chất lượng đọc tiếng Việt. Quyết định sau khi có số — chênh lệch giữa 2 giờ và
25 giờ là chênh lệch giữa "chạy lại được vài lần" và "chỉ chạy được một lần".

**Khuyến nghị mặc định:** OCR chuyên dụng (EasyOCR/PaddleOCR) cho toàn bộ
177k frame. Qwen-VL để dành cho P1-6, nơi nó thật sự hơn.

---

#### Kết quả benchmark thật (Kaggle, 2026-08-15)

Notebook `natsukiseba/aic-ocr-benchmark` và `natsukiseba/aic-ocr-quality`.
Ảnh kéo thẳng từ R2 bằng presigned URL, notebook không chứa credential.

**Hạ tầng — một nút thắt phải xử lý bằng tay.** Kaggle cấp **Tesla P100
(sm_60)**, mà torch cài sẵn (2.10+cu128) chỉ build sm_70+. Nguy hiểm ở chỗ
`torch.cuda.is_available()` vẫn trả `True` rồi mọi phép tính CUDA mới nổ
`no kernel image is available` — một job 8 tiếng sẽ chết ở phút đầu. Thử hạ
torch về 2.4.1+cu121 (bản còn build sm_60) **thất bại**: gói phụ thuộc
`nvidia-cudnn-cu12==9.1.0.70` đã bị gỡ khỏi PyPI.

→ Phải vào **web UI chọn GPU T4 x2**; API chỉ có `enable_gpu` bật/tắt, không
chọn được loại. Notebook phải luôn chạy thử một phép tính CUDA thật trước khi
vào vòng lặp, vì `is_available()` nói dối.

| Đo được | |
| --- | --- |
| R2 → Kaggle | 87 MB/s → kéo cả 28.7 GB ảnh ≈ 6 phút/phiên |
| Ảnh keyframe | 1280×720 |
| OCR trên CPU | **0.17 ảnh/s** → 140 h mức shot. Không khả thi, GPU là bắt buộc |
| Ảnh có chữ | **100%**, 9.1 vùng chữ/ảnh |

**Chất lượng phụ thuộc mạnh vào loại nội dung.** EasyOCR, tin cậy trung vị:

| Lô | Trung vị | |
| --- | ---: | --- |
| L21 (tin tức "60 Giây") | **0.72** | đọc ra headline thật |
| L23 (đua xe đạp) | **0.47** | chữ đồ hoạ cách điệu, hỏng nặng |

Trên tin tức, thứ đọc được đúng là cái cần cho retrieval:

```
[0.89] leo thang giữa Israel và Hezbollah
[0.80] Ucraina mất 3 pháo phản lực HIMARS tại Kursk
[0.79] giá dất cũ dược tiếp tuc áp dung đến hết ngày 31/12/2025
[0.75] dưa dón ổ nhóm lừa dảo bằng công nghệ cao nhập cảnh trái phép vào Việt Nam
```

**⚠ Lỗi OCR khác hẳn lỗi ASR, và sửa được.** Lỗi gần như chỉ có một loại:
`đ` đọc thành `d` (`dường`←`đường`, `dối`←`đối`, `dảo`←`đảo`, `dến`←`đến`).
Đây là lỗi **dấu**, nên bỏ dấu là gộp được:

```
dường → duong  ==  đường → duong     ✓ thu hồi được
```

Ngược hẳn với ASR, nơi lỗi nằm ở phụ âm (`sục`≠`sụt`, `lúng`≠`lún`) và bỏ dấu
vô dụng. → **Index thêm một biến thể đã bỏ dấu bên cạnh text gốc** là thu hồi
gần hết lỗi OCR. Rẻ, và chỉ hiệu quả với OCR chứ không với ASR.

**Lọc theo ĐỘ DÀI, không phải theo độ tin cậy.** Đây là phát hiện đáng giá
nhất:

| | |
| --- | --- |
| Vùng ≤4 ký tự | **50% số vùng nhưng chỉ 12% tổng ký tự** — logo, đồng hồ, rác (`IF`, `1S`, `Hd`, `IH`) |
| Vùng ≥25 ký tự | 28/228 vùng, tin cậy trung vị 0.74 — **gần như toàn bộ là headline** |

Ngưỡng tin cậy 0.3 giữ 91% ký tự trong khi bỏ 22% số vùng. Nhưng lọc theo độ
dài sạch hơn nhiều: bỏ vùng ngắn dưới ~5 ký tự trừ khi tin cậy rất cao.

**Ticker chạy chữ bị cắt cụt** (`...dến hết ngà`, `...tại 5 sc`). Vì chữ cuộn
ngang, frame kế tiếp chứa phần còn lại. → Thêm một lý do gộp OCR **ở mức
shot**: nối text các frame trong cùng shot dựng lại được câu đầy đủ.

**Chưa đánh giá được PaddleOCR.** Hai lần thử đều hỏng vì PaddleOCR 3.7.0 đổi
API (`show_log` bị gỡ, `predict()` trả kiểu mới). **Đừng coi EasyOCR là đã
thắng so sánh** — nó mới chỉ là cái duy nhất chạy được.

**Điều chỉnh khuyến nghị.** Trên tin tức, OCR chuyên dụng là đủ; chưa cần
Qwen-VL cho việc đọc chữ. Nhưng với các lô thể thao/chữ đồ hoạ cách điệu
(L23 và tương tự), engine hỏng nặng và **đó mới là chỗ Qwen-VL đáng dùng** —
tức dùng có chọn lọc theo loại nội dung, không phải chạy đại trà 177k frame.

**Acceptance.** Trên tập eval P1-3, Recall@10 của nhóm query có chữ trên màn
hình tăng đo được so với baseline dense-only.

---

### P1-2. ASR — cào transcript YouTube trước, Whisper chỉ vá lỗ

**Phát hiện quan trọng.** `media-info/*.json` có `watch_url` cho đủ 873 video.
Nghĩa là phần lớn transcript **cào được**, không cần chạy Whisper toàn bộ.

| Cách | Chi phí |
| --- | --- |
| Whisper toàn bộ 873 video | ~15–25 giờ GPU |
| Cào transcript YouTube | **vài phút**, 0 GPU |
| Hybrid (cào + Whisper phần thiếu) | vài phút + GPU cho phần sót |

**Không có nguồn transcript công khai** (khảo sát 2026-08-14). Đã tìm GitHub
code search (190 repo dùng đúng cấu trúc `map-keyframes`/`media-info` này),
Hugging Face datasets, Kaggle, và paper của các đội năm trước. Không đội nào
release dữ liệu transcript. Paper MERVIN (AIC HCMC 2025) ghi rõ họ cũng làm
đúng cách này — YouTube Transcript API, Whisper vá lỗ — rồi **dùng Gemini 1.5
Flash làm sạch transcript**, nhưng không công bố output.

**Thực tế chạy 2026-08-14: YouTube chặn IP sau ~64 video.**
`scripts/scrape_transcripts.py` lấy được 64/873 (19.4 giờ lời nói, 29.525
segment, 100% tiếng Việt) rồi dính `IpBlocked`. yt-dlp xác nhận 809 video còn
lại **vẫn có phụ đề** `vi-orig, vi` — thuần tuý là rate-limit theo IP, không
phải thiếu dữ liệu. Script có resume + cooldown, nhưng **không thể coi đây là
đường đi tin cậy được**: thời gian hoàn thành phụ thuộc vào YouTube.

**⚠ Chất lượng transcript cào thấp hơn giả định ban đầu.** Đây là caption tự
động, không phải người viết. Mẫu thật từ `L21_V001`:

| Cào về | Đúng ra |
| --- | --- |
| **Đại** truyền hình TP.HCM | **Đài** Truyền hình TP.HCM |
| **đùng băng sông cử Long** | **Đồng bằng sông Cửu Long** |
| tình trạng **sục lúng** | tình trạng **sụt lún** |
| nước biển **dân** | nước biển **dâng** |
| **vẫn chuyển C tốc** trái tim | **vận chuyển siêu tốc** trái tim |

Trong cùng một video, "sông Cửu Long" hỏng thành **hai kiểu khác nhau**
(`cử Long`, `Cổ Long`) — một thực thể không tự khớp với chính nó.

Và **chuẩn hoá bỏ dấu không cứu được**, vì lỗi nằm ở phụ âm chứ không chỉ
thanh điệu: `sục→suc` ≠ `sụt→sut`, `lúng→lung` ≠ `lún→lun`,
`dân→dan` ≠ `dâng→dang`. Chỉ `Đại/Đài → dai` là gộp được — 1/4 trường hợp.

→ Hệ quả: transcript cào **không đủ tin cậy để làm entity index**. Giám khảo
gõ "đồng bằng sông Cửu Long sụt lún" thì sparse vector khớp bằng 0.

**Rủi ro có thật.** Video publish 08/2024, giờ là 08/2026. Chắc chắn một số đã
bị xoá/private → coverage không thể đạt 100%. Vì vậy **bắt buộc** có bước
Whisper vá lỗ, không dựa hoàn toàn vào cào.

**Kết luận điều chỉnh: đảo thứ tự ưu tiên.** ASR chạy bằng model mới là nguồn
chính, cào chỉ là bản vá. Lý do: (a) chất lượng cao hơn hẳn, (b) không phụ
thuộc YouTube có chặn hay không, (c) hạ tầng ingest đã có sẵn. Tham khảo
`khang1108/MLeCDanBGold` — pipeline ASR hoàn chỉnh dùng **Qwen3-ASR-1.7B** +
Silero VAD + pyannote diarization, có resume/manifest/Parquet; Qwen3-ASR nhiều
khả năng tốt hơn Whisper cho tiếng Việt. Cân nhắc thêm một bước LLM làm sạch
transcript như MERVIN đã làm.

**Bước bắt buộc trước khi tin transcript: verify alignment.**
`media-info.length` (giây) phải khớp duration thật từ `probe`. Lệch nghĩa là
BTC đã cắt/encode lại → timestamp YouTube gắn sai frame → transcript thành
rác có hại. Video nào lệch quá ngưỡng thì loại, đẩy sang Whisper.

**Việc cần làm**
- CLI `app/ingestion/video/asr.py`, hai chế độ: `--source youtube` và
  `--source whisper`, cùng ghi ra `asr.parquet`
  (`video_id`, `start_sec`, `end_sec`, `text`, `source`).
- Nguồn `youtube` đọc từ `data/transcripts/*.json` do
  `scripts/scrape_transcripts.py` sinh ra.
- **⚠ Span thật của một segment là `[start_i, start_{i+1})`, không phải
  `[start, start+duration]`.** Caption YouTube là cửa sổ cuộn: đo trên dữ liệu
  thật, **98% segment liền kề chồng lấn thời gian** (text thì không trùng —
  nối liền vẫn ra câu liền mạch). Dùng `start+duration` thì một frame dính 2–3
  segment, text bị nhân bản, term frequency của sparse vector lệch hẳn.
- Map timestamp → frame range bằng FPS trong `videos.parquet`. Pipeline đã từ
  chối video VFR (`VariableFrameRateError`, `manifest.py:12`) nên phép quy đổi
  an toàn — giữ nguyên bất biến đó.
- Cột `source` để P1-3 đo riêng chất lượng hai nguồn.

**Acceptance.** ≥80% video có transcript; alignment sai lệch < 1 giây trên mẫu
kiểm tra; query mô tả nội dung lời nói trả đúng video trong top-10.

---

### P1-6. Image captioning bằng Qwen-VL

**Yêu cầu từ quản lý.** Sinh caption + mô tả OCR cho keyframe bằng Qwen-VL, nạp
vào database.

**Vì sao có lý.** Đây là chỗ Qwen-VL thật sự hơn OCR chuyên dụng — không phải ở
việc đọc chữ, mà ở việc mô tả **quan hệ** giữa các vật thể. Dual encoder như
SigLIP nổi tiếng không phân biệt được "người đàn ông cầm ô đỏ" với "ô đỏ và
người đàn ông"; caption thì giữ được trật tự đó. Caption tiếng Việt còn khớp
từ vựng thẳng với query tiếng Việt qua sparse vector.

**Nói thẳng về chi phí.** 177,321 frame là con số lớn:

| Model | Ước tính | Khả thi? |
| --- | --- | --- |
| Qwen2-VL-2B | ~5–8 giờ | Chạy được trong một đêm |
| Qwen2-VL-7B | ~15–25 giờ | Cần 1–2 ngày, gần như chỉ chạy được một lần |

Đây là **ước tính**, sai số lớn tuỳ GPU. Phải benchmark 500 frame trước khi
cam kết — xem Open question Q4 (cấu hình máy).

**Cách giảm chi phí mà giữ phần lớn giá trị**
- Caption ở mức **shot** thay vì mức frame. Sau shot detection, 177k frame gom
  lại còn khoảng vài chục nghìn shot → giảm chi phí nhiều lần, mà caption của
  một frame đại diện thường mô tả đúng cả shot.
- Prompt một lần ra **cả hai**: caption + text đọc được trong ảnh. Một lượt
  inference thay vì hai.
- Chạy sau cùng, coi là *nice-to-have*: nếu hết thời gian thì bỏ, hệ thống vẫn
  chạy đủ với OCR + ASR.

**Việc cần làm**
- CLI `app/ingestion/video/caption.py` → `caption.parquet`
  (`video_id`, `original_frame_id`, `caption`, `ocr_text`).
- Nạp vào sparse vector chung với OCR/ASR.

**Acceptance.** Benchmark 500 frame có số thật về giờ GPU trước khi chạy full;
P1-3 cho thấy caption cải thiện Recall@10 trên nhóm query mô tả quan hệ/hành động.

---

### P1-7. Nạp objects và media-info vào payload — gần như miễn phí

**Vì sao rẻ.** BTC đã tính sẵn. Không tốn một giây GPU nào, chỉ là parse JSON.

**Objects** — 177,321 file, khớp 1:1 keyframe. Lọc `score >= 0.5`, lấy tập nhãn
duy nhất, nạp vào payload `objects: ["Food", "Chopsticks", ...]` + payload index
kiểu keyword.

**Nói thẳng về giá trị:** thấp hơn OCR nhiều. Nhãn là tiếng Anh, vocab
OpenImages, mà SigLIP vốn đã mã hoá vật thể trong ảnh rồi — "Food" không thêm
thông tin nào vector chưa có. Giá trị thật nằm ở **filter** (P1-8), không phải
ở ranking.

**Media-info** — nạp cấp video: `title`, `description`, `keywords`,
`publish_date`, `author`, `channel_id`. Đây là text tìm kiếm được miễn phí:
query kiểu "bản tin ngày 01/08/2024" khớp thẳng `publish_date`.

**Acceptance.** Payload index tạo xong; filter theo `objects` và
`publish_date` chạy được từ API.

---

### P1-8. Bộ công cụ khoanh vùng nhanh khi đang thi

**Bối cảnh.** Trong vòng thi tương tác, operator thường nhận ra đúng video (hoặc
thu hẹp còn vài video) trước khi tìm ra đúng frame. Lúc đó thứ cần không phải
là model tốt hơn — mà là công cụ nhảy nhanh **trong phạm vi đã khoanh**.

**Điểm mấu chốt: phần lớn đã có sẵn trong code, chỉ chưa được expose ra API.**

`vector_store/search.py` đã có `build_filter(video_ids=..., shot_ids=...)` và
`engine.retrieve()` đã nhận tham số `video_ids`. Nhưng `KisSearchRequest` /
`QaSearchRequest` không có field nào để truyền vào. Nghĩa là tính năng đắt giá
nhất ở đây chỉ cách một field trong schema.

**Danh sách theo tỷ lệ lợi ích / công sức**

| # | Tính năng | Công sức | Ghi chú |
| --- | --- | --- | --- |
| 1 | **Filter theo `video_ids`** trong request KIS/QA | Rất nhỏ | `build_filter` đã hỗ trợ; chỉ thêm field vào schema + truyền xuống |
| 2 | **Duyệt frame theo video/shot** (timeline) | Nhỏ | Endpoint scroll payload theo `video_id`, sort `original_frame_id` |
| 3 | **Frame lân cận** — cho frame X, lấy ±N frame | Nhỏ | Filter `original_frame_id` trong khoảng |
| 4 | **Tìm giống frame này** (P1-5) | Nhỏ | Query bằng vector của point, không cần encode |
| 5 | **Filter `publish_date` / `author`** | Nhỏ | Từ media-info (P1-7) |
| 6 | **Filter theo `objects`** | Nhỏ | Từ objects (P1-7) |
| 7 | **Tìm chữ trong phạm vi một video** | Vừa | Sparse search + filter `video_ids` (cần P1-1) |

**Vì sao đáng làm sớm.** Nhóm này rẻ hơn nhiều so với OCR/caption nhưng ăn điểm
trực tiếp trong vòng thi tương tác — nơi thời gian mỗi câu bị tính. Một operator
khoanh đúng video rồi mà vẫn phải gõ lại mô tả từ đầu là lãng phí lớn nhất
trong toàn bộ workflow.

**Cảnh báo contract.** Nhóm này đổi `schemas/search.py` → **bắt buộc**
regenerate `docs/openapi.json` và báo team FE. Làm P3-8 (test contract drift)
trước sẽ chặn được lỗi âm thầm.

**Acceptance.** Operator khoanh 1 video rồi search lại trong đó cho kết quả
< 200 ms; `docs/openapi.json` được regenerate; có test cho từng filter mới.

---

### P1-3. Bộ đánh giá offline — làm sớm, mọi tuning phụ thuộc vào đây

**Vấn đề.** `CLIP_FUSION_WEIGHT=0.5`, `RERANK_TOP_N=30`, `DEFAULT_OVERFETCH=5`,
chọn `so400m` hay `giant` — **tất cả đang là số đoán**. Không đo được thì không
tune được, và tệ hơn: không biết một thay đổi làm tốt lên hay xấu đi.

**Việc cần làm**
- Tập query có nhãn (dùng lại đề các mùa trước) dưới dạng
  `qrels.parquet`/`json`: `query_id`, `query_text`, `task`, relevant
  `(video_id, original_frame_id)`.
- Script `scripts/evaluate.py`: chạy qua đúng `engine.retrieve()` của production
  (không viết lại retrieval — vi phạm rule "do not duplicate"), xuất
  Recall@1/5/10, MRR, nDCG, kèm breakdown latency theo stage (`Timings` đã trả
  sẵn dữ liệu này).
- Chạy được ở chế độ "ablation": bật/tắt rerank, đổi `clip_weight`, đổi profile.

**Acceptance.** Một lệnh cho ra bảng số so sánh ≥2 cấu hình trên cùng tập query.

---

### P1-4. Chuyển fusion sang RRF của Qdrant Query API

**Vấn đề.** `ranking/fusion.py:41-46` cộng điểm có trọng số và impute điểm sàn
cho shot chỉ xuất hiện ở một list. Chính comment trong file đã ghi chú sẵn RRF
là phương án dự phòng khi hai thang điểm lệch nhau. Khi thêm sparse vector
(P1-1) thì cộng điểm trực tiếp **chắc chắn sai** — BM25 và cosine không cùng
thang.

**Việc cần làm**
- Dùng `prefetch` + `FusionQuery(fusion=Fusion.RRF)` của Qdrant Query API để
  gộp frame/clip/sparse trong **một** round trip (thay luôn P2-4).
- Giữ `fusion.py` cho đường weighted-sum để P1-3 A/B được hai cách.

**Cần verify trước khi code.** Xác nhận `qdrant-client==1.12.1` (đã pin) expose
đúng `prefetch`/`FusionQuery`; nếu chưa, cân nhắc nâng client mà **không** nâng
server image (compose pin `v1.12.1` vì lý do snapshot compatibility).

**Acceptance.** P1-3 cho thấy RRF ≥ weighted-sum, và số round trip Qdrant mỗi
query giảm từ 2+ xuống 1.

---

### P1-5. Relevance feedback ("tìm giống frame này")

**Vấn đề.** Operator tìm được frame gần đúng nhưng chưa đúng thì hiện chỉ có
cách gõ lại chữ. Trong vòng thi tương tác đây là lãng phí thời gian lớn nhất.

**Việc cần làm.** Endpoint mới nhận `(video_id, original_frame_id)`, lấy vector
của point đó trong Qdrant và query bằng chính nó (Qdrant recommend API / query
by id). Không cần encode lại, không cần đọc ảnh → cực nhanh.

**Acceptance.** Endpoint trả kết quả < 100ms (không qua text encoder), có test
contract, `docs/openapi.json` được regenerate.

---

## P2 — Tốc độ

### P2-1. Ingest đang chạy batch size 1 trên GPU

**Vấn đề — nghiêm trọng nhất về hiệu năng.** `ingestion/pipeline.py:47` lặp
**từng row**, gọi `embedder.embed_row` cho mỗi row; với keyframe thì
`embedder.py:15` truyền đúng `[một ảnh]`. Vòng batch trong `multimodal.py:40`
chạy 1 vòng với 1 ảnh. `image_batch_size=4` trong profile là **config chết đối
với frames**. Decode ảnh cũng tuần tự, GPU đứng chờ đĩa.

**Giờ đã có con số thật: 177,321 keyframe.** Với batch size 1 ước tính
**~5 giờ GPU**; sau khi gộp batch 64 ước tính **~30–45 phút**. Bốn tiếng chênh
lệch đó lặp lại **mỗi lần** phải ingest lại (đổi profile, sửa OCR, thêm
caption). Đây là lý do P2-1 phải xong **trước** lần ingest thật đầu tiên.

**Việc cần làm**
- Gộp 64–256 row mỗi forward pass. Sửa `pipeline.upsert_points` để gom row
  trước khi gọi feature layer, thêm hàm batch ở `embedder.py`.
- Decode ảnh song song (thread pool hoặc `DataLoader` với `num_workers`) chồng
  lấn với compute GPU.
- Giữ nguyên `deterministic_point_id` và thứ tự payload → point id không đổi,
  collection cũ vẫn so sánh được.

**Acceptance.** Đo throughput (frames/giây) trước và sau trên cùng một slice;
vector sinh ra khớp bit-wise với đường cũ trên một mẫu nhỏ.

---

### P2-2. Rerank: bỏ decode video, song song hoá decode ảnh

**Vấn đề.** `ranking/rerank.py:78` (`_hit_images`) decode ảnh từ đĩa cho **từng**
candidate. Với clip hit, nó gọi `media.sample_clip_frames` → seek + decode 3
frame từ video gốc. Bật fusion (P0-3) lên thì một query top-30 có thể kéo theo
hàng chục lần seek video. Đây là thứ nuốt latency, không phải bản thân model.

**Việc cần làm, theo thứ tự lợi ích/chi phí**
1. Clip hit **không** decode lại video: map về keyframe JPEG đại diện đã có sẵn
   trên đĩa từ bước sampling. Cùng thông tin, rẻ hơn nhiều bậc.
2. Decode ảnh song song bằng thread pool (PyAV nhả GIL).
3. LRU cache pixel đã decode — operator refine query liên tục trên cùng tập
   frame nóng.
4. Nâng `BATCH_SIZE` (`rerank.py:35`, đang là 8) cho đủ lấp GPU.

**Acceptance.** `latency_ms["rerank"]` giảm đo được trên cùng query; P1-3 xác
nhận Recall không đổi.

---

### P2-3. TRAKE chạy tuần tự

**Vấn đề.** `retrieval/tracks.py:74` gọi `retrieve()` cho từng event. 5 event =
5 lần encode + 10 lần query Qdrant + 5 lượt rerank.

**Việc cần làm.** Encode cả N event trong một forward pass (text encoder nhận
batch), dùng `query_batch_points` cho N query, rerank một lượt trên hợp các
list. Phần DP `_best_increasing_sequence` giữ nguyên — nó đã đúng.

**Acceptance.** Latency TRAKE 5 event không còn tuyến tính theo số event.

---

### P2-4. Gộp round trip Qdrant

`engine.py:63` rồi `:71` là hai round trip tuần tự cho frame và clip. Hợp nhất
vào P1-4 (RRF prefetch) nếu làm P1-4; nếu chưa, dùng `query_batch_points`.

---

### P2-5. Warmup model lúc khởi động

**Vấn đề.** Cả hai runtime load lazy qua `lru_cache` (`multimodal.py:91`,
`rerank.py:129`). Query đầu tiên sau restart phải chờ load SigLIP so400m **và**
BLIP large. Trong phòng thi đó là một lượt mất trắng.

**Việc cần làm.** Warm cả hai trong `core/lifespan.py` bằng một query giả.
Kiểm tra VRAM headroom vì hai model cùng resident.

**Acceptance.** Query thật đầu tiên sau restart có `latency_ms` tương đương
query thứ hai.

---

### P2-6. Cache query embedding

Operator gõ lại và sửa nhẹ query liên tục. LRU trên `(profile, text)` quanh
`engine.encode_query` là lợi ích gần như miễn phí, không thêm dependency.

---

### P2-7. Tune HNSW và bật quantization

`vector_store/collections.py:29` đang dùng toàn bộ mặc định: không scalar
quantization, không tune `hnsw_ef` lúc search. Ở quy mô thật (hàng triệu vector
1152 chiều) int8 quantization cắt ~4× bộ nhớ và tăng tốc rõ rệt.

**Chỉ làm sau P1-3** — quantization đánh đổi recall, phải đo.

---

## P3 — Chống sập trong lúc thi

### P3-1. `/health/ready` phải kiểm tra thật

Runbook tự thừa nhận `ready` không warm model, không query Qdrant, không kiểm
tra active collection. Đổi nhầm một dòng `.env` lúc 2 giờ sáng → API vẫn báo
ready, search vẫn trả kết quả, nhưng là rác.

Thêm: collection tồn tại, `vector_size == embedding_dimension(FEATURE_PROFILE)`,
point count > 0.

### P3-2. Log query

`Timings` đã trả `latency_ms` trong mỗi response nhưng không lưu lại. Persist
vào SQLite (đã có sẵn pattern ở `ingestion/store.py`, không cần dep mới) để sau
mỗi vòng thi phân tích được query nào trượt và trượt ở stage nào.

### P3-3. Timeout và giới hạn đồng thời

Một query TRAKE chậm hiện chiếm threadpool và làm nghẽn mọi request khác. Thêm
timeout mỗi request và semaphore giới hạn số search chạy song song.

### P3-4. Dọn config bẫy

- `core/config.py:21` `QDRANT_COLLECTION_NAME` không ai dùng — xoá.
- `ingestion.db` đang bị commit ở root repo; `.gitignore` chỉ chặn `data/*`.
- `docs/architecture.md` rỗng (0 byte).
- `stubs/search.py` chỉ còn dùng cho dev — xác nhận còn cần không.

---

### P3-5. Job chết là kẹt vĩnh viễn, và làm hỏng luôn tên collection

**Vấn đề — cái bẫy vận hành tệ nhất hiện tại.** Schema `ingestion_jobs`
(`ingestion/store.py:6`) không có PID, không có heartbeat, không có
`updated_at`. `runner.py:50` chỉ ghi `status='failed'` khi bắt được exception.
Runner bị `SIGKILL`, OOM, hay mất điện → row kẹt ở `status='running'` mãi mãi.

Chuyện này **đã xảy ra rồi**: `ing-be427841f5` có error
`"...killed manually"` — dòng chữ đó do người gõ tay vào DB, không phải code sinh ra.

Hậu quả nặng hơn nằm ở `store.py:53`:

```sql
SELECT 1 FROM ingestion_jobs WHERE collection_name = ? AND status != 'failed'
```

Job kẹt ở `running` khiến **tên collection đó bị khoá vĩnh viễn**. Muốn ingest
lại đúng tên cũ thì phải `UPDATE` tay vào SQLite. Gặp cảnh này lúc 2 giờ sáng
trước ngày thi là mất giờ vô ích.

**Việc cần làm** (không cần thêm dependency nào)
- Thêm cột `pid`, `heartbeat_at` vào schema; runner cập nhật heartbeat theo
  từng batch (đã có sẵn callback `on_progress`).
- Khi `list_jobs`/`get_job_status` đọc lên, job `running` mà heartbeat quá hạn
  và PID không còn sống → đánh dấu `failed` kèm lý do rõ ràng.
- Cho phép ingest lại một tên collection thuộc job đã chết.

**Acceptance.** `kill -9` runner giữa chừng → job tự chuyển `failed` trong vòng
một khoảng heartbeat, và tạo lại job cùng `collection_name` được chấp nhận.

---

### P3-6. SQLite chưa bật WAL, sẽ gặp `database is locked`

**Vấn đề.** `store._connect` (`ingestion/store.py:22`) mở connection với toàn bộ
mặc định: journal mode `DELETE`, `busy_timeout = 0`. Trong khi đó runner ghi
progress liên tục còn API đọc song song mỗi lần FE poll. Writer khoá toàn bộ DB
→ reader nhận `sqlite3.OperationalError: database is locked` ngay lập tức thay
vì chờ.

Ingest càng nhanh (sau P2-1) thì tần suất ghi càng cao, xác suất đụng càng lớn.

**Việc cần làm.** `PRAGMA journal_mode=WAL` và `PRAGMA busy_timeout=5000` trong
`_connect`. Hai dòng, không thêm dependency, cho phép đọc song song với ghi.

**Acceptance.** Test chạy một writer liên tục và một reader song song không nhận
lỗi lock.

---

### P3-7. Không có logging — chỉ có `print` trong CLI

**Vấn đề.** Toàn repo không có một `getLogger` nào. Chỉ có `print()` trong các
CLI preprocessing và `scripts/qdrant_snapshot.py`. Process API **không log gì
cả**: không request log, không error log, không stack trace. Lúc có sự cố trong
phòng thi, thứ duy nhất bạn có là access log mặc định của uvicorn.

Đây cũng là tiền đề của P3-2 (log query): không có nền logging thì không có chỗ
gắn vào.

**Việc cần làm.** Cấu hình logging tập trung trong `core/`, gắn vào lifespan.
Log ra file có rotate để còn đọc lại sau vòng thi. Runner chạy detached
(`subprocess.Popen`) nên stdout hiện đang đi vào hư vô — phải redirect vào file
log riêng theo `job_id`.

**Acceptance.** Một exception trong query path để lại stack trace trên đĩa; log
của runner truy được theo `job_id`.

---

### P3-8. Không có test nào cho API contract

**Vấn đề.** `grep -rln "TestClient" backend/tests/` trả về rỗng. Test hiện phủ
ranking, manifest, video, vector_store — nhưng **không có test nào gọi vào
endpoint**. Trong khi `docs/openapi.json` chính là thứ FE codegen ra client.

Nghĩa là: sửa một field trong `schemas/`, test vẫn xanh, OpenAPI đổi âm thầm,
FE vỡ lúc build. Đúng thứ review rule "do not change API contracts silently"
đang cố ngăn, nhưng không có gì thực thi.

**Việc cần làm**
- Test bằng `fastapi.testclient.TestClient` với container override bằng
  `StubSearchService` (`stubs/search.py` đang có sẵn cho việc này) — không cần
  Qdrant thật.
- Một test so `create_app().openapi()` với `docs/openapi.json` đã commit và fail
  nếu lệch. Đây là cách biến review rule thành CI gate.

**Acceptance.** Đổi một field trong `schemas/search.py` mà quên regenerate
OpenAPI → test đỏ.

---

## Lược đồ dữ liệu đích

Phần này là **spec** cho mọi bước sinh dữ liệu (P1-1, P1-2, P1-6, P1-7). Giả
định compute không phải ràng buộc — câu hỏi không còn là *chạy được bao nhiêu*
mà là *sinh ra hình dạng nào để truy vấn ăn được*.

### Nguyên tắc chi phối

**1. Caption và OCR phải ra tiếng Việt.** Quyết định quan trọng nhất ở đây.
Dense retrieval thì tiếng gì cũng được (SigLIP2 đa ngữ), nhưng **sparse/BM25 —
lý do tồn tại của OCR — đòi ngôn ngữ khớp query**. Caption tiếng Anh + query
tiếng Việt = không khớp một token nào. Sinh cả `caption_vi` (chính) và
`caption_en` (dự phòng cho query tiếng Anh, tên riêng nước ngoài).

**2. OCR phải giữ vùng, không chỉ giữ chữ.** Tin tức truyền hình có các vùng
chữ mang ý nghĩa khác hẳn nhau:

| Vùng | Nội dung | Query khớp |
| --- | --- | --- |
| Lower-third | Tên người, chức danh | "ông A phát biểu" |
| Ticker đáy | Tin chạy | "bản tin về giá xăng" |
| Trong cảnh | Biển hiệu, băng rôn | "biển hiệu ghi ..." |
| Góc trên | Logo kênh, đồng hồ | **nhiễu — cắt bỏ** |

Gộp hết vào một trường là vứt thông tin. Cả PaddleOCR lẫn Qwen-VL đều trả
bounding box → phân vùng theo toạ độ y.

**3. Chuẩn hoá Unicode — thứ giết BM25 âm thầm.** Tiếng Việt có hai dạng
Unicode cho cùng một chữ: `ệ` là **1 codepoint** (NFC) hoặc **2** (NFD). OCR
thường ra NFD, người gõ query ra NFC. Nhìn giống hệt nhau, BM25 coi là hai
token khác nhau.

```python
import unicodedata, re
def norm_vi(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def strip_diacritics(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return d.replace("đ", "d").replace("Đ", "D")
```

Áp `norm_vi` cho **mọi** text, cả lúc index lẫn lúc query. Lưu thêm bản không
dấu — operator gõ vội dưới áp lực thời gian ("gia xang" phải khớp "giá xăng").

**4. Tách trường thực thể.** Tên riêng là token hiếm, đúng chỗ BM25 mạnh nhất,
và gần như là khoá duy nhất định vị một khoảnh khắc. Trộn vào caption là pha
loãng.

**5. Đừng nhét mọi text vào một trường sparse.** Caption dài sẽ nhấn chìm OCR
ngắn theo cơ chế chuẩn hoá độ dài của BM25. Để riêng từng trường, gán trọng số
khác nhau lúc fuse.

### Lược đồ

```
FRAME  (177,321 điểm — frame collection)
  video_id, shot_id, original_frame_id, pts_sec, path
  ocr_lower_third, ocr_ticker, ocr_scene         + bản NFC và bản không dấu
  caption_vi, caption_en
  objects[]            nhãn score >= 0.5
  object_counts{}      vd {"Person": 3} — phục vụ query đếm
  entities_person[], entities_place[], entities_org[]
  sparse_vector        sinh từ các trường text trên

SHOT  (clip collection)
  video_id, shot_id, start_frame, end_frame, start_sec, end_sec
  shot_caption_vi      HÀNH ĐỘNG diễn ra — phục vụ TRAKE
  camera_motion        tĩnh | lia | zoom | theo chân
  representative_frame_id

VIDEO  (873 — payload cấp video, nhân bản xuống frame hoặc để riêng)
  title, description, keywords, publish_date, author, channel_id
  asr_segments[]       (start_sec, end_sec, text, source)
  duration, fps
```

### Prompt Qwen-VL

Đặt `temperature=0` cho cả hai — cần tái lập được kết quả khi chạy lại.

**Frame caption + OCR theo vùng** (một lượt inference ra cả hai):

```text
Đây là một khung hình từ bản tin truyền hình Việt Nam.
Mô tả bằng tiếng Việt. Chỉ ghi những gì NHÌN THẤY RÕ, không suy đoán.
Trả về đúng JSON, không thêm chữ nào ngoài JSON:

{
  "caption": "1-2 câu: ai, đang làm gì, ở đâu, bối cảnh thế nào",
  "objects": ["vật thể nổi bật trong khung"],
  "text_on_screen": {
    "lower_third": "chữ ở dải dưới màn hình (thường là tên người và chức danh)",
    "ticker": "dòng chữ chạy ở đáy màn hình",
    "scene": "chữ xuất hiện trong cảnh quay: biển hiệu, băng rôn, bảng biểu"
  },
  "entities": {
    "person": ["tên người"],
    "place": ["địa danh"],
    "org": ["tổ chức, cơ quan"]
  }
}

Trường nào không có thì để chuỗi rỗng hoặc mảng rỗng.
Chép chữ đúng nguyên văn kể cả dấu tiếng Việt. Bỏ qua logo kênh và đồng hồ.
```

**Shot caption** (đưa 4–8 frame trải đều trong shot):

```text
Đây là các khung hình lấy tuần tự từ MỘT cảnh quay liên tục.
Mô tả HÀNH ĐỘNG diễn ra xuyên suốt cảnh, không mô tả từng khung riêng lẻ.
Viết tiếng Việt. Trả về đúng JSON:

{
  "action": "1-2 câu: chuyện gì đang diễn ra, ai làm gì, chuyển động ra sao",
  "camera": "một trong: tĩnh | lia ngang | phóng to | thu nhỏ | theo chân",
  "change": "điều gì thay đổi từ khung đầu đến khung cuối"
}
```

Shot caption là thứ **duy nhất** mô tả được chuyển động — một keyframe đơn lẻ
không bao giờ diễn tả nổi "người đàn ông ngã xuống", mà TRAKE thì hỏi đúng loại
đó. Đây là đầu tư trực tiếp cho track đang yếu nhất.

### Món lợi kèm theo: caption tự sinh bộ eval

P1-3 đang bị chặn vì không có tập query có nhãn — mà nó lại chặn gần hết mọi
thứ còn lại. Có caption rồi thì tháo được nút này:

Cho model đọc `caption_vi` và sinh ngược ra câu query mô tả frame đó. Cặp
(query sinh ra → frame gốc) chính là ground truth. Không bằng đề thi thật,
nhưng **hơn hẳn con số không** — đủ để so RRF với weighted-sum, chỉnh
`clip_weight`, chốt so400m hay giant.

Đây có lẽ là lý do đáng giá nhất để chạy caption sớm: nó tháo nút thắt của cả
kế hoạch.

---

## Lược đồ point trong Qdrant

Phần trên nói về *artifact sinh ra*. Phần này nói về *hình dạng cuối cùng trong
vector DB* — mỗi point mang vector gì, payload gì, trường nào cần index.

Giữ nguyên hai collection hiện có (frames + clips). Metadata cấp video
**denormalize xuống từng frame**: 177k bản sao của `title` nghe lãng phí nhưng
chỉ tốn ~90 MB, đổi lại lọc được trong **một** query thay vì phải join — mà
query path hiện không hỗ trợ join.

```
FRAME POINT  (177,321 điểm)

vectors:
  dense           1152d SigLIP2       tương đồng thị giác
  sparse_text     OCR                 chữ trên màn hình
  sparse_caption  caption_vi + shot_caption
  sparse_speech   ASR + title/keywords

payload CÓ INDEX  (dùng để lọc — P1-8):
  video_id             keyword     đã có
  shot_id              integer     đã có
  original_frame_id    integer     đã có
  objects[]            keyword     lọc "có xe cứu hoả"
  entities_person[]    keyword
  entities_place[]     keyword
  publish_date         datetime    "bản tin ngày 01/08"
  channel_id           keyword

payload KHÔNG index  (chỉ trả về hiển thị):
  path, pts_sec
  ocr_lower_third, ocr_ticker, ocr_scene
  caption_vi, caption_en
  asr_text             đoạn ASR phủ đúng pts_sec của frame này
  title
  object_counts{}
```

### Vì sao tách 3 sparse vector thay vì gộp một

Quyết định kỹ thuật quan trọng nhất ở tầng vector DB.

BM25 chuẩn hoá theo độ dài tài liệu. Nhét OCR (~20 từ) chung với caption
(~50 từ) và ASR (~300 từ) vào một trường thì **OCR bị nhấn chìm** — mà OCR lại
là tín hiệu chính xác nhất.

Tách ra thì mỗi nguồn được chấm điểm độc lập, RRF gộp theo **hạng** chứ không
theo điểm. Frame khớp OCR mạnh vẫn lên đầu dù ASR không khớp gì.

Ba nhóm này còn hỏng theo ba kiểu khác nhau — OCR sai khi chữ mờ, caption sai
khi model ảo giác, ASR sai khi thiếu transcript. Tách ra thì một cái hỏng không
kéo hai cái kia xuống.

### Chỉ index thứ thật sự dùng để lọc

Payload index tốn RAM. `caption` và `ocr_*` **không cần index** — chúng đã được
chấm điểm qua sparse vector, index thêm chỉ tốn bộ nhớ.

Ngược lại `objects`, `entities_*`, `publish_date` **bắt buộc** có index vì đó
là thứ operator dùng để khoanh vùng lúc thi (P1-8). Filter không index thì
Qdrant phải quét toàn bộ collection.

**Chỗ sửa trong code:** `vector_store/payload_indexes.py:11` — `PAYLOAD_INDEXES`
hiện chỉ khai báo 3 field cho `FRAMES` (`video_id`, `shot_id`,
`original_frame_id`). Thêm field mới vào đúng dict đó; runner sẽ tự tạo index ở
stage `creating_payload_indexes`, không cần đụng pipeline.

**Chỗ sửa cho sparse vector:** `vector_store/collections.py:28` (`create_collection`)
hiện chỉ tạo một `VectorParams` không tên. Chuyển sang named vectors +
`sparse_vectors_config` là thay đổi **phá vỡ tương thích** với collection cũ —
bắt buộc tạo collection versioned mới, đúng architecture rule.

### Ba thứ tuyệt đối đừng nhét vào

**1. `detection_boxes`.** 100 box × 4 số mỗi frame. Vô dụng cho retrieval, chỉ
phình payload. Giữ đúng nhãn có `score >= 0.5`.

**2. `description` nguyên văn từ YouTube.** Mỗi video HTV có cùng một khối
boilerplate lặp lại trên **cả 873 video**:

```
► Đăng ký KÊNH để xem Tin Tức Mới Nhất: https://bit.ly/2HoUna4
✅ Web / Wap mobile: https://hplus.com.vn
#TintucthoisuVietnam #HTVTintuc ...
```

Nhét vào sparse vector là bơm một đống token vô nghĩa. Cắt URL, hashtag,
emoji-bullet; chỉ giữ dòng đầu mô tả nội dung thật.

**3. Text chưa chuẩn hoá NFC.** Chuẩn hoá **trước khi ghi vào Qdrant**, không
phải lúc query — nếu không, dữ liệu đã sai từ gốc và không sửa được ở tầng trên.

### Dung lượng ước tính

| Thành phần | Ước tính |
| --- | ---: |
| Dense vector | 817 MB |
| HNSW index | ~1.2 GB |
| 3 sparse vector | ~50 MB |
| Payload | ~210 MB |
| **Tổng** | **~2.3 GB** |

Nhỏ — giữ toàn bộ trong RAM được, chưa cần quantization. Bật int8 sẽ đưa dense
xuống ~204 MB, nhưng chỉ làm sau khi P1-3 đo được nó không hụt recall (P2-7).

### Vì sao hình dạng này thắng

Nó phục vụ **hai pha khác nhau** của một lượt thi:

**Pha 1 — tìm rộng.** Operator gõ mô tả tiếng Việt. `dense` bắt cảnh,
`sparse_text` bắt chữ trên màn hình, `sparse_caption` bắt quan hệ vật thể,
`sparse_speech` bắt nội dung nói. RRF gộp bốn nguồn trong một request.

**Pha 2 — thu hẹp.** Operator nhận ra đúng video hoặc đúng nhân vật. Lúc này
thứ cần không phải model tốt hơn mà là **filter**: `video_id`,
`entities_person`, `publish_date`, `objects`.

Đây là chỗ hầu hết các đội đầu tư thiếu — dồn hết vào pha 1, rồi pha 2 phải gõ
lại mô tả từ đầu và cầu may. Payload có index chính là thứ biến pha 2 thành
"lọc và duyệt".

---

## Thứ tự thực hiện (chi tiết)

Chia làm hai luồng chạy song song: **luồng GPU** (dài, chạy nền, ít người) và
**luồng code** (ngắn, nhiều người làm được cùng lúc). GPU là tài nguyên hiếm
nhất — đừng để nó rảnh trong lúc chờ code.

```
LUỒNG GPU (chạy nền, ưu tiên khởi động sớm)
  benchmark 500 frame  →  shot detection 873 video  →  OCR 177k  →  caption (nếu còn giờ)
       (Q4)                    (bắt buộc)             (P1-1)         (P1-6)

LUỒNG CODE (làm song song)
  P0-1 ┐
  P0-2 ┤
  P0-4 ┼→ P3-5,6 → P1-3 → P2-1 → P1-8 → P1-7 → P1-4 → P2-2 → P1-5 → P2-3,5,6 → P2-7 → P3-còn lại
  P0-3 ┘  (job+DB) (bộ đo) (ingest) (trick) (payload) (RRF)  (latency)

  P1-2 (cào transcript) — không cần GPU, chen vào bất cứ lúc nào
```

Lý do sắp xếp:

- **Benchmark GPU là việc số 0.** Chưa biết Qwen-VL mất 5 giờ hay 25 giờ thì
  không lên lịch được gì. Chạy 500 frame trước mọi thứ khác.
- **Shot detection khởi động sớm nhất trong luồng GPU.** Nó chặn cả `shot_id`
  (dedupe + fusion) lẫn phương án caption-theo-shot của P1-6. Là đường găng dài
  nhất.
- **P1-3 đứng rất sớm.** Không có bộ đo thì mọi item sau chỉ là đoán, và không
  ai biết một thay đổi làm tốt lên hay xấu đi.
- **P3-5 và P3-6 chen lên trước P2-1.** Cả hai đều là bug hạ tầng của đúng
  đường ingest mà P2-1 sắp làm nhanh lên hàng chục lần. Ingest nhanh hơn nghĩa
  là ghi SQLite dày hơn → P3-6 (WAL) từ "hiếm gặp" thành "gặp thường xuyên".
- **P2-1 trước mọi lần ingest thật.** 177k frame: batch-1 mất ~5 giờ, batch-64
  mất ~40 phút. Chênh lệch đó lặp lại mỗi lần ingest lại.
- **P1-8 (trick search) đẩy lên trước P1-4.** Rẻ hơn nhiều mà ăn điểm trực tiếp
  trong vòng thi tương tác. `build_filter` đã có sẵn, chỉ thiếu field trong
  schema.
- **P1-2 (cào transcript) không cần GPU** → giao cho người khác làm bất cứ lúc
  nào, không tranh tài nguyên với luồng GPU.
- **P0-3 rẻ nhất** nhưng làm sau P3-5, nếu không job chết lần nữa là lại khoá
  luôn tên collection.
- **P3-8 (test contract) phải xong trước P1-8**, vì P1-8 sửa `schemas/` — nó là
  thứ duy nhất thực thi được rule "do not change API contracts silently".

**Bỏ khỏi kế hoạch:** bước `sampling`. BTC đã cho `map-keyframes` với đủ
`frame_idx`/`pts_time`/`fps`. Tiết kiệm việc decode 77 GB video.

---

## Open-source — tầng retrieval / AI

Ràng buộc: query path **không được** phụ thuộc cloud; weight phải nằm sẵn trong
HF cache local trước khi thi. Mọi dep mới phải giải thích trước khi thêm (review
rule).

### Đã có sẵn trong stack, nên tận dụng trước khi thêm dep

| Có sẵn | Dùng cho item | Ghi chú |
| --- | --- | --- |
| **Qdrant sparse vectors + Query API RRF** | P1-1, P1-2, P1-4 | Server `v1.12.1` đã hỗ trợ hybrid dense+sparse với `prefetch` + RRF trong **một** request. Không cần thêm Elasticsearch — và thêm cũng vi phạm rule "Qdrant là nguồn metadata filter". |
| `qdrant_client.query_batch_points` | P2-3, P2-4 | Gộp N query một round trip. |
| Qdrant recommend / query-by-id | P1-5 | Relevance feedback không cần encode lại. |
| `torch.utils.data.DataLoader` | P2-1 | `num_workers` cho decode ảnh song song. Không phải dep mới, torch đã pin. |
| `functools.lru_cache` | P2-6 | Cache query embedding. Không cần `cachetools`. |
| SQLite qua `ingestion/store.py` | P3-2 | Query log dùng lại pattern sẵn có. |
| `anyio` (đi kèm Starlette) | P3-3 | Semaphore + timeout, không phải dep mới. |

### Dep mới thực sự cần — kèm lý do

| Thư viện | Cho item | Vì sao chọn | Chi phí |
| --- | --- | --- | --- |
| **PaddleOCR** (PP-OCRv4) | P1-1 | Scene text tiếng Việt tốt nhất trong nhóm open-source, chạy local GPU, weight tải sẵn được. | Kéo theo `paddlepaddle` — một deep learning framework thứ hai bên cạnh torch. Nặng. |
| **EasyOCR** | P1-1 (thay thế) | Chạy trên chính torch đã có → **không thêm framework mới**, hỗ trợ `vi`. | Accuracy trên scene text dày thường thua PaddleOCR. |
| **faster-whisper** | P1-2 | Whisper chạy trên CTranslate2, nhanh hơn nhiều lần bản gốc, int8 trên GPU, hoàn toàn offline. Tiếng Việt tốt ở large-v3. | Thêm runtime CTranslate2, nhưng nhẹ và chỉ dùng ở ingest, **không nằm trong query path**. |
| **WhisperX** | P1-2 (tuỳ chọn) | Word-level timestamp → map sang frame range chính xác hơn. | Chỉ thêm nếu timestamp thô của faster-whisper không đủ. |
| **fastembed** | P1-1, P1-2 | Lib của chính Qdrant, sinh sparse vector (BM25/SPLADE) bằng ONNX, tích hợp thẳng với sparse vector của Qdrant. Nhẹ, không cần torch. | Một dep nhỏ, rủi ro thấp. |
| **ranx** | P1-3 | Thư viện đánh giá IR: Recall@K, MRR, nDCG, **và** có sẵn RRF/CombSUM để so sánh với fusion tự viết. Đúng hai thứ P1-3 và P1-4 cần. | Chỉ là dev dependency → cho vào `requirements-dev.txt`, không đụng production. |
| **youtube-transcript-api** | P1-2 | Cào transcript qua `watch_url` có sẵn trong media-info. Vài phút thay cho ~20 giờ GPU. | Rất nhẹ, chỉ HTTP. Chỉ chạy lúc chuẩn bị, **không nằm trong query path**. |
| **Qwen2-VL** (2B hoặc 7B) | P1-6 | Caption + đọc chữ trong một lượt inference. Giữ được quan hệ giữa vật thể mà dual encoder làm mất. | Chạy trên `transformers` đã có. Chi phí **thời gian** mới là vấn đề, không phải dependency — xem P1-6. |

**Khuyến nghị chọn OCR.** Bắt đầu bằng EasyOCR để không kéo `paddlepaddle` vào
máy thi; nếu P1-3 cho thấy OCR là điểm nghẽn accuracy thì mới đổi sang
PaddleOCR. Đây chính là lý do P1-3 phải làm trước P1-1.

**Phân vai Qwen-VL và OCR chuyên dụng — đừng dùng lẫn.** Qwen-VL chậm hơn OCR
chuyên dụng khoảng một bậc độ lớn. Dùng nó để *đọc chữ* trên 177k frame là trả
giá đắt cho việc mà EasyOCR làm được. Chỗ nó thật sự hơn là **mô tả cảnh và
quan hệ giữa vật thể** (P1-6). Chạy OCR chuyên dụng cho toàn bộ, Qwen-VL cho
caption — và nếu thiếu giờ thì Qwen-VL là thứ cắt trước.

### Đã cân nhắc và loại

| Loại | Lý do |
| --- | --- |
| Elasticsearch / OpenSearch cho text | Xem mục quyết định riêng bên dưới — lý do chính không phải architecture rule mà là **tách từ tiếng Việt phải tự làm dù dùng engine nào**, nên lợi thế analyzer của OpenSearch gần như biến mất. |
| NVIDIA DALI cho decode ảnh | Tăng tốc thật nhưng nặng; `DataLoader` + `num_workers` đã giải quyết phần lớn P2-1. |
| ONNX Runtime / TensorRT cho encoder | Text encoder nhỏ, không phải điểm nghẽn (điểm nghẽn là rerank I/O — P2-2). Cân nhắc lại sau khi P2-2 xong. |
| `sentence-transformers` CrossEncoder | Là reranker text–text, không dùng được cho image–text. BLIP-ITM hiện tại vẫn đúng lựa chọn. |
| Redis cache | Single-team, single-process; `lru_cache` in-process là đủ và bớt một service phải chăm. |
| Prometheus/Grafana | Overkill cho một đội. P3-2 (log vào SQLite) đủ để phân tích sau vòng thi. |

### Quyết định: **không** thêm OpenSearch / Elasticsearch

Câu hỏi hợp lý, vì sau P1-1/P1-6 hệ thống sẽ có rất nhiều text. Nhưng câu trả
lời là không, và lý do quyết định không phải architecture rule.

**Lập luận quyết định: tách từ tiếng Việt phải tự làm dù dùng engine nào.**

Từ tiếng Việt nhiều âm tiết — "giá xăng" là 2 âm tiết, 1 khái niệm. Tokenizer
mặc định của cả Qdrant lẫn OpenSearch đều cắt theo khoảng trắng, tức xé đôi
khái niệm. Muốn đúng thì phải chạy tách từ (`underthesea`, `pyvi`,
`VnCoreNLP`) **ở cả lúc index lẫn lúc query**, tự làm, bên ngoài search engine.

Mà một khi đã tự tách từ, lợi thế lớn nhất của OpenSearch — bộ analyzer — gần
như không còn giá trị. Còn lại chỉ là chi phí.

**Thứ thật sự mất khi không dùng OpenSearch**

| Mất | Thay bằng |
| --- | --- |
| Phrase query ("giá xăng" đúng cụm) | Sparse vector cho điểm cao khi cả hai token cùng có; đủ dùng trên thực tế |
| Fuzzy / chịu lỗi gõ | Trường không dấu (mục Lược đồ) xử lý phần lớn lỗi gõ thật |
| Highlight vị trí khớp | **Làm ở FE** — OCR text đã nằm trong payload, token query đã biết. Không cần search engine cho việc này |
| Boost theo trường | Qdrant làm được bằng nhiều sparse vector + trọng số RRF |

**Thứ phải trả nếu thêm**

- Hai hệ thống phải đồng bộ → cả một lớp bug mới ("frame có trong Qdrant, thiếu
  trong OpenSearch")
- Hai hệ thống phải deploy, snapshot, restore, và cứu lúc 2 giờ sáng
- Query path phải join kết quả từ hai nguồn → thêm độ trễ và độ phức tạp, trong
  khi Qdrant làm dense + sparse + RRF trong **một** request
- Vi phạm architecture rule "Qdrant là nguồn metadata filter"

**Chốt:** dùng sparse vector của Qdrant với IDF modifier (tương đương BM25),
tự tách từ tiếng Việt trước khi đưa vào. Đổi lại mất phrase query chính xác —
đánh đổi rẻ so với việc phải nuôi thêm một hệ thống trong tuần thi.

---

## Open-source — tầng hạ tầng backend

Nguyên tắc chi phối toàn bộ mục này, lấy từ architecture rule của repo: *"single-team
competition tool, not a multi-user production system"*. Mỗi service thêm vào là
một thứ nữa có thể chết lúc 2 giờ sáng và cần người biết cách cứu. Với một đội
thi, **chi phí vận hành thường lớn hơn lợi ích kỹ thuật**.

Vì vậy phần lớn khuyến nghị dưới đây là "đừng thêm" — kèm lý do cụ thể.

### Nên thêm

| Công cụ | Cho việc gì | Vì sao đáng |
| --- | --- | --- |
| **uv** (thay `pip`) | Dựng máy mới | `requirements.txt` có `torch>=2.6,<3` — resolve và tải wheel vài GB bằng pip rất chậm. `uv pip sync` nhanh hơn nhiều bậc, và `uv.lock` pin cả transitive dep (thứ `requirements.txt` hiện **không** làm). `runbook-ubuntu.md` dài 952 dòng chủ yếu vì setup thủ công. Drop-in, rủi ro thấp. |
| **systemd** | Chạy API bền | Runbook đã có `ExecStart` ở dòng 709. Chỉ cần đảm bảo `Restart=always` + `RestartSec`, và một unit riêng cho runner nếu tách. Có sẵn trên Ubuntu, không phải dep. |
| **structlog** *hoặc* `logging` stdlib | P3-7 | Log JSON có `request_id` (`SearchResponse` đã sinh sẵn `request_id`) để join log với kết quả. stdlib là đủ; structlog chỉ tiện hơn. |
| **`logrotate`** | P3-7 | Ingest chạy hàng giờ sẽ sinh log lớn. Có sẵn trên hệ thống. |
| **Docker + NVIDIA Container Toolkit** cho backend | Dựng máy mới | *Cân nhắc, chưa quyết.* Hiện chỉ Qdrant chạy Docker, backend chạy bare-metal để lấy GPU. Đóng gói cả backend sẽ thu gọn phần lớn runbook. Đánh đổi: GPU passthrough, mount HF cache, và các CLI preprocessing phải chạy qua `docker exec`. **Chỉ đáng nếu phải dựng từ 3 máy trở lên.** |

### Nên **không** thêm — và vì sao

| Bị loại | Lý do |
| --- | --- |
| **Celery / RQ / Dramatiq + Redis hoặc RabbitMQ** | Nghe hợp lý vì ingestion là job nền, nhưng đổi lại là **hai** service phải chăm (broker + worker). Lịch sử job cho thấy tổng cộng 2 job, chạy tay, kéo dài hàng giờ. Thứ thực sự thiếu không phải queue mà là **crash detection** — P3-5 giải quyết bằng ~60 dòng và 0 dependency. Nếu sau này vẫn muốn queue thật, **Huey với backend SQLite** là lựa chọn duy nhất không thêm service. |
| **Airflow / Prefect / Dagster** | Pipeline `probe → shot_detect → sampling → ingest` đúng là một DAG, nhưng đã có `--resume` ở probe/shot_detect và Parquet là checkpoint tự nhiên giữa các bước. Orchestrator ở đây nặng hơn thứ nó điều phối. |
| **Triton / TorchServe / Ray Serve** | Tách model khỏi API process cho phép chạy nhiều uvicorn worker. Nhưng một GPU thì tách hay không cũng vẫn nghẽn ở GPU, mà lại thêm một service và một lớp serialize ảnh qua network. Vấn đề thật là "một query chậm chặn mọi query khác" — P3-3 xử lý bằng semaphore + timeout, rẻ hơn nhiều. |
| **Redis làm cache** | Single process, single machine. `functools.lru_cache` in-process nhanh hơn (không serialize, không network hop) và không thêm service. |
| **Prometheus + Grafana** | `latency_ms` đã được trả về theo từng stage trong mỗi response. Đẩy vào SQLite (P3-2) rồi query bằng SQL là đủ để phân tích sau vòng thi, không cần time-series DB. |
| **Sentry** | Vi phạm "no cloud dependency". Self-host được nhưng nặng gấp nhiều lần thứ nó giám sát. |
| **nginx / Caddy / Traefik** | Chỉ cần nếu phải phục vụ FE qua LAN kèm TLS. Với vấn đề trước mắt — FE khác origin — thì **CORS middleware (P0-4) mới là lời giải đúng**, không phải reverse proxy. Thêm proxy chỉ để sửa CORS là đi vòng. |
| **Kubernetes / Docker Swarm** | Một máy, một đội, một GPU. |
| **PostgreSQL thay SQLite** | Job state chỉ có vài chục row. SQLite + WAL (P3-6) thừa sức. |
| **Alembic** | Schema `ingestion_jobs` là `CREATE TABLE IF NOT EXISTS` chạy mỗi lần connect. P3-5 thêm cột thì viết `ALTER TABLE` có kiểm tra là xong. |

### Dev dependency đáng thêm

| Công cụ | Cho việc gì |
| --- | --- |
| **`ruff`** | Repo hiện **không có** linter/formatter nào, cũng không có `pyproject.toml` — trong khi review rule lại yêu cầu "run relevant formatting, linting". Ruff làm cả hai việc trong một binary, cấu hình tối thiểu. |
| **`httpx`** | Bắt buộc cho `TestClient` ở P3-8. |
| **`pytest-cov`** | Biết chỗ nào chưa được phủ trước khi đụng vào ranking. |
| **`pre-commit`** *(tuỳ chọn)* | Chạy ruff + test OpenAPI drift (P3-8) trước mỗi commit. Chỉ đáng nếu cả đội chịu cài. |

---

## Open questions

1. **Format file nộp của BTC 2026?** Chặn P0-1. Không đoán — cần văn bản BTC.
2. **QA đi hướng nào** — operator nhập tay hay VQA model? Chặn P0-2.
3. **Dataset đầy đủ nằm ở đâu và bao nhiêu keyframe?** `ingestion.db` cho thấy
   lần chạy thành công chỉ có 5314 frame trên server `/home/ubuntu/tung/`. Con
   số thật quyết định P2-1 và P2-7 gấp tới mức nào.
4. **Cấu hình máy thi (VRAM)?** Quyết định được giữ `so400m` hay lên `giant`,
   và liệu SigLIP + BLIP + OCR có cùng resident được không.
5. **Có tập query có nhãn từ mùa trước không?** Chặn P1-3, mà P1-3 lại chặn gần
   như mọi thứ còn lại.

Câu 3 đã có đáp án sau khảo sát 2026-08-14: **873 video, 177,321 keyframe.**

Câu 4 giờ là câu **chặn nặng nhất**. Không biết GPU/VRAM thì không ước lượng
được shot detection, OCR hay Qwen-VL mất bao lâu — mà ba việc đó cộng lại có
thể từ 8 giờ tới hơn 2 ngày. Chênh lệch đó quyết định caption có làm được
không, và có kịp ingest lại lần hai không.

**Việc số 0 của cả kế hoạch:** benchmark 500 frame trên GPU thật cho
EasyOCR / PaddleOCR / Qwen2-VL-2B / Qwen2-VL-7B, lấy số giờ thật rồi mới chốt.
