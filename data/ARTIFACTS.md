# Artifacts phái sinh — AIC 2026

Dữ liệu đã xử lý, sinh từ `raw/` trên cùng bucket. Kéo về là dùng được ngay,
không phải chạy lại bước nào.

Tổng quan đã làm gì / mượn gì / còn thiếu gì: [`docs/data-pipeline.md`](../docs/data-pipeline.md).

Tải:

```bash
rclone copy r2:aicc26/artifacts/ ./data/ --s3-no-check-bucket --progress
```

## Nội dung

| File | Kích thước | Nội dung |
| --- | ---: | --- |
| `frames.parquet` | 143 MB | 177.321 keyframe × 21 cột — manifest chính |
| `clips.parquet` | 23 MB | 97.811 shot |
| `video_bounds.parquet` | 12 KB | 873 video — cận trên frame hợp lệ khi nộp bài |
| `eval_set.jsonl` | 300 câu | Câu hỏi đánh giá **sinh tự động** — đọc kỹ cảnh báo bên dưới |
| `shots_hf.tar.gz` | 577 KB | 873 shot list gốc, dạng `[start_frame, end_frame]` |
| `transcripts.tar.gz` | 803 KB | 64/873 transcript YouTube (chỉ để đối chứng) |

SHA-256:

```
e23229373bb2d3ccdbe18e8a2f772b063e1145bb2c3612ca4a4b5ea4ff6cea38  frames.parquet
7871685986915a332cb8b96d1214cf8a0223e963bce28e9c4e1d6a5cc935edf0  clips.parquet
f2ce275c716cfb612557b0a5f74b2ac89ef55a4d725a5b3097c047893757baf0  video_bounds.parquet
```

## `video_bounds.parquet`

`app.submissions.service` đọc file này để chặn một dòng nộp không thể nào ghi
điểm: video không có trong corpus, hoặc frame nằm ngoài độ dài video. Mỗi lần
thi chỉ có vài lượt nộp nên bắt được sớm là đáng.

| Cột | Ý nghĩa |
| --- | --- |
| `video_id` | |
| `fps` | lấy từ `map-keyframes` |
| `length_sec` | lấy từ `media-info`, đơn vị giây nguyên |
| `frame_upper_bound` | frame hợp lệ là `0 <= f < frame_upper_bound` |

### ⚠ Cận trên cố tình nới rộng, không siết

`length_sec` là **số giây nguyên**, nên `length_sec × fps` luôn ngắn hơn video
thật một chút. Ở **43/873 video**, keyframe cuối cùng đã vượt qua tích đó, nhiều
nhất là 10 frame — chính BTC lấy mẫu những frame ấy.

Nên cận trên lấy giá trị **lớn hơn** giữa `length_sec × fps` và frame lớn nhất
quan sát được (keyframe cuối, shot cuối). Chặn hụt thì mất một đáp án đúng và
không lấy lại được; chặn dư thì chỉ cho lọt vài frame không tồn tại, mà BTC
mới là bên chấm. Hai sai lầm này không cân nhau.

Đã kiểm: **0/177.321 keyframe** trong `frames.parquet` bị cận trên này từ chối.

Sinh lại bằng `--out-videos` của `scripts/build_frames_manifest.py`. Nếu thiếu
file, export vẫn chạy — chỉ mất bước kiểm tra.

## `frames.parquet`

Sáu cột đầu là hợp đồng bắt buộc của `app.ingestion.manifest`; phần còn lại là
enrichment. Pydantic bỏ qua cột lạ nên pipeline đọc thẳng file này.

| Cột | Kiểu | Ghi chú |
| --- | --- | --- |
| `video_id` | string | ví dụ `L21_V001` |
| `shot_id` | int32 | chỉ số shot trong video, đếm từ 0 |
| `keyframe_n` | int32 | **định danh keyframe**, đếm từ 1 |
| `original_frame_id` | int64 | **giá trị đem đi nộp** |
| `pts_sec` | float64 | mốc thời gian trong video |
| `path` | string | `keyframes/<video_id>/<n:03d>.jpg` |
| `objects` | list\<string\> | thực thể phát hiện được, ngưỡng 0.3 |
| `object_counts` | map\<string,int32\> | số lượng mỗi thực thể |
| `asr_text` | string | lời nói trong shot, **bản thô** |
| `asr_text_corrected` | string | bản đã thêm dấu câu và viết hoa |
| `asr_entities` | list\<string\> | NER: người, tổ chức, địa danh |
| `ocr_text` | string | chữ trên màn hình, gộp theo shot, **giữ nguyên văn** |
| `ocr_regions` | int32 | số vùng chữ giữ lại của shot |
| `ocr_text_vlm` | string | chữ trên màn hình do VLM trích, **có dấu đầy đủ** |
| `caption_vi` | string | mô tả cảnh bằng tiếng Việt, trung vị 465 ký tự |
| `title`, `author`, `channel_id`, `publish_date`, `keywords`, `watch_url` | | từ `media-info` |

### ⚠ Dùng cả `asr_text` lẫn `asr_text_corrected`, đừng bỏ bản thô

`asr_text_corrected` thêm dấu câu và viết hoa, nhưng **không sửa lỗi nghe
nhầm**. Ví dụ thật: `"đường băng sông cửu long ... sục lúng"` (đúng ra là
"Đồng bằng sông Cửu Long ... sụt lún") được viết lại thành
`"Đường băng sông Cửu Long của tình trạng sục lún."` — trôi chảy nhưng vẫn
sai. Text sai mà trông tự tin nguy hiểm hơn text sai mà trông lộn xộn.

Lỗi ASR nằm ở **phụ âm** (`sục`≠`sụt`, `lúng`≠`lún`) nên bỏ dấu **không** gộp
lại được — khác hẳn lỗi OCR vốn nằm ở dấu và chuẩn hoá cứu được.
`asr_entities` vẫn bắt đúng `sông Cửu Long`, `Hà Nội` kể cả khi câu quanh nó
hỏng, nên đó là trường đáng tin nhất trong ba trường ASR.

### ⚠ Hai cột dễ nhầm lẫn chết người

**`keyframe_n` là danh tính, `original_frame_id` là giá trị đem nộp.** Không
hoán đổi hai cái này.

`frame_idx` được suy ra từ presentation timestamp đã làm tròn, nên **hai
keyframe liên tiếp có thể trùng `frame_idx`** — xảy ra ở 192/873 video, tổng
614 keyframe. Dùng `(video_id, original_frame_id)` làm khoá thì 614 keyframe
ghi đè lên nhau lúc upsert vào Qdrant, **không có lỗi nào báo ra**.

Tên file ảnh keyframe theo `keyframe_n`, **không** theo `original_frame_id`.

## Độ phủ

| Lớp | Phủ | |
| --- | ---: | --- |
| `shot_id` | 100% | đã kiểm chứng, drift thời lượng max 0.57s |
| `original_frame_id` | 100% | |
| `objects` | 95.0% | 5% còn lại là ảnh chữ/cảnh chuyển, không detection nào vượt 0.3 |
| media-info | 100% | |
| ASR | **94.1%** | 863/873 video; 25 video còn lại là nhạc thuần tuý (Music/Percussion/Flute), không có lời để chép |
| ASR entities | 57.3% | 38% câu có text nhưng không chứa thực thể nào — NER không sót (kiểm 0/510) |
| **OCR** | **90.7%** | 160.776/177.321 keyframe; 78.341/97.811 shot (80.1%) |
| **Caption** | **97.4%** | Vintern VLM, tiếng Việt — mượn, Apache-2.0 |

10.169/97.811 shot (10.4%) không chứa keyframe nào — shot ngắn hơn lưới
~1 keyframe/giây của BTC. Không phải lỗi; những shot đó chỉ tìm được qua clip
vector.

## Nguồn gốc

- `shots_hf` lấy từ HF dataset `tanp21/aic-hcmc-2025-videos`,
  `annotations/shot_json/`, phần L21–L30. Đã kiểm chứng bằng
  `scripts/verify_shots.py` đối chiếu với `map-keyframes` + `media-info`.
- `frames.parquet` / `clips.parquet` sinh bằng
  `scripts/build_frames_manifest.py`.
- **ASR** lấy từ Kaggle `zzzlazy/aic-asr` (**Apache-2.0**), phủ đúng 873/873
  video của bộ này. Tải lại bằng
  `kaggle datasets download zzzlazy/aic-asr --unzip`.
- `transcripts` sinh bằng `scripts/scrape_transcripts.py` — chỉ 64/873 vì
  YouTube chặn IP; giữ lại làm đối chứng chéo với nguồn ASR trên, chỗ nào hai
  nguồn bất đồng là chỗ ASR không đáng tin.

## Vẫn cần tải riêng từ `raw/`

| | Dung lượng |
| --- | ---: |
| Ảnh keyframe — bắt buộc để chạy embedding | 28.7 GiB |
| Video — chỉ cần nếu làm clip embedding | 77.3 GiB |
| `clip-features-32` của BTC | 168 MB |

Cột `path` trỏ tới ảnh keyframe; chưa tải ảnh thì chưa ingest được.

## `eval_set.jsonl`

300 câu hỏi sinh tự động từ ASR bằng `scripts/build_eval_set.py`. Chấm điểm
bằng `app.eval.metrics`, chạy bằng `python -m app.eval.runner`.

**Không có ground truth công khai cho cuộc thi này** — BTC chưa từng phát hành
bộ query cũ, ba paper 2025 cũng không công bố tập đánh giá của họ. Nên đây là
thứ suy ra được, không phải thứ mượn được.

### ⚠ Vòng lặp: đừng dùng nó để chứng minh nhánh lexical tốt

Câu hỏi lấy từ ASR của shot, mà chính ASR đó nằm trong index lexical. Nhánh
speech **thắng theo thiết kế**. Câu "hybrid search tăng recall@1" đo bằng bộ
này không chứng minh được gì.

Mỗi dòng có trường `source` để biết nó sinh ra từ kênh nào. Muốn đọc trung
thực thì **tắt kênh đó đi**:

```bash
python -m app.eval.runner --eval-set ../data/eval_set.jsonl --no-hybrid
```

Thứ nó **vẫn làm được**:

- Đo index thị giác có tìm ra khoảnh khắc từ lời nói trong đó không (tắt speech)
- **Bắt hồi quy** — recall@5 tụt 10 điểm là thật, bất kể kênh nào ăn điểm ban đầu
- So sánh hai cấu hình **cùng thiết lập kênh**

Thứ nó **không làm được**: nói độ chính xác tuyệt đối của hệ thống, hoặc phân
xử giữa hai kênh mà một trong hai đã ra đề. Muốn vậy phải có người xem video và
gõ câu hỏi tay.

### Bốn sai lầm đã sửa khi dựng

| Sai lầm | Hậu quả | Cách sửa |
| --- | --- | --- |
| Tổng IDF toàn bộ token | Thưởng shot **dài**, top toàn bài giảng 157 keyframe | Chỉ tính `TOP_TOKENS=8` token hiếm nhất |
| Cắt 32 từ đầu | Câu hỏi thành *"Xin chào tất cả các bạn"* | Chọn **cửa sổ 32 từ nhiều IDF nhất** |
| Không lọc token hiếm | IDF thưởng **lỗi ASR** — *"Bi tơ ri Cu nốp"* thắng *"Angelina Jolie"* | Bỏ token có df < `MIN_TOKEN_DF=3` |
| Một shot một câu hỏi | Shot kề nhau chung đoạn ASR → hỏi trùng, một đáp án chắc chắn bị tính sai | Gom theo câu hỏi, **một câu nhiều shot đúng** |

### Số liệu bản hiện tại

```
shots có ASR      : 93.368 / 97.811
ứng viên hợp lệ   : 82.990
câu hỏi phân biệt : 17.350
đã chọn           : 300      (0 câu trùng)
  >1 shot đúng    : 202
video phủ         : 123
trung vị shot/đáp : 4
```

Trường `reviewed` đang là `false` ở toàn bộ 300 dòng — **chưa ai duyệt tay**.
Đó là việc còn lại và không tự động hoá được.

Cách chấm chính thức là **Mean of Top-k R-Score**, k ∈ {1, 5, 20, 50, 100},
nộp tối đa 100 dòng (cả MERVIN lẫn Vortex đều ghi vậy). Công thức R-Score chi
tiết không được công bố, nên `mean_top_k_recall` dùng hit-at-k nhị phân — giống
về hình dạng, **không được trích dẫn như điểm chính thức**.

## `ocr_text` — chữ trên màn hình

EasyOCR trên 2×Tesla T4, **133.757 ảnh** = keyframe đầu và cuối của mỗi shot
(thanh chữ chạy nên hai đầu shot mang hai nửa khác nhau của một dòng tin).
Nối vào manifest bằng `scripts/join_ocr.py`.

```
ảnh đã OCR      : 133.757
  có chữ        : 127.336 (95.2%)
vùng chữ thô    : 732.286
vùng chữ giữ lại: 458.062 (62.6%)
shot có OCR     : 78.341 / 97.811 (80.1%)
frame có OCR    : 160.776 / 177.321 (90.7%)
```

### ⚠ TUYỆT ĐỐI KHÔNG LỌC THEO CONFIDENCE

Đây là quyết định quan trọng nhất của bước này. Vùng chữ **có thật**, nguyên
văn, kèm điểm tin cậy EasyOCR trả về:

```
'Tam DUnG LuU Thong'                 0.15
'doi Voi Xe 3 BaNH TRO LeN'          0.11
'NGuoi Dan Di Lai CHu Y Quan Sat'    0.20
```

Đó là *"Tạm dừng lưu thông"*, *"Đối với xe 3 bánh trở lên"*, *"Người dân đi lại
chú ý quan sát"* — **đọc đúng hết**. Điểm thấp vì chữ in hoa không dấu, không
phải vì đọc sai. Cắt ở ngưỡng 0.3 sẽ **vứt 29,2% tổng số ký tự**, và vứt đúng
vào phần thanh tin — thứ mà cả job này chạy để lấy.

Chúng sống sót nhờ `app.features.sparse` bỏ dấu: `"Tam DUnG LuU Thong"` lưu
trong index và `"tạm dừng lưu thông"` người dùng gõ đều rút về
`"tam dung luu thong"`. **Đã kiểm: 4/4 và 8/8 token khớp.**

Lọc theo **độ dài** thay vì độ tin cậy:

| Độ dài | % vùng | % ký tự | Xử lý |
| --- | ---: | ---: | --- |
| 1–2 | 16.3% | 3.3% | **bỏ** — `IA`, `1n`, `IH`, dễ đụng từ ngắn thật |
| 3–4 | 31.6% | 15.2% | giữ nếu score ≥ 0.5 — lẫn lộn `giây`(0.99) với `Hhd`(0.28) |
| ≥5 | 52.1% | 81.5% | **giữ hết**, bất kể score |

### OCR sửa được đúng chỗ ASR sai nhất

Cùng một shot `L21_V001 kf6` — chính là ví dụ hỏng ghi ở mục ASR bên trên:

```
ASR đã sửa : "Đường băng sông Cửu Long của tình trạng sục lún..."
OCR        : "TIN CHÍNH TÌNH TRẠNG SỤT LÚN Ở ĐBSCL ĐANG DIỄN RA RẤT NHANH"
```

Ở mức token: ASR có `sục` (sai), OCR có `sụt` (đúng) và có thêm `đbscl`. Đây là
lý do hai kênh phải để **riêng hai sparse vector**, không gộp chung.

### Gộp theo shot, không theo frame

Chữ của cả hai keyframe được OCR sẽ hợp lại rồi gán cho **mọi keyframe trong
shot**, giống cách `asr_text` đang làm. `dedupe.dedupe_by_shot` gom một shot
thành một hit trước khi kết quả rời engine, nên shot là mức mà ranking thực sự
nhìn thấy.

## `caption_vi` và `ocr_text_vlm` — mô tả cảnh bằng VLM

Nguồn: Kaggle `zzzlazy/aic-vintern-ocr` (**Apache-2.0**) — cùng tác giả với bộ
ASR đang dùng. Tên có chữ "ocr" nhưng thực chất là **Vintern**, một VLM tiếng
Việt, mô tả từng keyframe bằng văn xuôi. Nối vào bằng
`scripts/join_captions.py`.

Dùng lại nó thay cho **27–49 giờ suy luận VLM** phải tự chạy. Đo trước khi nhận:

```
phủ keyframe của ta        97.4%   (172.684 / 177.321)
caption phân biệt trong video  trung vị 100%, không video nào < 50%
độ dài                     trung vị 465 ký tự
khớp OCR của ta            32.6% token chung, so với 12.3% nếu lấy
                           caption của frame ngẫu nhiên — gấp 2.7×
```

Con số cuối là con số quan trọng. Nó vừa xác nhận caption mô tả đúng frame nó
gắn vào, vừa xác nhận phép nối theo `(video_id, keyframe_n)` là đúng — nối sai
thì kết quả phải rơi về mức ngẫu nhiên.

### ⚠ Đã loại một bộ caption khác

`dngomnh/aic-captions-v2` (MIT, tiếng Anh) phủ 873/873 video và **lấp đủ 30
video Vintern thiếu**, nhưng đo cùng cách chỉ đạt **gấp 2.1×** (5.7% tuyệt đối
so với 32.6%). Đã đo lại bằng token trung lập ngôn ngữ để không thiệt cho tiếng
Anh — vẫn thấp.

Mẫu cụ thể trên `L21_V001`:

| keyframe | OCR đọc pixel | Caption tiếng Anh nói |
| --- | --- | --- |
| 6 | thanh tin HTV về sụt lún ĐBSCL | `"KOREA"`, logo `"KBS TV 9"` |
| 18 | biển báo sạt lở | hai người dẫn cầm bát đỏ trong trường quay |

Thêm hai dấu hiệu: chỉ **83.8%** trùng theo `(video, n)`, và cột `frame_idx`
của họ **chỉ khớp 2.309/177.321** với `original_frame_id` — họ đánh số keyframe
theo cách khác. **Chưa chứng minh được là lệch, nhưng đủ để không dùng.**

### Hai cột, không phải một

`caption_vi` là mô tả — một bản diễn giải. `ocr_text_vlm` là chữ VLM **trích
trong ngoặc kép**, tức bản chép lại. Chúng vào hai sparse vector khác nhau và
**không được gộp**: 465 ký tự mô tả cảnh sẽ nhấn chìm một dòng tin ngắn.

`ocr_text_vlm` đáng giá vì **Vintern đọc chữ Việt tốt hơn EasyOCR rõ rệt**.
Cùng keyframe `L21_V001` #18:

```
EasyOCR : CẢMH BÁO SẠT LỎ ... ĐẾl VdI XE 3 BÁNH TRỈ LÊNl~ ... CHÚ Ý_qUAILSÁT
Vintern : CẢNH BÁO SẠT LỞ NGUY HIỂM TẠM DỪNG LƯU THÔNG
          ĐỐI VỚI XE 3 BÁNH TRỞ LÊN NGƯỜI DÂN ĐI LẠI CHÚ Ý QUAN SÁT
```

Đúng hết, đủ dấu. Nên `ocr_text_vlm` được **thêm cạnh** `ocr_text` chứ không
thay thế — cả hai cùng nuôi một sparse vector `ocr`, và bản đọc của EasyOCR
không bao giờ bị xoá. Hai bên hỏng ở kiểu chữ khác nhau.

```
ocr_text_vlm có nội dung        : 84.690 / 177.321  (47.8%)
có chữ mà EasyOCR không thấy gì :  2.024
```

### Chỉ gắn vào frame, không gắn vào clip

Caption là **theo keyframe**, khác với lời nói và chữ màn hình vốn theo shot.
Ghép caption của cả shot lại sẽ nhồi hàng nghìn ký tự vào một dòng, mà
`clips.parquet` chủ yếu phục vụ 10,4% shot **không có keyframe nào** — những
shot đó cũng không có caption.
