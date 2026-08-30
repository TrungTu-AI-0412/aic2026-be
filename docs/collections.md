# Collection trên máy GPU — có gì bên trong

Cập nhật 27/08/2026. Tài liệu này mô tả **collection đã nạp**, khác với
[`data-pipeline.md`](data-pipeline.md) là mô tả **parquet nguồn**. Hai thứ
không trùng nhau: parquet của repo này lấy mẫu keyframe khác với bộ đã nạp,
nên `keyframe_n` ở hai bên **không** cùng nghĩa (xem mục 5).

---

## 1. Ba collection

| Tên | Điểm | Dense | Sparse | Trạng thái |
| --- | ---: | --- | --- | --- |
| `aic2026-frames-v2` | 289.881 | `dense_video` 1536-d | không | nguồn, chỉ đọc |
| `aic2-frames-v1` | 289.881 | `dense` 1536-d SigLIP2-giant | 3 kênh | **đang dùng** |
| `aic2-frames-jinaclip2` | 289.881 | `dense` 1024-d Jina CLIP v2 | 3 kênh | đang dựng |

`aic2026-frames-v2` là bộ vector có sẵn của nhóm. `aic2-frames-v1` **đọc lại
đúng những float đó** thay vì embed lại — vừa tiết kiệm một giờ L40S (đo được
77 ảnh/giây, batch 64), vừa khiến hai collection so sánh được: chênh lệch số
liệu quy hết về phần đã thêm, vì phần thị giác không chỉ tương đương mà là
cùng một dãy số.

`aic2-frames-jinaclip2` cũng vậy: sao nguyên payload và cả ba vector sparse,
**chỉ thay `dense`**. Mọi chênh lệch giữa nó và `aic2-frames-v1` quy về image
encoder.

Lý do thử Jina: tháp text của SigLIP2 xử lý tiếng Việt kém. Triệu chứng đo
được trên index thật — truy vấn `sat lo bo song`, tức tiếng Việt gõ không dấu,
đúng kiểu người ta gõ thật, trả về ảnh cận cảnh một con rùa ở vị trí đầu. Jina
CLIP v2 có tiếng Việt trong 89 ngôn ngữ huấn luyện. Nó có thắng hay không thì
**chưa đo**; dựng riêng một collection để so, không thay thẳng.

---

## 2. Một điểm trông như thế nào

```
id      : uuid5 sinh từ (video_id, keyframe_n)

vector  : dense    [1536 float, đã L2-norm]   ← cosine = dot product
          speech   {indices, values}          ← sparse, Modifier.IDF
          ocr      {indices, values}          ← sparse, Modifier.IDF
          caption  {indices, values}          ← sparse, Modifier.IDF

payload :
  video_id            "L21_V001"
  keyframe_n          137          ← số thứ tự keyframe — LÀ khoá
  original_frame_id   4102         ← số frame để nộp bài — KHÔNG phải khoá
  shot_id             58
  pts_sec             102.24       ← mốc thời gian của keyframe
  shot_start_sec      101.6
  shot_end_sec        104.44
  path                "…/L21_V001/137.jpg"
  ocr_text            chữ trên màn hình, EasyOCR
  ocr_text_vlm        chữ trên màn hình, VLM
  ocr_regions         toạ độ vùng chữ
  caption_vi          mô tả cảnh, tiếng Việt
  asr_text            lời nói
  asr_text_corrected  lời nói đã thêm dấu câu
  objects             RỖNG — xem mục 6
  asr_entities        RỖNG — xem mục 6
  title               RỖNG — xem mục 6
  author              RỖNG — xem mục 6
  channel_id          RỖNG — xem mục 6
  publish_date        RỖNG — xem mục 6
```

Ba vector sparse gộp nguồn khác nhau:

| Vector | Gộp từ | Bỏ dấu? |
| --- | --- | :---: |
| `speech` | `asr_text` + `asr_text_corrected` | **không** |
| `ocr` | `ocr_text` + `ocr_text_vlm` | **có** |
| `caption` | `caption_vi` | có |

Bất đối xứng này là cố ý. Lỗi OCR nằm ở mức dấu thanh — bỏ dấu thì cứu được.
Lỗi ASR nằm ở mức phụ âm — bỏ dấu chỉ làm nhập nhằng thêm mà không cứu được gì.

Giá trị sparse được **chuẩn hoá L2** (`sparse.encode_document`), không phải đếm
thô. Thiếu bước này thì điểm không có chuẩn hoá độ dài, và những frame nhiều
chữ nhất thắng cả những truy vấn chúng không liên quan: content@1 0,625 so với
0,735 trên cùng 300 câu. Chọn L2 thay BM25 vì số hạng bão hoà của BM25 cần độ
dài tài liệu trung bình của toàn corpus, mà một lượt duyệt manifest theo dòng
thì không biết được. Phần IDF do Qdrant tự lo ở phía server.

---

## 3. Payload index

15 field có index: `video_id`, `keyframe_n`, `original_frame_id`, `shot_id`,
`title`, `author`, `channel_id`, `publish_date`, `objects`, `ocr_text`,
`ocr_text_vlm`, `caption_vi`, `asr_text`, `asr_text_corrected`, `asr_entities`.

Index **không thêm dữ liệu gì** — nó dựng cấu trúc tra cứu trên payload đã có
sẵn. Thiếu nó thì filter không báo lỗi, chỉ âm thầm quét toàn bộ 289.881 điểm
để trả về vài trăm.

---

## 4. Độ phủ

Đếm trực tiếp trên collection (`points/count` với `must_not: is_empty`), không
phải suy từ parquet. Hai collection khớp nhau từng con số, tức là bước sao chép
payload sang `jinaclip2` không mất gì.

| Trường | Điểm có | Phủ |
| --- | ---: | ---: |
| TỔNG | 289.881 | 100% |
| `pts_sec` | 289.881 | 100% |
| `caption_vi` | 281.754 | 97,2% |
| `asr_text_corrected` | 247.818 | 85,5% |
| `asr_text` | 247.669 | 85,4% |
| `ocr_text` | 233.491 | 80,5% |
| `ocr_text_vlm` | 149.360 | 51,5% |
| `title`, `author`, `channel_id`, `publish_date` | **0** | **0%** |
| `objects`, `asr_entities` | **0** | **0%** |

`ocr_text_vlm` thấp vì VLM chỉ chạy trên những frame EasyOCR trả về rỗng, không
chạy lại toàn bộ — đó là thiết kế, không phải thiếu sót.

Ước lượng trước đây suy từ parquet là **bi quan**: đoán `caption_vi` 86,6% và
`ocr_text_vlm` 40,9%, số thật cao hơn cả hai. Ghi lại đây để lần sau đừng báo
cáo con số suy diễn như thể đã đo.

---

## 5. Danh tính điểm — chỗ dễ sai nhất

Khoá của một keyframe là `(video_id, keyframe_n)`, **không bao giờ** là
`(video_id, original_frame_id)`.

`original_frame_id` suy ra từ presentation timestamp làm tròn, nên hai keyframe
liên tiếp có thể trùng số — xảy ra ở 192/873 video của bộ này. Point id dựng từ
frame index đã khiến 614 keyframe ghi đè lên nhau lúc upsert mà **không có gì
báo lỗi**. `original_frame_id` là thứ bài nộp báo cáo; nó không phải khoá.

Sai lầm thứ hai, tinh vi hơn: `keyframe_n` của repo này và `keyframe_n` của bộ
đã nạp là **hai cách đánh số khác nhau** vì hai lần lấy mẫu khác nhau. Ghép hai
manifest trên cột đó thì gán nhầm chữ vào frame, âm thầm, không lỗi.
`scripts/join_server_frames.py` vì thế ghép trên `(video_id, shot_id)` cộng
timestamp gần nhất **trong đúng shot đó**. 89,4% frame đích tìm được dòng nguồn.

---

## 6. Lỗ đã biết

**Sáu trường có index nhưng payload rỗng.** `objects` và `asr_entities` bị loại
khỏi `CARRIED` của `join_server_frames.py`. `title`, `author`, `channel_id`,
`publish_date` thì chưa bao giờ được ghép vào. Hệ quả: **bốn hướng lọc không
hướng nào chạy** — theo vật thể, theo tên riêng, theo kênh, theo ngày đăng. Đều
trả về rỗng, không lỗi, chỉ là không có gì. Sửa mất khoảng 10 giây ghép cộng
một lượt ghi Qdrant.

**Không có ngưỡng khớp yếu.** Truy vấn `Nguyen Xuan Son` trả về `CHÙA HỮU SƠN`
vì trúng mỗi chữ "sơn", và điểm số không hề báo đó là khớp rác. Cần một ngưỡng
điểm tối thiểu hoặc ngưỡng độ phủ token.

**Bộ eval sinh từ ASR.** Chạy eval khi vector speech đang bật tức là chấm index
lexical bằng chính văn bản của nó. Bộ này không phán xử công bằng được kênh OCR
lẫn index dense. Cần một bộ eval viết từ chữ trên màn hình.

---

## 7. Dùng collection nào thì đặt gì

`FEATURE_PROFILE` trong `.env` **phải** khớp profile mà collection đang hoạt
động được nạp — cùng model, cùng số chiều, cùng không gian.

| Collection | `FEATURE_PROFILE` | Giới hạn token của tháp text |
| --- | --- | ---: |
| `aic2-frames-v1` | `siglip2-giant-opt-patch16-384-v1` | **64** |
| `aic2-frames-jinaclip2` | `jina-clip-v2` | 8192 |

Đặt sai thì không sập ngay — truy vấn vẫn chạy nếu số chiều tình cờ khớp, và
trả về rác.

### Giới hạn token, và vì sao nó quan trọng với query rewriting

`embed_text` gọi processor với `truncation=True`. Query dài quá giới hạn bị cắt
đuôi **âm thầm**: không exception, không lỗi, và kết quả trả về vẫn là một bảng
xếp hạng trông hoàn toàn bình thường, chỉ là tính từ phần đầu câu.

Với `aic2-frames-v1` thì trần là **64 token**, rất chật. Module rewriting nào
sinh câu dài hơn thế đang ném đi phần đuôi — mà rewriting hay đẩy chi tiết phân
biệt xuống cuối, tức mất đúng phần đáng giá nhất.

**Đếm token, đừng đếm từ.** Tokenizer đa ngữ của SigLIP2 cắt một từ tiếng Việt
có dấu thành hai đến ba token. Một câu 40 từ tiếng Việt vượt 64 token dễ như
không, trong khi 40 từ tiếng Anh thì không. Cap theo số từ là đoán.

Con số nằm ở `FeatureProfile.max_text_tokens` để đọc bằng code thay vì chép tay:

```python
get_profile(settings.feature_profile).max_text_tokens
```

Với `jina-clip-v2` trần là 8192 — rewriting không cần dè chừng gì cả.

`embed_text` giờ ghi log `WARNING` kèm số token bị mất khi có cắt. Cố ý **không**
raise: một query bị cắt vẫn cho kết quả dùng được, làm hỏng hẳn truy vấn giữa
giờ thi thì tệ hơn. Nó chỉ cần thôi vô hình.

---

## 8. Dựng lại

```bash
# ghép chữ của repo này lên bộ lấy mẫu khác
python scripts/join_server_frames.py \
    --source data/frames.parquet --target <frames của bộ đã nạp> \
    --out data/frames-joined.parquet

# dựng collection, dùng lại vector dense có sẵn
python scripts/build_lexical_collection.py \
    --frames data/frames-joined.parquet \
    --source-collection aic2026-frames-v2 --collection aic2-frames-v1

# chạy VLM lên những frame OCR bỏ trống, rồi ghép ngược vào
python scripts/vlm_enrich.py --frames … --out data/vlm.jsonl
python scripts/merge_vlm_enrich.py --input data/vlm.jsonl --collection aic2-frames-v1

# dựng lại với image encoder khác
python scripts/ingest_jina.py \
    --source aic2-frames-v1 --collection aic2-frames-jinaclip2 --recreate
```

Cả bốn đều chạy trên máy GPU và đều nên chạy tách phiên
(`setsid nohup … > log 2>&1 < /dev/null &`) — mất VPN thì job vẫn sống.

`merge_vlm_enrich.py` hiện gửi **một request `set_payload` cho mỗi điểm**;
30.750 điểm mất khoảng 10 phút. Gộp theo lô sẽ nhanh hơn nhiều.
