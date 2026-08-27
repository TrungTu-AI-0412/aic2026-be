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
  path                "…/L21_V001/137.jpg"
  title, author, channel_id, publish_date
  ocr_text            chữ trên màn hình, EasyOCR
  ocr_text_vlm        chữ trên màn hình, VLM
  ocr_regions         toạ độ vùng chữ
  caption_vi          mô tả cảnh, tiếng Việt
  asr_text            lời nói
  asr_text_corrected  lời nói đã thêm dấu câu
  objects             RỖNG — xem mục 6
  asr_entities        RỖNG — xem mục 6
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

## 4. Độ phủ — đo ở đâu, chưa đo ở đâu

| Trường | Phủ | Nguồn con số |
| --- | ---: | --- |
| `caption_vi` | ~86,6% | ghép parquet |
| `asr_text` | ~85,4% | ghép parquet |
| `ocr_text` | ~80,5% | ghép parquet |
| `ocr_text_vlm` | ~40,9% | ghép parquet |

**Cảnh báo:** những con số này đo trên parquet *trước khi nạp*, cộng bộ đếm của
bước merge. **Chưa đếm trực tiếp trong collection.** Bằng chứng gián tiếp thì
nhất quán — merge báo 30.750 điểm cập nhật, 27.608 thêm OCR, 30.600 thêm
caption, 0 điểm lạc; 18/20 tiêu đề lấy ngẫu nhiên truy ra đúng video — nhưng
tin không phải là đo.

Số duy nhất đo trực tiếp trên collection: **289.881 điểm, exact, status green.**

Vì sao `ocr_text_vlm` chỉ 40,9%: VLM chỉ chạy trên 30.750 frame mà EasyOCR trả
về rỗng, không chạy lại toàn bộ. Đúng những frame đó là phần merge lấp vào.

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

**`objects` và `asr_entities` có index nhưng payload rỗng.** Chúng bị loại khỏi
`CARRIED` của `join_server_frames.py` và chưa ghép lại. Hệ quả: filter theo vật
thể hoặc theo tên riêng trả về rỗng — không lỗi, chỉ là không có gì. Sửa mất
khoảng 10 giây ghép cộng một lượt ghi Qdrant.

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

| Collection | `FEATURE_PROFILE` |
| --- | --- |
| `aic2-frames-v1` | `siglip2-giant-opt-patch16-384-v1` |
| `aic2-frames-jinaclip2` | `jina-clip-v2` |

Đặt sai thì không sập ngay — truy vấn vẫn chạy nếu số chiều tình cờ khớp, và
trả về rác.

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
