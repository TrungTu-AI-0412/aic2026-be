# ASR transcripts — `data/transcripts/`

Nguồn: Kaggle `zzzlazy/aic-asr` (tải bằng `scripts/download_kaggle_transcripts.py`).
873 file `<video_id>_segments_enriched.csv`, 50 MB, **phủ đủ 873/873 video** của
corpus (L21–L30). Đây là nguồn ASR mà `scripts/build_frames_manifest.py --asr-csv`
đọc để sinh `asr_text` / `asr_text_corrected` / `asr_entities` trong
`frames.parquet`; `--transcripts` (phụ đề YouTube, 64 video) chỉ là fallback.

## Hình dạng

40.023 segment  / 23 cột. Một dòng = một đoạn audio, không phải một keyframe.

| Số liệu | Giá trị |
| --- | ---: |
| segment / video (min / trung vị / max) | 1 / 19 / 572 |
| tổng thời lượng có tiếng nói | ~123 giờ |
| segment có `text` | 35.997 (90%) |
| segment rỗng cả hai cột text | 4.017 |
| tổng số từ (bản thô) | ~1,48 triệu |
| dòng có ít nhất một entity | 8.780 |

## Cột

Nhóm timing + nội dung (khác nhau theo dòng):

| Cột | Kiểu | Ghi chú |
| --- | --- | --- |
| `segment` | int | thứ tự trong video, đếm từ 1 |
| `start`, `end` | int | **giây, đã làm tròn** — dùng `duration` nếu cần chính xác |
| `duration` | float | độ dài thật; lệch `end - start` ở 16% dòng, tối đa ~4s |
| `text` | str | lời nói **bản thô**: toàn chữ thường, không dấu câu |
| `text_corrected` | str | LLM thêm dấu câu + viết hoa; **không sửa từ nghe sai** |
| `entities` | str | dict Python (`ast.literal_eval`, không phải JSON): `persons` / `orgs` / `locations` / `others` |
| `top_classes` | str | AudioSet tags `Nhãn:score;...` |
| `top_classes_pairs` | str | y hệt `top_classes`, dạng list-of-tuple Python — **dư** |
| `labels_en`, `labels_vi` | str | tên nhãn tách riêng, EN và VI |
| `speech_score` | float | 0–1, trung vị 0,97 |
| `has_speech` | bool | 36.051 True / 3.972 False |

Nhóm metadata video (**hằng số trong cả file**, lặp lại ở mọi dòng — đây là lý do
50 MB): `video_id`, `title`, `publish_date`, `author`, `channel_id`,
`channel_url`, `watch_url`, `thumbnail_url`, `length`, `keywords_text`.

## Cạm bẫy

- **`has_speech` không đáng tin theo cả hai chiều.** 162 segment có transcript
  nhưng cờ báo `False`; 216 segment cờ `True` mà text rỗng. Lọc theo cờ này là
  mất dữ liệu — `load_asr_csv` lọc theo "có text hay không".
- **`text_corrected` rỗng ghi là chuỗi `"nan"`**, không phải ô trống. Phải xử lý
  riêng, nếu không sẽ nhét chữ "nan" vào index.
- **`text_corrected` trôi chảy nhưng vẫn sai.** LLM chỉ chuẩn hoá hình thức; từ
  nghe sai vẫn sai và giờ trông như câu đúng. Giữ cả hai cột: bản thô là thứ
  duy nhất cho biết đoạn đó có đáng tin không.
- **`start`/`end` là giây nguyên** nên không map 1-1 sang `pts_sec` của keyframe.
  Manifest builder lấy mọi segment giao với `[start/fps, (end+1)/fps)` của shot.
- **16.821 khoảng trống** giữa các segment liên tiếp (không có overlap). Trung
  bình `sum(duration) / length` = 0,955 — 4,5% thời lượng video không có segment
  nào, không phải mọi giây đều tra được ASR.
- **`labels_*` gần như vô dụng để lọc**: 36.375/40.023 dòng là `Speech`,
  10.558 là `Music`. Đuôi phân bố (`Frying (food)`, `Livestock`, `Insect`) thì
  quá thưa.
- Không có `nan`/`None` ở nhóm metadata, và không có video nào `end` vượt
  `length`, nên không cần clamp.
