# Nguồn dữ liệu mượn — GHI CHÚ NỘI BỘ

**KHÔNG ĐẨY FILE NÀY LÊN GIT.** Thư mục `private/` đã bị `.gitignore` chặn.

Lý do giữ riêng là **cạnh tranh**, không phải pháp lý: cả bốn bộ dưới đây đều
có giấy phép mở và dùng lại hoàn toàn hợp lệ. Chỉ là không có lý do gì để chỉ
đường cho đội khác.

Ghi công đầy đủ đi kèm dữ liệu trên R2 (`artifacts/ARTIFACTS.md`), nên nghĩa vụ
attribution của Apache-2.0 và MIT vẫn được thoả. Repo công khai chỉ chứa code,
không phát tán dữ liệu của ai.

---

## 1. Shot boundaries → cột `shot_id`

| | |
| --- | --- |
| Nguồn | HuggingFace dataset **`tanp21/aic-hcmc-2025-videos`** |
| Đường dẫn trong bộ | `annotations/shot_json/`, phần `L21`–`L30` |
| Phủ | **873/873** video |
| Thay thế được việc gì | Chạy TransNetV2 trên 77 GiB video — bước từng deadlock vì callback logging của PyAV |

Đã kiểm chứng bằng `scripts/verify_shots.py`, đối chiếu `map-keyframes` +
`media-info`, **không cần tải video**:

```
873/873 video khớp
drift thời lượng : trung vị 0.24s, tối đa 0.57s
keyframe nằm đúng shot : 99.914%
```

## 2. ASR → `asr_text`, `asr_text_corrected`, `asr_entities`

| | |
| --- | --- |
| Nguồn | Kaggle **`zzzlazy/aic-asr`** |
| Giấy phép | **Apache-2.0** |
| Phủ | **873/873** video (94,1% keyframe) |
| Tải lại | `kaggle datasets download zzzlazy/aic-asr --unzip` |
| Định dạng | `<video_id>_segments_enriched.csv` |

Thay thế việc tự cào transcript YouTube, vốn **chết hẳn ở 64/873** vì YouTube
chặn IP vĩnh viễn (thử lại 3 lần cách 15 phút rồi 2 lần cách 60 phút đều về 0;
yt-dlp cũng dính 429).

⚠ Cột `has_speech` **không đáng tin** — 162 đoạn có transcript nhưng cờ báo
`False`. Lọc theo cờ là mất hết. `scripts/build_frames_manifest.load_asr_csv`
lọc theo *có text*, không theo cờ.

## 3. Caption VLM tiếng Việt → `caption_vi`, `ocr_text_vlm`

| | |
| --- | --- |
| Nguồn | Kaggle **`zzzlazy/aic-vintern-ocr`** (cùng tác giả mục 2) |
| Giấy phép | **Apache-2.0** |
| Model | **Vintern** — VLM tiếng Việt |
| Phủ | **97,4%** keyframe (172.684 / 177.321) |
| Tải lại | `kaggle datasets download zzzlazy/aic-vintern-ocr --unzip` |
| Thay thế được việc gì | **27–49 giờ** suy luận VLM + tải 28,7 GiB ảnh |

⚠ Tên có chữ "ocr" nhưng **không phải OCR** — là caption văn xuôi. Đừng nhầm.

⚠ File mã hoá **UTF-16**. Đọc bằng UTF-8 thì decode ra ký tự null xen kẽ và
**mất sạch dòng mà không báo lỗi**.

Kiểm chứng trước khi nhận:

```
phủ keyframe                   97,4%
caption phân biệt trong video  trung vị 100%, không video nào < 50%
độ dài                         trung vị 465 ký tự
khớp OCR của ta                32,6% vs 12,3% (frame ngẫu nhiên) — gấp 2,7×
```

Chỉ số cuối làm hai việc: xác nhận caption tả đúng frame, và xác nhận phép nối
`(video_id, keyframe_n)` đúng — nối sai thì phải rơi về mức ngẫu nhiên.

## 4. Caption tiếng Anh → **ĐÃ LOẠI, KHÔNG DÙNG**

| | |
| --- | --- |
| Nguồn | Kaggle **`dngomnh/aic-captions-v2`** |
| Giấy phép | MIT |
| Phủ | 873/873 video, lấp đủ 30 video mục 3 thiếu |
| **Kết luận** | **Không đưa vào index** |

Đo cùng cách chỉ đạt **gấp 2,1×** (5,7% tuyệt đối so với 32,6%). Đã đo lại bằng
token trung lập ngôn ngữ để không thiệt cho tiếng Anh — vẫn thấp.

Mẫu sai rõ trên `L21_V001`:

| keyframe | OCR đọc pixel | Caption tiếng Anh nói |
| --- | --- | --- |
| 6 | thanh tin HTV, sụt lún ĐBSCL | `"KOREA"`, logo `"KBS TV 9"` |
| 18 | biển báo sạt lở | hai người dẫn cầm bát đỏ trong trường quay |

Cột `frame_idx` của họ chỉ khớp **2.309/177.321** với `original_frame_id` — họ
đánh số keyframe theo cách khác. **Chưa chứng minh được là lệch, nhưng đủ để
không dùng.**

Nếu sau này cần lấp 30 video thiếu thì đây là ứng viên — nhưng phải nối theo
`(video, n)`, **tuyệt đối không theo `frame_idx`**.

## 5. Ảnh keyframe (chưa tải)

| | |
| --- | --- |
| Nguồn | Kaggle **`quninhphmanh/ai-challenge-hcmc-2026-keyframes`** |
| Dùng để | Job OCR chạy trên Kaggle đọc thẳng từ đây |
| Kích thước | 28,7 GiB |

Chính vì bộ này đã có sẵn trên Kaggle mà job OCR không phải tải ảnh đi đâu.

---

## Tự làm, không mượn

| | |
| --- | --- |
| **OCR** (`ocr_text`) | EasyOCR, 2×Tesla T4, **7 giờ**, 133.757 ảnh |
| `video_bounds.parquet` | suy từ `media-info` + `map-keyframes` |
| `eval_set.jsonl` | sinh từ ASR bằng `scripts/build_eval_set.py` |

Output OCR thô: `r2:aicc26/artifacts/ocr/` (23 MB, 10 file JSONL + worker).
Giữ lại vì đó là 7 giờ GPU — đổi ngưỡng lọc thì chạy lại `join_ocr.py`, không
phải OCR lại.

## Không tìm được

- **Ground truth công khai** — BTC chưa từng phát hành bộ query cũ; trang chính
  thức không lưu trữ; ba paper 2025 đều không công bố tập đánh giá của họ.
- **Code của các đội** — repo có code chạy được thì **không có giấy phép**
  (mặc định giữ toàn quyền); repo có giấy phép MIT thì cây repo chỉ 4 file.

---

## Dựng lại toàn bộ từ số không

```bash
kaggle datasets download zzzlazy/aic-asr           --unzip -p data/asr_ext
kaggle datasets download zzzlazy/aic-vintern-ocr   --unzip -p data/ext_ocr
# shots: HF tanp21/aic-hcmc-2025-videos -> annotations/shot_json/

python scripts/verify_shots.py --shots DIR --map-keyframes DIR --media-info DIR
python scripts/build_frames_manifest.py --map-keyframes DIR --shots DIR \
    --media-info DIR --objects DIR --asr-csv data/asr_ext \
    --out-frames data/frames.parquet --out-clips data/clips.parquet \
    --out-videos data/video_bounds.parquet
python scripts/join_ocr.py --ocr data/ocr_raw/ocr
python scripts/join_captions.py --captions data/ext_ocr/ocr
python scripts/build_eval_set.py --limit 300
```

Nhanh hơn nhiều: kéo thẳng `r2:aicc26/artifacts/` về là xong.
