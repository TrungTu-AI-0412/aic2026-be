# Tải dataset AIC 2026 từ R2

Hướng dẫn cho thành viên trong đội lấy dataset về máy. Dùng được cho
**Windows**, **macOS** và **Linux**.

Dataset đã được đội mirror lên Cloudflare R2 nên tải nhanh và ổn định hơn
nguồn gốc. Kéo về bao nhiêu lần cũng không phát sinh chi phí.

## Cần chuẩn bị

Xin người quản lý bucket ba giá trị sau:

```
ACCESS_KEY_ID       = ...
SECRET_ACCESS_KEY   = ...
ACCOUNT_ID          = ...
```

Đây là credential **chỉ đọc**, không xoá được gì.

## Dung lượng cần trống

| Bạn cần gì | Zip | Sau giải nén | Nên có trống |
| --- | ---: | ---: | ---: |
| Chỉ metadata | 0.76 GiB | ~1.9 GiB | 5 GiB |
| Metadata + keyframes (đủ để ingest frames) | 29.5 GiB | ~29.5 GiB | **70 GiB** |
| Đầy đủ (thêm videos) | 106.7 GiB | ~106.7 GiB | **230 GiB** |

Xoá zip ngay sau khi giải nén thì cần khoảng một nửa con số cuối.

---

## Bước 1 — Cài rclone

### Windows (PowerShell)

```powershell
winget install Rclone.Rclone
```

Không có `winget` thì dùng scoop:

```powershell
scoop install rclone
```

Hoặc tải thủ công tại <https://rclone.org/downloads/>, giải nén, thêm thư mục
chứa `rclone.exe` vào `PATH`.

Kiểm tra: mở **PowerShell mới** rồi chạy `rclone version`.

### macOS

```bash
brew install rclone
```

### Linux

```bash
curl https://rclone.org/install.sh | sudo bash
```

---

## Bước 2 — Cấu hình

### Windows (PowerShell)

Thay ba giá trị `<...>` trước khi chạy:

```powershell
$dir = "$env:APPDATA\rclone"
New-Item -ItemType Directory -Force $dir | Out-Null

@'
[r2]
type = s3
provider = Cloudflare
access_key_id = <ACCESS_KEY_ID>
secret_access_key = <SECRET_ACCESS_KEY>
endpoint = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
region = auto
acl = private
'@ | Out-File -FilePath "$dir\rclone.conf" -Encoding ascii -Append
```

> Dùng `-Encoding ascii`. Nếu chọn `utf8`, PowerShell 5.1 sẽ chèn BOM vào đầu
> file và rclone có thể không đọc được section đầu tiên.

Không chắc file config nằm ở đâu thì chạy `rclone config file`.

### macOS / Linux

```bash
mkdir -p ~/.config/rclone
cat >> ~/.config/rclone/rclone.conf <<'EOF'
[r2]
type = s3
provider = Cloudflare
access_key_id = <ACCESS_KEY_ID>
secret_access_key = <SECRET_ACCESS_KEY>
endpoint = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
region = auto
acl = private
EOF
```

---

## Bước 3 — Kiểm tra kết nối

```bash
rclone ls r2:aicc26/raw/meta/ --s3-no-check-bucket
```

Phải hiện 4 dòng. Ra được là cấu hình đúng.

> **`rclone lsd r2:` sẽ báo lỗi 403 — đây là bình thường.** Credential chỉ có
> quyền trên bucket `aicc26`, không được phép liệt kê toàn bộ tài khoản. Luôn
> chỉ đích danh `r2:aicc26/...`.

---

## Bước 4 — Tải

**Bắt buộc có `--s3-no-check-bucket` ở mọi lệnh.** Thiếu cờ này là lỗi 403.

Windows thay `./data/zips/` bằng `D:\aic2026\zips\`.

```bash
# 1. Metadata — 773 MB. Tải cái này TRƯỚC, nó nhỏ và cần nhất.
rclone copy r2:aicc26/raw/meta/ ./data/zips/ \
  --s3-no-check-bucket --transfers 4 --progress

# 2. Keyframes — 28.7 GiB. Đủ để ingest frames.
rclone copy r2:aicc26/raw/keyframes/ ./data/zips/ \
  --s3-no-check-bucket --transfers 4 --progress

# 3. Videos — 77.3 GiB. Chỉ cần nếu làm shot detection hoặc clip embedding.
rclone copy r2:aicc26/raw/videos/ ./data/zips/ \
  --s3-no-check-bucket --transfers 4 --progress
```

Đứt mạng thì chạy lại **đúng lệnh cũ** — rclone tự bỏ qua phần đã xong.

Muốn tải tất cả trong một lệnh:

```bash
rclone copy r2:aicc26/raw/ ./data/zips/ \
  --s3-no-check-bucket --transfers 4 --progress
```

---

## Bước 5 — Đối chiếu

```bash
rclone size r2:aicc26/raw/ --s3-no-check-bucket
```

Kết quả đúng phải là:

```
Total objects: 32
Total size: 106.735 GiB (114605488279 Byte)
```

Đối chiếu với những gì đã tải về:

**macOS / Linux**
```bash
du -sh ./data/zips/ && ls ./data/zips/ | wc -l
```

**Windows (PowerShell)**
```powershell
$f = Get-ChildItem D:\aic2026\zips
"{0} file, {1:N2} GiB" -f $f.Count, (($f | Measure-Object Length -Sum).Sum / 1GB)
```

---

## Bước 6 — Giải nén

### Windows

`Expand-Archive` của PowerShell rất chậm với file lớn. Dùng 7-Zip:

```powershell
winget install 7zip.7zip

cd D:\aic2026\zips
Get-ChildItem *.zip | ForEach-Object {
    & 'C:\Program Files\7-Zip\7z.exe' x $_.FullName -oD:\aic2026\data -y
}
```

### macOS / Linux

```bash
mkdir -p ./data/extracted
cd ./data/zips
for f in *.zip; do unzip -q -o "$f" -d ../extracted; done
```

Thiếu chỗ thì giải nén rồi xoá từng file:

```bash
for f in *.zip; do unzip -q -o "$f" -d ../extracted && rm -f "$f"; done
```

### Cấu trúc sau khi giải nén

```
data/extracted/
├── keyframes/          L23_V001/020.jpg, ...
├── video/              L23_V001.mp4, ...          ← "video", KHÔNG phải "videos"
├── map-keyframes/      L23_V001.csv, ...
├── media-info/         L21_V001.json, ...
├── objects/            L26_V361/155.json, ...
└── clip-features-32/   L21_V001.npy, ...
```

Pipeline mong đợi thư mục tên `videos/`, còn zip giải nén ra `video/`. Đổi tên:

```bash
mv data/extracted/video data/extracted/videos          # macOS/Linux
```
```powershell
Rename-Item D:\aic2026\data\video videos               # Windows
```

---

## Điều quan trọng nhất cần biết về dữ liệu

**Tên file keyframe KHÔNG phải số frame trong video.**

File `keyframes/L23_V001/020.jpg` mang số **`n` = 20**, tức "keyframe thứ 20
của video này". Đó **không** phải `original_frame_id`.

Ánh xạ nằm trong `map-keyframes/L23_V001.csv`:

```csv
n,pts_time,fps,frame_idx
1,0.0,25.0,0
2,5.4,25.0,135
3,10.8,25.0,270
```

| Cột | Ý nghĩa |
| --- | --- |
| `n` | số thứ tự keyframe → **tên file** (`020.jpg`) |
| `frame_idx` | **`original_frame_id` thật** — số này mới đi lên bài nộp |
| `pts_time` | mốc thời gian (giây) trong video |
| `fps` | frame rate, dùng quy đổi thời gian ↔ frame |

Nộp nhầm `n` thay vì `frame_idx` là **sai toàn bộ bài**. Mọi bước xây manifest
phải join qua CSV này.

File trong `objects/` cũng đánh số theo `n`, khớp với tên file keyframe.

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách sửa |
| --- | --- | --- |
| `403 AccessDenied` khi `rclone lsd r2:` | Bình thường — token chỉ scope một bucket | Dùng `rclone ls r2:aicc26/...` |
| `403` ở mọi lệnh | Thiếu `--s3-no-check-bucket` | Thêm cờ vào |
| `didn't find section in config file` | File config có BOM (Windows) | Ghi lại bằng `-Encoding ascii` |
| `rclone: command not found` | PATH chưa cập nhật | Mở terminal/PowerShell mới |
| Tải chậm | Ít luồng | Tăng `--transfers 8` |
| Tải đứt giữa chừng | Mạng | Chạy lại đúng lệnh cũ, tự resume |
| Tổng khác 106.735 GiB | Thiếu file | Chạy lại lệnh `copy`, rclone tự bù |

---

## Xem thêm

- [Ingestion](ingestion.md) — biến dữ liệu này thành Qdrant collection
- [Fresh-machine setup runbook](runbook-ubuntu.md) — dựng toàn bộ backend
- [Qdrant operations](qdrant-operations.md) — bàn giao snapshot giữa các máy

---

> Credential được gửi riêng cho từng người, không đặt trong file này và không
> commit vào repo.
