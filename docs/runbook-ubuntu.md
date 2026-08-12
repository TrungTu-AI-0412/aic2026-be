# Runbook triển khai AIC 2026 Retrieval Backend trên Ubuntu

Tài liệu này mô tả cách chạy repository backend từ đầu, bao gồm:

- cài môi trường Python và GPU;
- khởi động Qdrant bằng Docker Compose;
- chạy FastAPI;
- tạo Parquet manifests từ video;
- ingest keyframes và clips vào Qdrant;
- tìm kiếm thử;
- xuất và restore snapshot để máy khác không phải embed lại;
- chạy API lâu dài bằng systemd.

Phạm vi của tài liệu là repository retrieval backend này, không bao gồm
competition console/frontend.

## 1. Kiến trúc khi chạy

```text
Source videos / keyframes
          |
          v
Probe -> shot detection -> keyframe sampling -> Parquet manifests
                                                   |
                                                   v
FastAPI ingestion endpoint -> detached runner -> embedding model
                                                   |
                                                   v
                                                Qdrant
                                                   ^
                                                   |
Text query -> cùng embedding profile -> cosine search -> ranking -> API response
```

Hai loại collection được xây riêng:

- `frames`: mỗi point đại diện cho một keyframe;
- `clips`: mỗi point đại diện cho một shot/clip và được tạo từ tối đa tám
  frame lấy mẫu trong shot.

Hiện tại retrieval path tìm kiếm trên collection `frames`. Collection
`clips` đã có thể ingest và snapshot nhưng chưa được fusion vào search path.

## 2. Yêu cầu máy chủ

Khuyến nghị:

- Ubuntu 22.04 hoặc 24.04;
- Python 3.12;
- Docker Engine và Docker Compose v2;
- NVIDIA driver hoạt động nếu chạy ingestion bằng GPU;
- đủ dung lượng cho source video, extracted keyframes, Hugging Face cache,
  Qdrant storage và snapshot.

Code hiện tại sử dụng cú pháp generic của Python 3.12, vì vậy không nên chạy
bằng Python 3.10 hoặc 3.11.

Kiểm tra các công cụ:

```bash
python3.12 --version
docker --version
docker compose version
nvidia-smi
```

Nếu máy không có GPU, API vẫn có thể chạy trên CPU nhưng ingestion và encode
query bằng SigLIP sẽ chậm đáng kể. Cài Docker theo
[hướng dẫn Docker Engine cho Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
và cài PyTorch CUDA theo
[PyTorch Start Locally](https://pytorch.org/get-started/locally/) để khớp với
NVIDIA driver của server.

## 3. Chuẩn bị repository

Các ví dụ bên dưới giả sử repository nằm đúng tại `/opt/aic2026`. Nếu đặt ở
vị trí khác, phải sửa toàn bộ absolute path tương ứng trong `.env`.

```bash
sudo mkdir -p /opt/aic2026
sudo chown -R "$(id -u):$(id -g)" /opt/aic2026

# Chạy lệnh clone thật của repository vào đúng thư mục này.
git clone <BACKEND_REPOSITORY_URL> /opt/aic2026
cd /opt/aic2026
```

Tạo virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Nếu dùng GPU, cài CUDA-enabled PyTorch theo command được PyTorch selector sinh
ra trước. Sau đó cài dependency của project:

```bash
python -m pip install -r requirements.txt
```

Kiểm tra runtime:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import av, pyarrow, qdrant_client, transformers; print('backend dependencies: OK')"
```

Nếu `torch.cuda.is_available()` trả về `False` trên máy có NVIDIA GPU, chưa
nên ingest dataset lớn. Kiểm tra lại NVIDIA driver và wheel PyTorch đã cài.

## 4. Cấu hình môi trường

Từ repository root:

```bash
cd /opt/aic2026
cp .env.example .env
openssl rand -hex 32
```

Copy chuỗi ngẫu nhiên vừa sinh và thay giá trị
`replace-with-a-random-secret` trong `.env`:

```dotenv
QDRANT_BIND_ADDRESS=127.0.0.1
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=<RANDOM_SECRET>

INGESTION_DATA_ROOT=/opt/aic2026/data
INGESTION_DB_PATH=/opt/aic2026/data/ingestion.db

FEATURE_PROFILE=siglip2-so400m-patch14-384-v1
QDRANT_FRAMES_COLLECTION=aic2026-frames-siglip2-so400m-v1
QDRANT_CLIPS_COLLECTION=aic2026-clips-siglip2-so400m-v1
```

Các nguyên tắc quan trọng:

- Không commit `.env`.
- `INGESTION_DATA_ROOT` phải là absolute path và mọi manifest được gửi vào
  ingestion API phải nằm bên trong thư mục này.
- `FEATURE_PROFILE` phải giống chính xác profile dùng khi ingest collection.
- Mỗi lần thay dataset hoặc model phải dùng tên collection versioned mới.
- Không đổi `QDRANT_BIND_ADDRESS` thành `0.0.0.0` nếu chưa có firewall, TLS và
  lý do rõ ràng để expose Qdrant.

Export `.env` vào shell hiện tại. Snapshot tool và các Python subprocess đọc
biến môi trường của process; chúng không tự source shell file:

```bash
set -a
. /opt/aic2026/.env
set +a
```

## 5. Khởi động Qdrant

Compose đang pin Qdrant `v1.12.1`, bật API key, chỉ publish trên loopback và
persist dữ liệu xuống `data/qdrant/`.

```bash
cd /opt/aic2026
mkdir -p data/qdrant/storage data/qdrant/snapshots
docker compose up -d qdrant
docker compose ps
```

Kiểm tra Qdrant:

```bash
curl --fail \
  -H "api-key: ${QDRANT_API_KEY}" \
  http://127.0.0.1:6333/readyz

curl --fail \
  -H "api-key: ${QDRANT_API_KEY}" \
  http://127.0.0.1:6333/collections
```

Xem log:

```bash
docker compose logs -f qdrant
```

Dữ liệu Qdrant không mất khi restart container vì hai thư mục sau được mount:

```text
/opt/aic2026/data/qdrant/storage
/opt/aic2026/data/qdrant/snapshots
```

Không dùng `docker compose down -v` hoặc xóa các thư mục trên khi chưa có
snapshot/backup.

## 6. Cache embedding model

Profile mặc định tiết kiệm VRAM hơn là:

```text
siglip2-so400m-patch14-384-v1
model: google/siglip2-so400m-patch14-384
dimension: 1152
```

Profile ưu tiên accuracy nhưng cần nhiều VRAM hơn:

```text
siglip2-giant-opt-patch16-384-v1
model: google/siglip2-giant-opt-patch16-384
dimension: 1536
```

Lần ingest hoặc search đầu tiên, Transformers sẽ tải model từ Hugging Face.
Nên cache model trước thay vì đợi một job dài tải giữa chừng:

```bash
cd /opt/aic2026
source venv/bin/activate

python - <<'PY'
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-so400m-patch14-384"
AutoProcessor.from_pretrained(model_id)
AutoModel.from_pretrained(model_id)
print(f"cached: {model_id}")
PY
```

Nếu dùng Giant, đổi `model_id` cho đúng. Khi server phải chạy offline, cần
backup/copy Hugging Face cache sang server nhận. Snapshot Qdrant không chứa
model dùng để encode text query.

## 7. Chạy FastAPI bằng tay

Đây là cách chạy để thử trước khi tạo systemd service.

```bash
cd /opt/aic2026
source venv/bin/activate
set -a
. ./.env
set +a
cd backend

python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

Phải dùng `backend/` làm working directory. API tạo ingestion runner bằng
`python -m app.ingestion.runner`; working directory này giúp subprocess import
package `app` đúng cách.

Chỉ dùng một Uvicorn worker khi model chạy trên GPU. Nhiều worker sẽ load nhiều
bản model và có thể làm đầy VRAM.

Từ terminal khác:

```bash
curl --fail http://127.0.0.1:8000/api/v1/health/live
curl --fail http://127.0.0.1:8000/api/v1/health/ready
```

OpenAPI UI:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
http://127.0.0.1:8000/openapi.json
```

`/health/ready` hiện chỉ xác nhận application container đã khởi tạo. Nó chưa
kiểm tra model, Qdrant hay active collection. Một search request thành công
mới là end-to-end readiness check.

## 8. Hai cách có dữ liệu trong Qdrant

Có hai luồng triển khai độc lập:

1. Restore snapshot đã embed sẵn: nhanh nhất cho máy nhận code/dataset release.
2. Tự build manifests và ingest: dùng khi dataset/model thay đổi hoặc cần audit.

Nếu đã có snapshot, chuyển thẳng đến mục 9. Nếu phải ingest mới, dùng mục 10.

## 9. Restore snapshot đã embed sẵn

Copy cả release directory gồm snapshot frames, snapshot clips và
`snapshot-manifest.json` vào server. Sau khi Qdrant chạy:

```bash
cd /opt/aic2026
source venv/bin/activate
set -a
. ./.env
set +a

python scripts/qdrant_snapshot.py restore \
  --manifest /path/to/release-001/snapshot-manifest.json
```

Restore tool sẽ:

- kiểm tra SHA-256 của từng file;
- kiểm tra Qdrant cùng minor version và target patch không thấp hơn source;
- từ chối overwrite collection đang tồn tại;
- restore vectors, payloads, payload indexes và collection configuration.

Nếu muốn restore thành tên version mới:

```bash
python scripts/qdrant_snapshot.py restore \
  --manifest /path/to/release-001/snapshot-manifest.json \
  --map aic2026-frames-siglip2-so400m-v1=aic2026-frames-release-002 \
  --map aic2026-clips-siglip2-so400m-v1=aic2026-clips-release-002
```

Sau restore, sửa ba giá trị trong `.env` để trỏ đúng release vừa restore:

```dotenv
FEATURE_PROFILE=<profile trong snapshot-manifest.json>
QDRANT_FRAMES_COLLECTION=<restored frames collection>
QDRANT_CLIPS_COLLECTION=<restored clips collection>
```

Sau đó restart API. Người nhận không cần embed lại media, nhưng vẫn cần:

- đúng model/profile để encode text query;
- source video/keyframe nếu chức năng hiển thị media cần đọc payload `path`;
- Parquet manifests nếu muốn audit hoặc rebuild.

Collection snapshot không bao gồm Qdrant aliases, vì vậy app chọn collection
qua `.env`.

## 10. Ingest mới từ source video

### 10.1 Chuẩn bị layout dữ liệu

Ví dụ:

```text
/opt/aic2026/data/
├── videos/
│   ├── L01_V001.mp4
│   └── L01_V002.mp4
├── keyframes/
└── manifests/
```

Tên file video trở thành `video_id`. Với dữ liệu AIC, giữ convention
`Lxx_Vxxx`, ví dụ `L01_V001.mp4`.

Tất cả các lệnh preprocessing dưới đây chạy trong `backend/`:

```bash
cd /opt/aic2026
source venv/bin/activate
set -a
. ./.env
set +a
cd backend
```

### 10.2 Probe video

Probe codec, resolution, rotation, exact FPS và khả năng decode:

```bash
python -m app.ingestion.video.probe \
  --source /opt/aic2026/data/videos \
  --out /opt/aic2026/data/manifests/videos.parquet
```

Pipeline hiện yêu cầu constant frame rate. Video VFR sẽ bị từ chối ở các bước
sau vì không thể ánh xạ chính xác `original_frame_id` bằng một FPS duy nhất.

### 10.3 Detect shots và tạo clips manifest

Accuracy-first detector:

```bash
python -m app.ingestion.video.shot_detect \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet \
  --out /opt/aic2026/data/manifests/clips.parquet \
  --detector transnetv2
```

Fallback nhẹ hơn, không dùng TransNetV2:

```bash
python -m app.ingestion.video.shot_detect \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet \
  --out /opt/aic2026/data/manifests/clips.parquet \
  --detector content
```

`clips.parquet` chứa inclusive `start_frame`/`end_frame`, timestamps, video
path và `shot_id`. Nó chính là manifest dùng để ingest entity `clips`.

### 10.4 Extract keyframes và tạo frames manifest

```bash
python -m app.ingestion.video.sampling \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet \
  --shots-manifest /opt/aic2026/data/manifests/clips.parquet \
  --output-dir /opt/aic2026/data/keyframes \
  --out /opt/aic2026/data/manifests/frames.parquet
```

Mặc định sampler lấy khoảng một keyframe mỗi giây của shot, tránh biên shot
và chọn frame sắc nét nhất trong cửa sổ nhỏ. `frames.parquet` lưu
`original_frame_id`, không đánh lại index.

### 10.5 Gửi ingestion jobs

API phải đang chạy theo mục 7 hoặc 12.

Frames collection:

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/api/v1/ingestions \
  -H 'Content-Type: application/json' \
  -d '{
    "entity": "frames",
    "manifest_path": "/opt/aic2026/data/manifests/frames.parquet",
    "collection_name": "aic2026-frames-siglip2-so400m-v1",
    "feature_profile": "siglip2-so400m-patch14-384-v1"
  }'
```

Clips collection:

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/api/v1/ingestions \
  -H 'Content-Type: application/json' \
  -d '{
    "entity": "clips",
    "manifest_path": "/opt/aic2026/data/manifests/clips.parquet",
    "collection_name": "aic2026-clips-siglip2-so400m-v1",
    "feature_profile": "siglip2-so400m-patch14-384-v1"
  }'
```

Mỗi request trả `job_id`. Theo dõi toàn bộ jobs:

```bash
curl --fail http://127.0.0.1:8000/api/v1/ingestions
```

Theo dõi một job:

```bash
curl --fail \
  http://127.0.0.1:8000/api/v1/ingestions/<JOB_ID>
```

Job thành công khi có:

```json
{
  "status": "succeeded",
  "stage": "completed",
  "error": null
}
```

Ingestion tạo collection mới, payload indexes, upsert embedding theo batch rồi
đợi Qdrant optimize về trạng thái green. Không restart API hoặc server trong
lúc chưa chắc detached ingestion runner đã hoàn thành.

### 10.6 Activate collection

Sau khi cả hai jobs thành công, cập nhật `.env`:

```dotenv
FEATURE_PROFILE=siglip2-so400m-patch14-384-v1
QDRANT_FRAMES_COLLECTION=aic2026-frames-siglip2-so400m-v1
QDRANT_CLIPS_COLLECTION=aic2026-clips-siglip2-so400m-v1
```

Restart API để nạp configuration mới. Ingestion không tự đổi active
collection.

## 11. Kiểm tra search end-to-end

KIS search:

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/api/v1/search/kis \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "kis",
    "description": "một người đang đi xe đạp trên đường phố",
    "top_k": 10
  }'
```

Response phải có:

- `results` với `video_id`, `frame_ids`, `score`;
- `versions.frames_collection` đúng collection active;
- `versions.model_config_name` đúng feature profile;
- `latency_ms` có thời gian encode/query/rerank.

QA endpoint hiện tìm frame liên quan nhưng chưa có VQA model, vì vậy trường
`answer` vẫn là `null`. TRAKE tìm một chuỗi frame tăng dần theo thứ tự events.

Các endpoint media `/api/v1/videos/...` và submission export hiện chưa có
service implementation trong runtime container. Không dùng chúng làm tiêu chí
đánh giá deployment đã sẵn sàng.

## 12. Chạy API lâu dài bằng systemd

Sau khi chạy tay thành công, tạo service:

```bash
sudo editor /etc/systemd/system/aic2026-api.service
```

Nội dung dưới đây giả sử user Linux chạy app là `ubuntu`. Thay `User` và
`Group` cho đúng máy:

```ini
[Unit]
Description=AIC 2026 Retrieval Backend
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/aic2026/backend
EnvironmentFile=/opt/aic2026/.env
ExecStart=/opt/aic2026/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aic2026-api
sudo systemctl status aic2026-api
```

Xem log:

```bash
journalctl -u aic2026-api -f
```

Sau mỗi lần sửa `.env` hoặc deploy code mới:

```bash
sudo systemctl restart aic2026-api
```

FastAPI cũng bind loopback. Expose API qua reverse proxy có TLS hoặc SSH
tunnel, không expose trực tiếp development server ra internet.

## 13. Tạo snapshot để bàn giao release

Chỉ tạo snapshot sau khi ingestion job thành công và collection đã optimize:

```bash
cd /opt/aic2026
source venv/bin/activate
set -a
. ./.env
set +a

python scripts/qdrant_snapshot.py create \
  --collection aic2026-frames-siglip2-so400m-v1 \
  --collection aic2026-clips-siglip2-so400m-v1 \
  --feature-profile siglip2-so400m-patch14-384-v1 \
  --output-dir artifacts/qdrant-snapshots/release-001
```

Bàn giao toàn bộ `release-001/`, không chỉ một file snapshot. Manifest đi kèm
là nơi ghi profile và checksum của cả release.

Không copy trực tiếp live directory `data/qdrant/storage` sang máy khác. Dùng
Qdrant collection snapshot để có artifact nhất quán.

## 14. Validation trước khi bàn giao

Chạy test từ repository root:

```bash
cd /opt/aic2026
source venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest backend/tests -q
```

Checklist release:

- Qdrant `/readyz` trả thành công.
- Frames và clips ingestion jobs đều `succeeded`.
- Qdrant collections đúng point count dự kiến.
- `.env` trỏ đúng versioned collection và feature profile.
- KIS search trả kết quả và versions đúng.
- Snapshot manifest và tất cả `.snapshot` đã được chuyển cùng nhau.
- Hugging Face model cache đã có nếu máy nhận chạy offline.
- Source media được chuyển riêng nếu UI cần xem frame/video.
- Parquet manifests được giữ lại để audit/rebuild.

## 15. Troubleshooting

### Qdrant không khởi động

```bash
docker compose ps
docker compose logs --tail=200 qdrant
sudo ss -ltnp | grep -E '6333|6334'
```

Kiểm tra `.env` có `QDRANT_API_KEY` không rỗng và port chưa bị process khác
chiếm.

### API chạy nhưng search lỗi collection not found

Kiểm tra collection thực tế:

```bash
curl --fail \
  -H "api-key: ${QDRANT_API_KEY}" \
  http://127.0.0.1:6333/collections
```

Sau đó đối chiếu `QDRANT_FRAMES_COLLECTION` trong `.env` và restart API.

### Ingestion trả HTTP 400

`manifest_path` phải nằm trong `INGESTION_DATA_ROOT`. Dùng absolute path như:

```text
/opt/aic2026/data/manifests/frames.parquet
```

### Job chuyển sang failed

Đọc trường `error`:

```bash
curl --fail http://127.0.0.1:8000/api/v1/ingestions/<JOB_ID>
```

Các nguyên nhân thường gặp:

- model chưa tải được hoặc server không có internet/cache;
- CUDA out of memory;
- đường dẫn ảnh/video trong Parquet không tồn tại trên server;
- manifest thiếu column hoặc dùng sai entity;
- collection Qdrant cùng tên đã tồn tại;
- source video là variable frame rate.

Nếu thiếu VRAM, tạo collection mới bằng
`siglip2-so400m-patch14-384-v1`. Không đổi profile giữa collection đang ingest
và API query.

### Runner không import được package app

API phải có:

```text
WorkingDirectory=/opt/aic2026/backend
```

Khi chạy tay, phải `cd /opt/aic2026/backend` trước khi chạy Uvicorn.

### Restore snapshot bị version mismatch

Chạy source và target bằng Qdrant cùng minor version; target patch phải bằng
hoặc mới hơn source patch. Cách an toàn nhất là dùng cùng
`docker-compose.yml`. Không dùng `--allow-version-mismatch` trừ khi đã đọc và
xác minh compatibility từ Qdrant.

## 16. Tài liệu liên quan

- [Ingestion architecture](ingestion.md)
- [Qdrant deployment and snapshot hand-off](qdrant-operations.md)
- [Qdrant snapshot documentation](https://qdrant.tech/documentation/snapshots/)
- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
