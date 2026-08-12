# Runbook thiết lập máy mới và vận hành AIC 2026

Tài liệu này hướng dẫn dựng retrieval backend từ một máy mới, không chỉ Ubuntu
server. Tên file `runbook-ubuntu.md` được giữ để không làm hỏng các liên kết cũ.

Hai môi trường được hỗ trợ trong runbook:

- **Windows 11 workstation**: development, preprocessing, ingestion và demo.
- **Ubuntu 24.04/22.04 server**: deployment ổn định, chạy lâu dài bằng systemd.

Các phần dùng chung bao gồm Qdrant, model cache, restore snapshot, build manifest,
ingestion, activate collection, search validation và troubleshooting.

## 1. Kết quả cuối cùng

Sau khi hoàn tất, máy có:

```text
Qdrant v1.12.1       http://127.0.0.1:6333
FastAPI backend      http://127.0.0.1:8000
Swagger UI           http://127.0.0.1:8000/docs
Optional frontend    http://127.0.0.1:5173
```

Luồng dữ liệu:

```text
Source videos
    │
    ├─ probe → shot detection → keyframe sampling → Parquet manifests
    │                                                   │
    │                                                   ▼
    └────────────────────────────────────── detached ingestion runner
                                                        │
                                              image/clip embedding
                                                        │
                                                        ▼
Text query → matching text encoder ─────────────────→ Qdrant
                                                        │
                                                        ▼
                                              dedupe/rank → API
```

Hai collection được tạo độc lập:

- `frames`: một point cho mỗi sampled keyframe;
- `clips`: một point cho mỗi shot, pooling tối đa tám sampled frames.

Query path hiện dùng frame collection. Clip collection có thể ingest/snapshot
nhưng chưa được fusion vào search ranking.

## 2. Chọn đường triển khai

Có hai đường để máy có collection:

1. **Restore snapshot** — nhanh nhất, không embed lại; dùng cho máy thi/máy nhận release.
2. **Ingest từ source** — dùng khi dataset, sampling hoặc model thay đổi.

Nếu đã có release gồm `.snapshot` và `snapshot-manifest.json`, setup Qdrant,
Python và model cache rồi chuyển tới [mục 10](#10-restore-snapshot-không-embed-lại).

Nếu cần build mới, làm tiếp [mục 11](#11-build-manifest-và-ingest-từ-source).

## 3. Yêu cầu phần cứng và phần mềm

### Bắt buộc

- CPU x86-64 hoặc ARM64 được Docker/PyTorch hỗ trợ;
- Python **3.12**;
- Git;
- Docker Engine + Compose v2, hoặc Docker Desktop;
- đủ disk cho video, keyframes, model cache, Qdrant storage và snapshot.

Code dùng cú pháp Python 3.12, không hỗ trợ Python 3.10/3.11.

### GPU

NVIDIA GPU được khuyến nghị mạnh cho ingestion và query encoding bằng SigLIP.
Backend Python chạy trực tiếp trên host; chỉ Qdrant chạy container. Vì vậy:

- cần NVIDIA driver và CUDA-compatible PyTorch wheel;
- **không cần NVIDIA Container Toolkit** cho Qdrant;
- CPU vẫn chạy được nhưng ingestion và request đầu tiên có thể rất chậm.

Kiểm tra:

```text
Python 3.12.x
Docker + docker compose
NVIDIA driver/nvidia-smi nếu dùng GPU
```

### Dung lượng cần dự trù

Không có một con số cố định. Tính riêng:

- source video;
- extracted JPEG keyframes;
- Hugging Face cache (SigLIP Giant lớn hơn nhiều So400m/CLIP);
- live Qdrant storage;
- snapshot release;
- Parquet manifests và SQLite job DB.

Không đặt live Qdrant storage trên ổ tạm hoặc thư mục tự dọn.

## 4. Cài công cụ trên máy mới

### 4.1 Windows 11 workstation

1. Cài [Git for Windows](https://git-scm.com/download/win).
2. Cài Python 3.12 từ [python.org](https://www.python.org/downloads/). Bật
   Python Launcher (`py`) trong installer.
3. Cài [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
   với WSL 2 backend và Linux containers.
4. Nếu dùng frontend, cài Node.js `^20.19.0` hoặc `>=22.12.0`.
5. Nếu dùng GPU, cài NVIDIA driver và chọn command PyTorch phù hợp tại
   [PyTorch Start Locally](https://pytorch.org/get-started/locally/).

Mở PowerShell mới và kiểm tra:

```powershell
git --version
py -3.12 --version
docker --version
docker compose version
nvidia-smi # bỏ qua nếu không có NVIDIA GPU
node --version # chỉ cần cho frontend
npm --version  # chỉ cần cho frontend
```

Nếu PowerShell không cho activate venv, chỉ mở quyền cho process hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Không cần thay execution policy toàn máy.

### 4.2 Ubuntu server

Ubuntu 24.04 được khuyến nghị vì có Python 3.12 trong repository mặc định.
Ubuntu 22.04 vẫn dùng được nhưng cần cài Python 3.12 qua nguồn tin cậy/pyenv.

Ubuntu 24.04:

```bash
sudo apt update
sudo apt install -y \
  git curl ca-certificates build-essential \
  python3.12 python3.12-venv python3.12-dev
```

Cài Docker Engine và Compose plugin từ repository chính thức theo
[Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
Các package cuối cùng cần có:

```bash
sudo apt install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

Để chạy Docker không cần `sudo`:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker version
docker compose version
```

Quyền trong group `docker` tương đương quyền quản trị container trên host; chỉ
thêm user vận hành tin cậy.

Nếu dùng GPU, cài NVIDIA driver trên host rồi kiểm tra:

```bash
nvidia-smi
```

Không cài CUDA toolkit hệ thống một cách ngẫu nhiên để sửa lỗi PyTorch. Chọn
wheel tương thích từ PyTorch selector và xác minh bằng `torch.cuda.is_available()`.

## 5. Clone và tạo Python environment

### Windows

Ví dụ workspace `D:\AIC2026`:

```powershell
New-Item -ItemType Directory -Force D:\AIC2026
Set-Location D:\AIC2026
git clone <BACKEND_REPOSITORY_URL> aic2026-be
Set-Location aic2026-be

py -3.12 -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Ubuntu

Ví dụ deployment root `/opt/aic2026`:

```bash
sudo mkdir -p /opt/aic2026
sudo chown -R "$(id -u):$(id -g)" /opt/aic2026
git clone <BACKEND_REPOSITORY_URL> /opt/aic2026
cd /opt/aic2026

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Cài dependencies

Nếu dùng GPU, cài CUDA-enabled PyTorch theo command do PyTorch selector cung
cấp **trước**, sau đó cài project dependencies:

```bash
python -m pip install -r requirements.txt
```

Cho development/tests:

```bash
python -m pip install -r requirements-dev.txt
```

Xác minh runtime:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import av, pyarrow, qdrant_client, transformers; print('backend dependencies: OK')"
```

Nếu máy có NVIDIA nhưng CUDA là `False`, chưa ingest dataset lớn. Kiểm tra driver
và PyTorch wheel trước.

## 6. Tạo cấu hình `.env`

Từ repository root:

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
```

PowerShell:

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy secret vừa tạo vào `QDRANT_API_KEY`.

### Ubuntu example

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

CLIP_FUSION_WEIGHT=0.5
RERANK_ENABLED=true
RERANK_TOP_N=30
RERANK_MODEL=Salesforce/blip-itm-large-coco
```

Cross-encoder rerank phải có sẵn weights trong local Hugging Face cache trước
khi thi — query path không được phép ra mạng:

```bash
python -c "
from transformers import AutoProcessor, BlipForImageTextRetrieval
AutoProcessor.from_pretrained('Salesforce/blip-itm-large-coco')
BlipForImageTextRetrieval.from_pretrained('Salesforce/blip-itm-large-coco')
"
```

### Windows example

Dùng forward slash để path dễ đọc và không phải escape:

```dotenv
QDRANT_BIND_ADDRESS=127.0.0.1
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=<RANDOM_SECRET>

INGESTION_DATA_ROOT=D:/AIC2026/aic2026-be/data
INGESTION_DB_PATH=D:/AIC2026/aic2026-be/data/ingestion.db

FEATURE_PROFILE=siglip2-so400m-patch14-384-v1
QDRANT_FRAMES_COLLECTION=aic2026-frames-siglip2-so400m-v1
QDRANT_CLIPS_COLLECTION=aic2026-clips-siglip2-so400m-v1
```

Quy tắc:

- không commit `.env`;
- dùng absolute path cho ingestion trên server;
- mọi manifest gửi vào API phải nằm dưới `INGESTION_DATA_ROOT`;
- `FEATURE_PROFILE` phải khớp collection active;
- dataset/model mới phải dùng collection name versioned mới;
- giữ Qdrant trên loopback trừ khi có private network/firewall/TLS rõ ràng.

## 7. Khởi động Qdrant

Compose pin Qdrant `v1.12.1`, yêu cầu API key và persist vào bind mounts.

Linux:

```bash
mkdir -p data/qdrant/storage data/qdrant/snapshots
docker compose up -d qdrant
docker compose ps
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force data\qdrant\storage, data\qdrant\snapshots
docker compose up -d qdrant
docker compose ps
```

Load `.env` vào Linux shell và kiểm tra:

```bash
set -a
. ./.env
set +a

curl --fail -H "api-key: ${QDRANT_API_KEY}" http://127.0.0.1:6333/readyz
curl --fail -H "api-key: ${QDRANT_API_KEY}" http://127.0.0.1:6333/collections
```

PowerShell:

```powershell
$QdrantKey = (Get-Content .env | Where-Object { $_ -match '^QDRANT_API_KEY=' }) -replace '^QDRANT_API_KEY=', ''
curl.exe --fail -H "api-key: $QdrantKey" http://127.0.0.1:6333/readyz
curl.exe --fail -H "api-key: $QdrantKey" http://127.0.0.1:6333/collections
```

Logs:

```bash
docker compose logs -f qdrant
```

Không chạy `docker compose down -v` và không xóa `data/qdrant/storage` khi chưa
có snapshot/backup đã kiểm tra.

## 8. Cache embedding model

Profiles hiện có:

| Profile | Model ID | Dimension | Ghi chú |
| --- | --- | ---: | --- |
| `siglip2-giant-opt-patch16-384-v1` | `google/siglip2-giant-opt-patch16-384` | 1536 | Accuracy-first, VRAM cao |
| `siglip2-so400m-patch14-384-v1` | `google/siglip2-so400m-patch14-384` | 1152 | Mặc định |
| `clip-b32-v1` | `openai/clip-vit-base-patch32` | 512 | Nhẹ/compatibility |

Pre-download model trước ingestion/competition:

```bash
python - <<'PY'
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-so400m-patch14-384"
AutoProcessor.from_pretrained(model_id)
AutoModel.from_pretrained(model_id)
print(f"cached: {model_id}")
PY
```

PowerShell không hỗ trợ Bash heredoc; dùng file tạm hoặc một dòng:

```powershell
python -c "from transformers import AutoModel, AutoProcessor; m='google/siglip2-so400m-patch14-384'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('cached:', m)"
```

Đổi model ID nếu dùng Giant/CLIP. Máy offline cần được copy Hugging Face cache.
Qdrant snapshot không chứa text encoder.

## 9. Chạy FastAPI bằng tay

API và detached runner cần import package `app`, vì vậy working directory phải
là `backend/`.

### Linux

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

### Windows PowerShell

```powershell
Set-Location D:\AIC2026\aic2026-be
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
Set-Location backend

Get-Content ..\.env |
  Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
  ForEach-Object {
    $name, $value = $_ -split '=', 2
    Set-Item -Path "Env:$name" -Value $value
  }

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Dùng một worker khi model chạy GPU; nhiều worker load nhiều bản model và có thể
làm đầy VRAM.

Health/API docs:

```bash
curl --fail http://127.0.0.1:8000/api/v1/health/live
curl --fail http://127.0.0.1:8000/api/v1/health/ready
```

Windows dùng `curl.exe` thay `curl`.

```text
Swagger: http://127.0.0.1:8000/docs
ReDoc:   http://127.0.0.1:8000/redoc
OpenAPI: http://127.0.0.1:8000/openapi.json
```

`ready` không warm model, không query Qdrant và không kiểm tra active collection.
Phải chạy search thật ở mục 13 để xác nhận end-to-end.

## 10. Restore snapshot không embed lại

Snapshot release phải gồm mọi `.snapshot` và `snapshot-manifest.json` trong cùng
thư mục. Sau khi Qdrant chạy và environment variables đã load:

```bash
python scripts/qdrant_snapshot.py restore \
  --manifest /path/to/release-001/snapshot-manifest.json
```

PowerShell:

```powershell
python scripts\qdrant_snapshot.py restore `
  --manifest D:\releases\release-001\snapshot-manifest.json
```

Tool sẽ kiểm tra SHA-256, Qdrant version compatibility và từ chối overwrite
collection đã tồn tại.

Restore dưới tên mới:

```bash
python scripts/qdrant_snapshot.py restore \
  --manifest /path/to/snapshot-manifest.json \
  --map aic2026-frames-siglip2-so400m-v1=aic2026-frames-release-002 \
  --map aic2026-clips-siglip2-so400m-v1=aic2026-clips-release-002
```

Sau restore, update `.env`:

```dotenv
FEATURE_PROFILE=<profile trong snapshot-manifest.json>
QDRANT_FRAMES_COLLECTION=<restored frames collection>
QDRANT_CLIPS_COLLECTION=<restored clips collection>
```

Restart API. Snapshot không chứa:

- Hugging Face model cache;
- source videos hoặc extracted keyframes;
- Parquet manifests;
- Qdrant aliases.

Media endpoint cần source/keyframes đúng path. Parquet manifests cần được giữ
để audit/rebuild.

## 11. Build manifest và ingest từ source

### 11.1 Layout dữ liệu

```text
<INGESTION_DATA_ROOT>/
├── videos/
│   ├── L01_V001.mp4
│   └── L01_V002.mp4
├── keyframes/
└── manifests/
```

Stem tên video trở thành `video_id`; giữ convention `Lxx_Vxxx`.

Mọi command preprocessing chạy từ `backend/` với environment đã activate.

### 11.2 Trial slice trước

Trước dataset lớn, dùng `--limit` để xác minh một vài video:

```bash
python -m app.ingestion.video.probe \
  --source /opt/aic2026/data/videos \
  --out /opt/aic2026/data/manifests/videos-trial.parquet \
  --limit 2
```

Trên Windows thay path bằng `D:/AIC2026/aic2026-be/data/...`.

### 11.3 Probe video

```bash
python -m app.ingestion.video.probe \
  --source /opt/aic2026/data/videos \
  --out /opt/aic2026/data/manifests/videos.parquet \
  --resume
```

Manifest giữ codec, resolution, rotation, exact FPS, frame count và duration.
Pipeline yêu cầu constant frame rate; video VFR bị từ chối để bảo vệ
`original_frame_id`.

### 11.4 Shot detection

Accuracy-first:

```bash
python -m app.ingestion.video.shot_detect \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet \
  --out /opt/aic2026/data/manifests/clips.parquet \
  --detector transnetv2 \
  --resume
```

Content detector nhẹ hơn:

```bash
python -m app.ingestion.video.shot_detect \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet \
  --out /opt/aic2026/data/manifests/clips.parquet \
  --detector content \
  --resume
```

Không đổi threshold trên full dataset trước khi kiểm tra trial output.

### 11.5 Extract keyframes

```bash
python -m app.ingestion.video.sampling \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet \
  --shots-manifest /opt/aic2026/data/manifests/clips.parquet \
  --output-dir /opt/aic2026/data/keyframes \
  --out /opt/aic2026/data/manifests/frames.parquet
```

Sampler giữ `original_frame_id`; không đánh lại index frame.

### 11.6 Import artifacts tạo bên ngoài

Keyframes có sẵn:

```bash
python -m app.ingestion.batch_builder keyframes \
  --source /path/to/keyframes \
  --out /opt/aic2026/data/manifests/frames.parquet \
  --shots-manifest /opt/aic2026/data/manifests/clips.parquet \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet
```

Shot CSV có columns `video_id,start_frame,end_frame`:

```bash
python -m app.ingestion.batch_builder shots \
  --csv /path/to/shots.csv \
  --out /opt/aic2026/data/manifests/clips.parquet \
  --videos-manifest /opt/aic2026/data/manifests/videos.parquet
```

Nếu không có video manifest, truyền FPS chính xác bằng `--fps 25` hoặc
`--fps 30000/1001`; tool không đoán FPS.

### 11.7 Chọn feature profile

```bash
curl --fail http://127.0.0.1:8000/api/v1/ingestions/feature-profiles
```

Frontend Ingestion page dùng endpoint này để render dropdown model.

### 11.8 Tạo ingestion jobs

Frames:

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

Clips:

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

Windows có thể dùng Swagger UI hoặc frontend Ingestion page để tránh escape JSON
trong PowerShell. Manifest path vẫn là path mà **backend process** đọc được.

Theo dõi:

```bash
curl --fail http://127.0.0.1:8000/api/v1/ingestions
curl --fail http://127.0.0.1:8000/api/v1/ingestions/<JOB_ID>
```

Job hoàn tất khi:

```json
{
  "status": "succeeded",
  "stage": "completed",
  "error": null
}
```

Runner là detached subprocess và ghi state vào SQLite. Không shutdown/reboot máy
khi job đang `running`.

### 11.9 Activate collection

Ingestion không tự đổi active collection. Sau khi đánh giá và cả jobs thành công:

```dotenv
FEATURE_PROFILE=siglip2-so400m-patch14-384-v1
QDRANT_FRAMES_COLLECTION=aic2026-frames-siglip2-so400m-v1
QDRANT_CLIPS_COLLECTION=aic2026-clips-siglip2-so400m-v1
```

Restart API để load config mới.

## 12. Chạy lâu dài trên Ubuntu bằng systemd

Chỉ tạo service sau khi chạy tay thành công.

```bash
sudo editor /etc/systemd/system/aic2026-api.service
```

Thay `User`, `Group` và path nếu deployment khác `/opt/aic2026`:

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

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aic2026-api
sudo systemctl status aic2026-api
journalctl -u aic2026-api -f
```

Sau khi thay `.env` hoặc deploy code:

```bash
sudo systemctl restart aic2026-api
```

API bind loopback. Expose qua private reverse proxy/TLS/SSH tunnel; không publish
development Uvicorn trực tiếp ra internet.

## 13. Kiểm tra end-to-end

KIS:

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

Response cần có:

- `results[].video_id`, `frame_ids`, `score`;
- `versions.frames_collection` đúng active collection;
- `versions.model_config_name` đúng profile;
- `latency_ms` có encode/Qdrant/rerank.

Kiểm tra media bằng một result thật:

```text
GET /api/v1/videos/<VIDEO_ID>/frames/<FRAME_ID>
GET /api/v1/videos/<VIDEO_ID>/clip?center_frame=<FRAME_ID>&radius=90
```

Frame chỉ tồn tại nếu đúng sampled keyframe. Clip cần `videos.parquet` và source
video path còn đọc được. Submission export hiện có contract nhưng runtime chưa
wire `SubmissionService`, không dùng nó làm readiness criterion.

QA hiện retrieval frame nhưng không tự sinh answer. TRAKE trả frame sequence tăng
dần theo event order.

## 14. Tạo snapshot release

Chỉ tạo sau khi ingestion thành công và collection green:

```bash
python scripts/qdrant_snapshot.py create \
  --collection aic2026-frames-siglip2-so400m-v1 \
  --collection aic2026-clips-siglip2-so400m-v1 \
  --feature-profile siglip2-so400m-patch14-384-v1 \
  --output-dir artifacts/qdrant-snapshots/release-001
```

Bàn giao toàn bộ `release-001/`, không chỉ file `.snapshot`. Không copy trực tiếp
live `data/qdrant/storage` sang máy khác.

## 15. Setup frontend trên cùng máy hoặc LAN

Frontend nằm ở repository riêng. Development:

```bash
cd /path/to/aic2026-fe
npm install
cp .env.example .env.local
npm run dev
```

Nếu backend ở máy khác:

```dotenv
VITE_API_BASE_URL=/api/v1
VITE_DEV_PROXY_TARGET=http://<BACKEND_LAN_IP>:8000
```

Vite proxy tránh CORS. Không mở thẳng browser sang origin backend khác khi backend
chưa có CORS middleware.

## 16. Validation trước handoff

Backend unit tests từ repository root:

```bash
source venv/bin/activate
python -m pytest backend/tests/unit -q
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\backend;$PWD"
.\venv\Scripts\python.exe -m pytest backend\tests\unit -q
```

Integration tests cần Qdrant:

```bash
python -m pytest backend/tests/integration -q
```

Checklist:

- Qdrant `/readyz` thành công với API key.
- API live/ready thành công.
- Model cache có sẵn; CUDA state đúng kỳ vọng.
- Frame/clip jobs `succeeded` hoặc snapshot restore hoàn tất.
- `.env` trỏ đúng collection/profile version.
- KIS search trả result và diagnostics đúng.
- Media paths tồn tại nếu UI cần frame/clip.
- Snapshot manifest và mọi snapshot được bàn giao cùng nhau.
- Parquet manifests được lưu để audit/rebuild.
- Không commit `.env`, dataset, model cache hoặc live Qdrant storage.

## 17. Troubleshooting

### Qdrant không start

```bash
docker compose ps
docker compose logs --tail=200 qdrant
```

Kiểm tra API key không rỗng, Docker đang chạy và port 6333/6334 chưa bị chiếm.
Trên Windows, xác nhận Docker Desktop đang dùng Linux containers.

### API import lỗi `app`

Chạy Uvicorn từ `backend/`. Với systemd:

```text
WorkingDirectory=/opt/aic2026/backend
```

### API không đọc `.env`

Compose tự đọc `.env` ở repository root. API chạy từ `backend/` nên runbook chủ
động export biến vào process hoặc systemd `EnvironmentFile`. Trên Windows, chạy
PowerShell import block tại mục 9.

### Health ready nhưng search lỗi collection not found

Liệt kê Qdrant collections và đối chiếu `.env`:

```bash
curl --fail -H "api-key: ${QDRANT_API_KEY}" http://127.0.0.1:6333/collections
```

Sửa `QDRANT_FRAMES_COLLECTION`, xác nhận `FEATURE_PROFILE`, rồi restart API.

### Request search đầu tiên rất chậm

Model đang được download/load hoặc CPU đang encode. Kiểm tra:

- backend log;
- Hugging Face cache;
- internet ở lần cache đầu;
- `torch.cuda.is_available()`;
- VRAM bằng `nvidia-smi`.

### Ingestion HTTP 400

Thường do:

- `manifest_path` nằm ngoài `INGESTION_DATA_ROOT`;
- feature profile không có trong registry;
- path Windows được gửi theo máy browser thay vì máy backend.

### Ingestion HTTP 409

Collection name đã được một job không-failed sử dụng. Tạo tên versioned mới;
không overwrite collection active.

### Job failed

Đọc `error`:

```bash
curl --fail http://127.0.0.1:8000/api/v1/ingestions/<JOB_ID>
```

Nguyên nhân thường gặp:

- model chưa cache/không tải được;
- CUDA out of memory;
- media path trong Parquet không tồn tại;
- manifest sai columns/entity;
- Qdrant không truy cập được;
- collection cùng tên đã tồn tại;
- video VFR.

Thiếu VRAM: dùng collection mới với So400m hoặc CLIP. Không đổi profile giữa
chừng trên cùng collection.

### Frame 404

Frame endpoint đọc sampled JPEG, không decode mọi arbitrary frame. Kiểm tra
`data/keyframes/<video_id>/` và frame ID từ search result.

### Clip 404/422

- `videos.parquet` phải nằm tại `data/manifests/videos.parquet`;
- source video path trong manifest phải tồn tại;
- truyền đúng một trong hai cặp `start_frame/end_frame` hoặc `center_frame/radius`;
- clip tối đa 300 frames.

### Restore snapshot version mismatch

Dùng cùng Qdrant minor version; target patch phải bằng hoặc mới hơn source patch.
Cách an toàn nhất là cùng `docker-compose.yml`. Không dùng
`--allow-version-mismatch` nếu chưa xác minh compatibility.

### Windows file sharing/Docker permission

Qdrant bind mounts nằm trong repository. Nếu Docker Desktop báo không mount
được, đặt repo trên ổ được WSL/Docker Desktop chia sẻ và kiểm tra quyền của
`data/qdrant/`.

## 18. Tài liệu liên quan

- [Backend README](../README.md)
- [Ingestion architecture](ingestion.md)
- [Qdrant deployment and snapshot hand-off](qdrant-operations.md)
- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Docker Desktop on Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
