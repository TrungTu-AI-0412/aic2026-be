# AIC 2026 Retrieval Backend

Backend tìm kiếm video cho AIC 2026: chuẩn bị dữ liệu video, tạo embedding cục bộ,
lưu vector trong Qdrant và cung cấp API cho các bài KIS, Q&A và TRAKE.

Hệ thống được thiết kế cho một đội thi chạy trên LAN, không phải dịch vụ đa người
dùng. Query path không phụ thuộc cloud; model và dữ liệu cần được cache/copy vào
máy trước khi thi.

## Bắt đầu nhanh

Để dựng một máy hoàn toàn mới, dùng [Fresh-machine setup runbook](docs/runbook-ubuntu.md).
Phần dưới đây dành cho máy đã có Python 3.12 và Docker Compose.

```bash
cp .env.example .env
# Thay QDRANT_API_KEY và các absolute path trong .env.

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/qdrant/storage data/qdrant/snapshots
docker compose up -d qdrant

set -a
. ./.env
set +a
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

PowerShell tương đương:

```powershell
Copy-Item .env.example .env
py -3.12 -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

New-Item -ItemType Directory -Force data\qdrant\storage, data\qdrant\snapshots
docker compose up -d qdrant

Set-Location backend
Get-Content ..\.env |
  Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
  ForEach-Object {
    $name, $value = $_ -split '=', 2
    Set-Item -Path "Env:$name" -Value $value
  }
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Kiểm tra:

```text
GET http://127.0.0.1:8000/api/v1/health/live
GET http://127.0.0.1:8000/api/v1/health/ready
API docs: http://127.0.0.1:8000/docs
```

> `ready` chỉ xác nhận application container đã khởi tạo. Search chỉ hoạt động
> sau khi active frame collection tồn tại trong Qdrant và `FEATURE_PROFILE`
> khớp profile đã dùng để tạo collection.

## Thành phần hệ thống

```text
Video/keyframe
    │
    ├─ probe → shot detection → sampling → Parquet manifests
    │                                         │
    │                                         ▼
    └──────────────────────────── ingestion runner → embedding model
                                                      │
                                                      ▼
Text query → cùng feature profile ────────────────→ Qdrant
                                                      │
                                                      ▼
                                      dedupe/rank → SearchResponse → frontend
```

- **FastAPI** cung cấp health, search, media, ingestion và submission API.
- **Qdrant** lưu vector, payload và payload index. Đây là nguồn metadata filter.
- **Parquet manifests** là nguồn rebuild/audit; snapshot chỉ là đường deploy nhanh.
- **SQLite** lưu trạng thái ingestion job để runner độc lập với vòng đời API.
- **Transformers/PyTorch** chạy image/text encoder trên máy cục bộ.

## Bản đồ repository

```text
aic2026-be/
├── backend/                 Python application và tests
│   ├── app/
│   │   ├── api/             FastAPI router, dependency wiring, HTTP endpoints
│   │   ├── core/            Settings và application lifespan
│   │   ├── features/        Profile registry, model loading, media embedding
│   │   ├── ingestion/       Manifest, job service, runner và preprocessing
│   │   │   └── video/       Probe, shot detection, keyframe sampling
│   │   ├── ranking/         Shot-level dedupe/ranking
│   │   ├── retrieval/       Shared retrieval engine và KIS/QA/TRAKE orchestration
│   │   ├── runtime/         Composition root/container của application
│   │   ├── schemas/         Pydantic API contracts
│   │   ├── services/        Protocols/domain boundaries
│   │   ├── stubs/           Test/development doubles còn được giữ riêng
│   │   └── vector_store/    Toàn bộ code phụ thuộc Qdrant
│   └── tests/
│       ├── unit/            Tests không yêu cầu Qdrant thật
│       └── integration/     Tests tạo collection trên Qdrant thật
├── data/                    Runtime data; bị ignore, không commit dataset/vector
├── docs/                    Kiến trúc, ingestion, OpenAPI và runbook
├── scripts/                 Công cụ vận hành độc lập, hiện có snapshot Qdrant
├── docker-compose.yml       Qdrant v1.12.1 với persistent bind mounts
├── requirements.txt        Production dependencies
├── requirements-dev.txt    Test dependencies
└── .env.example            Mẫu runtime/Compose configuration
```

### Các thư mục backend quan trọng

| Thư mục | Trách nhiệm | Khi cần sửa |
| --- | --- | --- |
| `backend/app/api/endpoints/` | Chuyển HTTP request/response và domain error | Thêm/sửa endpoint; không đặt retrieval logic ở đây |
| `backend/app/schemas/` | Contract được xuất sang OpenAPI | Khi request/response thay đổi; phải regenerate `docs/openapi.json` |
| `backend/app/services/` | Protocol giữa API và implementation | Khi thêm capability mà không muốn route phụ thuộc implementation |
| `backend/app/runtime/` | Chọn implementation thực và config active collection | Khi wire service mới |
| `backend/app/retrieval/` | Encode query, query Qdrant, ghép kết quả từng track | Khi thay retrieval behavior dùng chung hoặc KIS/QA/TRAKE orchestration |
| `backend/app/ranking/` | Dedupe theo shot và giữ top rank | Khi thay scoring/dedupe; luôn thêm test |
| `backend/app/vector_store/` | Collection, payload index, search, upsert | Mọi thay đổi Qdrant-specific |
| `backend/app/features/` | Registry model và inference image/text | Khi thêm feature profile/model mới |
| `backend/app/ingestion/` | Job lifecycle và build collection versioned | Khi thay ingestion pipeline |
| `backend/app/ingestion/video/` | Chuyển source video thành manifests/keyframes | Khi thay probe/detector/sampler |
| `backend/tests/unit/` | Contract và algorithm tests | Chạy trước mọi handoff |
| `backend/tests/integration/` | Xác minh Qdrant mapping/filter/upsert thật | Chạy khi Qdrant local đã sẵn sàng |

## Dữ liệu runtime

Layout khuyến nghị dưới `INGESTION_DATA_ROOT`:

```text
data/
├── videos/                  Source videos, ví dụ L01_V001.mp4
├── keyframes/               JPEG do sampler tạo, chia theo video
├── manifests/
│   ├── videos.parquet       Metadata video/FPS/path
│   ├── clips.parquet        Shot ranges
│   └── frames.parquet       Keyframes với original_frame_id
├── qdrant/
│   ├── storage/             Live Qdrant storage
│   └── snapshots/           Qdrant server snapshot staging
└── ingestion.db             Job status database
```

Không commit các file trên. Không xóa `data/qdrant/storage` hoặc chạy thao tác
xóa volume khi chưa có snapshot hợp lệ.

## Cấu hình

Các biến chính trong `.env`:

| Biến | Ý nghĩa |
| --- | --- |
| `QDRANT_URL` | URL backend dùng để kết nối Qdrant |
| `QDRANT_API_KEY` | API key dùng chung cho Compose, API và snapshot tool |
| `QDRANT_BIND_ADDRESS` | Interface publish Qdrant; mặc định chỉ loopback |
| `INGESTION_DATA_ROOT` | Root tuyệt đối chứa manifest và media được phép đọc |
| `INGESTION_DB_PATH` | SQLite job database, nên dùng absolute path |
| `FEATURE_PROFILE` | Text encoder của query path; phải khớp active collection |
| `QDRANT_FRAMES_COLLECTION` | Frame collection đang phục vụ search |
| `QDRANT_CLIPS_COLLECTION` | Clip collection cùng release, hiện chưa fusion vào ranking |
| `QDRANT_BATCH_SIZE` | Kích thước upsert batch |

Feature profiles đăng ký tại `backend/app/features/profiles.py`:

| Profile | Model | Dimension | Mục đích |
| --- | --- | ---: | --- |
| `siglip2-giant-opt-patch16-384-v1` | `google/siglip2-giant-opt-patch16-384` | 1536 | Accuracy-first, VRAM cao |
| `siglip2-so400m-patch14-384-v1` | `google/siglip2-so400m-patch14-384` | 1152 | Mặc định, cân bằng memory/quality |
| `clip-b32-v1` | `openai/clip-vit-base-patch32` | 512 | Compatibility và thử nghiệm nhẹ |

Không được đổi profile cho một collection đã tạo. Data/model mới phải tạo tên
collection versioned mới, đánh giá xong mới đổi `.env` và restart API.

## API chính

Base path: `/api/v1`.

| Endpoint | Trạng thái/chức năng |
| --- | --- |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Application container readiness |
| `POST /search/kis` | Tìm một frame theo description |
| `POST /search/qa` | Retrieval frame; `answer` hiện vẫn do operator nhập |
| `POST /search/trake` | Ghép chuỗi event có frame tăng dần trong cùng video |
| `GET /videos/{video_id}/frames/{frame_id}` | Đọc sampled JPEG keyframe |
| `GET /videos/{video_id}/clip` | Decode tối đa 300 frame thành MP4 |
| `GET /ingestions/feature-profiles` | Model registry cho ingestion UI |
| `POST /ingestions` | Tạo detached ingestion job |
| `GET /ingestions` | Liệt kê jobs |
| `GET /ingestions/{job_id}` | Theo dõi stage/progress/error |
| `POST /submissions/export` | Contract có sẵn nhưng runtime chưa wire implementation |

Backend OpenAPI là source of truth tại `docs/openapi.json`. Swagger UI khi API
chạy nằm tại `/docs`.

## Ingestion và collection lifecycle

1. Copy video/media vào `INGESTION_DATA_ROOT`.
2. Tạo `videos.parquet`, `clips.parquet`, `frames.parquet`.
3. Gửi frame/clip ingestion jobs với **tên collection mới**.
4. Runner validate manifest, tạo collection/index, embed, upsert và optimize.
5. Đánh giá collection ngoài competition query path.
6. Đổi active collection/profile trong `.env` và restart API.
7. Tạo snapshot để bàn giao máy khác mà không embed lại.

Chi tiết command và recovery: [Ingestion](docs/ingestion.md),
[Qdrant operations](docs/qdrant-operations.md) và
[Fresh-machine setup runbook](docs/runbook-ubuntu.md).

## Tests và validation

Từ repository root:

```bash
source venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest backend/tests/unit -q
```

Chạy toàn bộ unit tests trên Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\backend;$PWD"
.\venv\Scripts\python.exe -m pytest backend\tests\unit -q
```

Integration tests cần Qdrant thật và dùng collection tạm:

```bash
python -m pytest backend/tests/integration -q
```

## OpenAPI và đồng bộ frontend

Khi schema/endpoint thay đổi:

```bash
cd backend
python -c "import json; from pathlib import Path; from app.main import create_app; Path('../docs/openapi.json').write_text(json.dumps(create_app().openapi(), indent=2) + '\n', encoding='utf-8')"
cd ..
cp docs/openapi.json ../aic2026-fe/openapi/openapi.json
cd ../aic2026-fe
npm run codegen
```

Backend và frontend là hai repository độc lập; không gộp thay đổi của hai repo
vào một commit.

## Tài liệu liên quan

- [Fresh-machine setup and operations](docs/runbook-ubuntu.md)
- [Ingestion design](docs/ingestion.md)
- [Qdrant snapshot hand-off](docs/qdrant-operations.md)
- [Generated OpenAPI 3.1 contract](docs/openapi.json)
