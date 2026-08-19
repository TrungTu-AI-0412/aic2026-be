from enum import Enum

from pydantic import BaseModel, Field


class IngestionEntity(str, Enum):
    FRAMES = "frames"
    CLIPS = "clips"
    ASR_SEGMENTS = "asr_segments"


class IngestionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestionStage(str, Enum):
    VALIDATING = "validating"
    CREATING_COLLECTION = "creating_collection"
    CREATING_PAYLOAD_INDEXES = "creating_payload_indexes"
    UPSERTING = "upserting"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"


class CreateIngestionJobRequest(BaseModel):
    entity: IngestionEntity
    manifest_path: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    feature_profile: str = Field(min_length=1)


class CreateIngestionJobResponse(BaseModel):
    job_id: str
    status: IngestionStatus = IngestionStatus.QUEUED
    collection_name: str


class FeatureProfileOption(BaseModel):
    name: str
    model_id: str
    dimension: int = Field(gt=0)
    clip_frame_count: int = Field(gt=0)
    image_batch_size: int = Field(gt=0)


class IngestionFeatureProfilesResponse(BaseModel):
    profiles: list[FeatureProfileOption]
    default_profile: str


class IngestionProgress(BaseModel):
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    percent: float = Field(ge=0, le=100)


class IngestionJobStatusResponse(BaseModel):
    job_id: str
    status: IngestionStatus
    stage: IngestionStage | None = None
    progress: IngestionProgress | None = None
    collection_name: str
    error: str | None = None


class IngestionJobListResponse(BaseModel):
    jobs: list[IngestionJobStatusResponse]
