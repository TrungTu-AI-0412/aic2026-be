from pydantic import BaseModel, Field, model_validator

MAX_CLIP_FRAMES = 300


class ClipRequest(BaseModel):
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    center_frame: int | None = Field(default=None, ge=0)
    radius: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ClipRequest":
        has_range = self.start_frame is not None and self.end_frame is not None
        has_center = self.center_frame is not None and self.radius is not None

        if has_range == has_center:
            raise ValueError(
                "Provide either (start_frame, end_frame) or "
                "(center_frame, radius), but not both."
            )

        if has_range and self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")

        start, end = self._frame_range(has_range)
        if end - start + 1 > MAX_CLIP_FRAMES:
            raise ValueError(f"clip length exceeds maximum of {MAX_CLIP_FRAMES} frames")

        return self

    def _frame_range(self, has_range: bool) -> tuple[int, int]:
        if has_range:
            return self.start_frame, self.end_frame
        return max(self.center_frame - self.radius, 0), self.center_frame + self.radius

    @property
    def frame_range(self) -> tuple[int, int]:
        has_range = self.start_frame is not None and self.end_frame is not None
        return self._frame_range(has_range)


class NeighbourFrame(BaseModel):
    """A keyframe next to the one being verified, ordered by frame index."""

    frame_id: int
    keyframe_n: int
    pts_sec: float
    shot_id: int
    is_same_shot: bool


class FrameContext(BaseModel):
    """Everything the verify panel shows about one retrieved keyframe.

    `frame_id` is `original_frame_id` — what a submission reports. It is not a
    key: two keyframes of the same video can share one, so `keyframe_n` is the
    identity and the first match wins here.
    """

    video_id: str
    frame_id: int
    keyframe_n: int
    pts_sec: float
    shot_id: int
    shot_start_sec: float
    shot_end_sec: float
    shot_start_frame: int
    shot_end_frame: int
    fps: float
    duration_sec: float
    width: int
    height: int
    neighbours: list[NeighbourFrame]


class TimelineKeyframe(BaseModel):
    """One sampled frame exposed on the full-video verification timeline."""

    frame_id: int
    keyframe_n: int
    pts_sec: float
    shot_id: int
    shot_start_sec: float
    shot_end_sec: float


class VideoTimeline(BaseModel):
    """Source-video metadata plus every sampled frame available for scanning."""

    video_id: str
    fps_num: int
    fps_den: int
    frame_count: int | None
    duration_sec: float
    width: int
    height: int
    rotation: int
    is_vfr: bool
    codec: str
    keyframes: list[TimelineKeyframe]
