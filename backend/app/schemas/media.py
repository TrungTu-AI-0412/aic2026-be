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
