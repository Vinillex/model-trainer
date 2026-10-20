from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .models import AnnotationStatus, JobStatus, ReviewStatus, Source

NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


class ClassCreate(BaseModel):
    name: str = Field(pattern=NAME_PATTERN, max_length=64)
    is_negative: bool = False


class ClassOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_negative: bool
    index: int | None


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    # Seed the drone classes and hard-negative categories from the plan.
    seed_defaults: bool = True


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime
    classes: list[ClassOut]


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    class_id: int
    source: Source
    sha256: str
    width: int
    height: int
    group_key: str | None
    review_status: ReviewStatus
    created_at: datetime


class UploadResult(BaseModel):
    added: list[ImageOut]
    duplicates: list[str]
    rejected: list[str]


class StatRow(BaseModel):
    class_id: int
    class_name: str
    is_negative: bool
    uploaded: int = 0
    scraped: int = 0
    generated: int = 0
    pending: int = 0
    rejected: int = 0


class BulkClassCreate(BaseModel):
    names: list[str] = Field(max_length=500)
    is_negative: bool = False


class BulkClassResult(BaseModel):
    added: list[ClassOut]
    skipped: list[str]  # already exist in the project (as target or negative)
    invalid: list[str]


class BoxIn(BaseModel):
    class_id: int
    # Normalized YOLO-style box: center and size as fractions of the image.
    cx: float = Field(ge=0, le=1)
    cy: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class AnnotationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    status: AnnotationStatus
    confidence: float | None


class AnnotationsPut(BaseModel):
    """Replaces every box on an image. review_status=None leaves the image's status unchanged."""

    boxes: list[BoxIn] = Field(max_length=200)
    review_status: ReviewStatus | None = None


class ImageDetail(ImageOut):
    annotations: list[AnnotationOut]


class ScrapeJobCreate(BaseModel):
    class_id: int  # images are filed under this class (a hard-negative category is fine, e.g. birds)
    query: str = Field(min_length=1, max_length=300)
    max_images: int = Field(50, ge=1, le=500)
    # Each page is one SerpAPI search (~100 results). This caps how many searches a job can spend.
    max_pages: int = Field(3, ge=1, le=20)
    min_side: int = Field(200, ge=32, le=4000)  # skip images whose shorter side is smaller, in pixels
    near_dup_distance: int = Field(4, ge=0, le=16)  # dHash bits; 0 keeps near-duplicates


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    type: str
    status: JobStatus
    params: dict
    progress: dict
    error: str | None
    cancel_requested: bool
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
