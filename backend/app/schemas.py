import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import AnnotationStatus, JobStatus, ReviewStatus, RunStage, Source

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
    is_mixed: bool = False  # the auto-managed folder for images showing several classes; never a label


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
    is_mixed: bool = False
    uploaded: int = 0
    scraped: int = 0
    generated: int = 0
    imported: int = 0
    pending: int = 0
    rejected: int = 0
    boxes: int = 0  # boxes drawn with this class (on any image); a class that has any can't be removed


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
    # SerpAPI `ijn` is 0-based; the UI shows it 1-based. Each page is one SerpAPI search.
    start_page: int = Field(0, ge=0, le=99)
    max_pages: int = Field(3, ge=1, le=20)  # caps how many searches a job can spend
    min_side: int = Field(200, ge=32, le=4000)  # skip images whose shorter side is smaller, in pixels
    near_dup_distance: int = Field(4, ge=0, le=16)  # dHash bits; 0 keeps near-duplicates
    skip_known_urls: bool = True  # don't re-download URLs an earlier job already tried

    @model_validator(mode="after")
    def _within_serpapi_page_range(self):
        if self.start_page + self.max_pages > 100:
            raise ValueError("start_page + max_pages must be at most 100 (SerpAPI allows ijn 0-99)")
        return self


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


class SearchHistory(BaseModel):
    query: str  # normalized
    pages_searched: list[int]  # 0-based, ascending
    next_page: int  # first page after the highest one already searched
    exhausted_at: int | None  # a page that came back empty: the query has no results beyond it


class SerpApiAccount(BaseModel):
    plan_name: str | None
    searches_per_month: int | None
    this_month_usage: int | None
    plan_searches_left: int | None
    total_searches_left: int | None
    account_rate_limit_per_hour: int | None


class RunPodAccount(BaseModel):
    balance: float | None
    spend_per_hr: float | None
    spend_limit_per_hr: float | None
    pod_count: int


class DatasetCreate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)  # defaults to v1, v2, ...
    val_ratio: float = Field(0.1, ge=0, le=0.5)
    test_ratio: float = Field(0.1, ge=0, le=0.5)
    seed: int = 42
    include_negatives: bool = True  # hard-negative images become background (empty-label) training images
    # Which data to use. None = all of it. Imported datasets are chosen only here, by name: [] means no imported dataset.
    sources: list[Literal["uploaded", "scraped", "generated"]] | None = None
    imported_datasets: list[str] | None = Field(None, max_length=100)

    @field_validator("imported_datasets")
    @classmethod
    def _dataset_names_are_slugs(cls, v):
        if v is not None and any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", n) for n in v):
            raise ValueError("imported dataset names are lowercase letters, digits, '-' or '_'")
        return v

    @model_validator(mode="after")
    def _leaves_room_for_train(self):
        if self.val_ratio + self.test_ratio > 0.6:
            raise ValueError("val_ratio + test_ratio must be at most 0.6")
        return self


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    name: str
    created_at: datetime
    config: dict
    stats: dict


class TrainingRunCreate(BaseModel):
    dataset_id: int
    name: str | None = Field(None, min_length=1, max_length=120)  # defaults to run-<id>
    model: Literal["yolo11n", "yolov8n"] = "yolo11n"  # the only detectors Ultralytics can export for the IMX500
    epochs: int = Field(100, ge=1, le=1000)
    imgsz: int = Field(640, ge=320, le=640, multiple_of=32)  # the IMX500 takes up to 640x640
    batch: int = Field(16, ge=1, le=128)
    export_imx: bool = True  # False: stop after training and keep best.pt only


class TrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    dataset_id: int
    name: str
    config: dict
    stage: RunStage
    progress: dict
    metrics: dict
    artifacts: dict  # name -> {size}; storage keys stay server-side
    error: str | None
    cancel_requested: bool
    worker: str | None
    attempts: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator("artifacts", mode="before")
    @classmethod
    def _hide_keys(cls, v):
        return {name: {"size": a.get("size")} for name, a in (v or {}).items()}


class WorkerOut(BaseModel):
    name: str
    kind: str
    online: bool
    last_seen: datetime
    run_id: int | None


class ClaimIn(BaseModel):
    kind: Literal["gpu", "pi"]


class HeartbeatIn(BaseModel):
    progress: dict | None = None
    stage: Literal["exporting"] | None = None  # the GPU worker moves training -> exporting


class CompleteIn(BaseModel):
    metrics: dict | None = None


class FailIn(BaseModel):
    error: str = Field(max_length=4000)
