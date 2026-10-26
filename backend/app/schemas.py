import re
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ai.registry import DETECTOR_PATTERN
from .models import AnnotationStatus, JobStatus, ReviewStatus, RunStage, Source

NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
FOLDER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"  # what clean_folder() produces


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


class ClassRename(BaseModel):
    name: str = Field(pattern=NAME_PATTERN, max_length=64)


class ClassRenameOut(ClassOut):
    moved: int  # images whose files were moved to the folder of the new name


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
    dataset: str | None = None  # which imported dataset the image belongs to
    folder: str | None = None  # the folder an uploaded image was uploaded in
    ai_labelled: bool = False  # boxes drawn by the auto-label job that nobody has confirmed yet
    manually_checked: bool | None = None  # imported images only: opened in the review window by the person
    ai_status: str | None = None  # the AI check's verdict (ok, adjust, suspect, missing, too_small, none), "stale" after an edit, "dismissed" = flagged but the person reviewed those boxes, null = never checked
    sha256: str
    width: int
    height: int
    group_key: str | None
    review_status: ReviewStatus
    created_at: datetime


class ImportedDatasetOut(BaseModel):
    name: str
    total: int
    reviewed: int
    pending: int
    rejected: int
    checked: int  # opened in the review window by the person


class ImageIdsIn(BaseModel):
    image_ids: list[int] = Field(min_length=1, max_length=500)


class TrashDeleteIn(ImageIdsIn):
    allow_breaking_versions: bool = False  # delete even images that an existing dataset version contains (that version can then no longer be downloaded)


class RejectResult(BaseModel):
    rejected: int


class DismissResult(BaseModel):
    dismissed: int  # images whose AI doubts were set aside (the others had none to set aside)


class RestoreResult(BaseModel):
    restored: int


class ImageCount(BaseModel):
    count: int


class PurgeResult(BaseModel):
    deleted: int


class UploadResult(BaseModel):
    added: list[ImageOut]
    duplicates: list[str]
    rejected: list[str]
    folder: str | None = None  # the cleaned folder name the images went into (null for loose files)


class FolderOut(BaseModel):
    """One place images live: a class's source, and inside it an imported dataset or an upload folder (folder null = loose)."""

    class_id: int
    source: Source
    folder: str | None
    total: int
    reviewed: int
    pending: int
    rejected: int
    ai_ok: int = 0  # the AI agrees with the boxes (or found nothing on an unlabelled image)
    ai_flagged: int = 0  # "needs a look": same rule as the gallery filter ai=flagged
    ai_unchecked: int = 0  # the AI has never looked at them


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
    # The person saves these boxes as their decision on the AI's doubts (the review screen sends it with Accept and Reject). Only takes
    # effect where the image has a flagged AI check; the image then leaves "AI: needs a look" for these boxes (see AiDismissal).
    ai_dismiss: bool = False


class ImageDetail(ImageOut):
    annotations: list[AnnotationOut]


class AiCheckOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    image_id: int
    checked_at: datetime
    model: str
    stale: bool
    verdict: str
    dismissed: bool  # a person saved these boxes as reviewed despite the verdict: it no longer asks for a look
    result: dict  # {"labels": [...], "missing": [...], "summary": {...}}, boxes normalized x1,y1,x2,y2


class AiStatus(BaseModel):
    available: bool  # the AI extras are installed
    reason: str | None  # why not, and what to run
    sam_weights: str
    detector: str | None  # which weights find the drones, None until a training run has finished
    device: str | None  # what the loaded models run on (the setting until a check has loaded them)
    fallbacks: int = 0  # GPU calls that failed and were redone on the CPU since the models loaded
    last_fallback: str | None = None


class AiDetectorOut(BaseModel):
    id: str
    label: str
    kind: Literal["run", "file", "prompt"]  # a training run, an uploaded .pt, or the text-prompt detector
    map50: float | None = None  # a training run's validation mAP50
    downloaded: bool | None = None  # the text-prompt detector only: false = its files (about 285 MB) are fetched the first time it is used


class AiSamOut(BaseModel):
    size: Literal["t", "s", "b", "l"]
    file: str
    hint: str
    downloaded: bool  # false = downloaded (free, once) the first time it is used


class AiModelsOut(BaseModel):
    detectors: list[AiDetectorOut]
    default_detector: str | None  # what "default" means right now (a label), None = no detector yet
    sams: list[AiSamOut]
    default_sam: str


class AiObjectPrompt(BaseModel):
    class_id: int
    name: str
    prompt: str  # empty = nothing to look for in this class (no size check)


class AiPromptsOut(BaseModel):
    """What the text-prompt detector is told by default, for the panel to prefill (the person can change every word)."""

    drone_prompt: str
    negative_classes: list[AiObjectPrompt]


class AiPreviewOut(BaseModel):
    count: int  # images the job would process


class AiBoxIn(BaseModel):
    """Click-to-box: a point (normalized) on the thing to box, and/or a rough box (normalized x1,y1,x2,y2) around it."""

    x: float | None = Field(None, ge=0, le=1)
    y: float | None = Field(None, ge=0, le=1)
    box: list[float] | None = Field(None, min_length=4, max_length=4)


class AiBoxOut(BaseModel):
    found: bool
    box: list[float] | None = None  # normalized x1,y1,x2,y2
    score: float | None = None


class AiJobCreate(BaseModel):
    """A batch AI run over the images the filters select: `check` stores the AI's opinion of existing boxes, `autolabel` draws boxes on
    unlabelled pending images (as unconfirmed `auto` boxes), `negatives` looks inside the hard-negative classes (text-prompt detector only)."""

    mode: Literal["check", "autolabel", "negatives"] = "check"
    detector: str | None = Field(None, pattern=DETECTOR_PATTERN)  # "run:5", "file:name.pt" or "yoloe:26s"; None = the default detector
    sam: Literal["t", "s", "b", "l"] | None = None  # SAM 2.1 size; None = the MT_AI_SAM_WEIGHTS setting
    # autolabel: the least sure detection that becomes a box. negatives: the least sure drone hit that is reported. A trained detector is
    # only refused below 0.25 by the API (its own floor, the threshold baked into the IMX500 export); the text-prompt detector scores lower.
    min_score: float | None = Field(None, ge=0.10, le=0.95)  # None = the mode's own default: 0.5 (trained detector), 0.25 (text prompts, which score lower)
    prompt: str | None = Field(None, min_length=1, max_length=80)  # text-prompt detector: the words that find a drone (None = the default)
    object_prompts: dict[str, str] | None = None  # negatives: class id -> words for what that class holds ("bird"); blank = no size check; None = defaults
    min_size: float = Field(0.10, ge=0.02, le=0.5)  # negatives: the smallest the class's thing may be, as a share of the image's short side

    @field_validator("object_prompts")
    @classmethod
    def _object_prompts_are_short_and_keyed_by_class_id(cls, v):
        if v is not None:
            if len(v) > 50 or any(not k.isdigit() or len(w) > 80 for k, w in v.items()):
                raise ValueError("object_prompts: class ids (digits) to words of at most 80 characters")
        return v

    class_id: int | None = None
    source: Source | None = None
    dataset: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    folder: str | None = Field(None, pattern=FOLDER_PATTERN)  # an upload folder
    image_ids: list[int] | None = Field(None, min_length=1, max_length=500)  # exactly these images (an empty list is refused, never "everything")
    review_status: ReviewStatus | None = None
    limit: int = Field(500, ge=1, le=20000)
    force: bool = False  # also redo images whose boxes have not changed since their last check


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
    # What keeps pictures together in one train/val/test split: the website they came from, or the page (article) they were on.
    group_by: Literal["site", "article"] = "site"

    @model_validator(mode="after")
    def _within_serpapi_page_range(self):
        if self.start_page + self.max_pages > 100:
            raise ValueError("start_page + max_pages must be at most 100 (SerpAPI allows ijn 0-99)")
        return self


def check_http_urls(urls: list[str]) -> list[str]:
    out = []
    for u in urls:
        u = u.strip()
        parts = urlsplit(u)
        if len(u) > 2048 or parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"not an http(s) address: {u[:80]!r}")
        out.append(u)
    return list(dict.fromkeys(out))


class CollectJobCreate(BaseModel):
    """Read pictures off web pages (app/collect.py). No SerpAPI search is spent."""

    class_id: int
    urls: list[str] = Field(min_length=1, max_length=50)  # article or gallery pages
    follow_links: bool = False  # also read pages these link to: the same site only, one level deep
    link_contains: list[str] = Field(default_factory=list, max_length=10)  # when following: only links containing one of these words
    include_start_pages: bool = True  # False = use the start pages only to find links (an index page's own pictures are thumbnails)
    max_pages: int = Field(30, ge=1, le=300)  # pages read in total, start pages included
    max_images: int = Field(100, ge=1, le=1000)
    images_per_page: int = Field(12, ge=1, le=50)
    min_side: int = Field(300, ge=32, le=4000)
    near_dup_distance: int = Field(4, ge=0, le=16)
    skip_known_urls: bool = True
    group_by: Literal["site", "article"] = "article"  # pictures of one article stay in one split

    @field_validator("urls")
    @classmethod
    def _http_urls(cls, urls: list[str]) -> list[str]:
        return check_http_urls(urls)

    @field_validator("link_contains")
    @classmethod
    def _words(cls, words: list[str]) -> list[str]:
        return [w.strip()[:100] for w in words if w.strip()]


class VideoJobCreate(BaseModel):
    """Cut still frames out of videos (app/video.py). No SerpAPI search is spent."""

    class_id: int
    urls: list[str] = Field(min_length=1, max_length=20)  # video files, or Wikimedia Commons File: pages of videos
    frames_per_second: float = Field(1.0, gt=0, le=5)
    max_frames_per_video: int = Field(120, ge=1, le=600)  # spread over the whole video
    max_images: int = Field(500, ge=1, le=2000)
    min_side: int = Field(240, ge=32, le=4000)
    near_dup_distance: int = Field(5, ge=0, le=16)  # consecutive frames of a slow scene are near-duplicates and are dropped
    skip_known_urls: bool = True

    @field_validator("urls")
    @classmethod
    def _http_urls(cls, urls: list[str]) -> list[str]:
        return check_http_urls(urls)


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


SIMILAR_PCT_MIN, SIMILAR_PCT_MAX = 0.5, 25.0  # "leave out similar pictures": how far a box may move, % of the picture


class ClassMerge(BaseModel):
    """Classes a dataset version treats as one: its labels use a single class (and index) for all of them."""

    class_ids: list[int] = Field(min_length=2, max_length=10)
    name: str | None = Field(None, pattern=NAME_PATTERN, max_length=64)  # default: the member names joined with "+"


class DatasetCreate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)  # defaults to v1, v2, ...
    val_ratio: float = Field(0.1, ge=0, le=0.5)
    test_ratio: float = Field(0.1, ge=0, le=0.5)
    seed: int = 42
    include_negatives: bool = True  # hard-negative images become background (empty-label) training images
    # Which data to use. None = all of it. Imported datasets are chosen only here, by name: [] means no imported dataset.
    sources: list[Literal["uploaded", "scraped", "generated"]] | None = None
    imported_datasets: list[str] | None = Field(None, max_length=100)
    merge_classes: list[ClassMerge] = Field(default_factory=list, max_length=5)  # the project's own labels are not changed
    skip_similar: bool = False  # leave out repeats: same scene (dHash) with the boxes nearly where they were (similar.py)
    similar_box_pct: float = Field(5.0, ge=SIMILAR_PCT_MIN, le=SIMILAR_PCT_MAX)  # how far a box may move, % of the picture
    group_scenes: bool = False  # keep look-alike scenes (also across imported datasets) in one split: no train/test leak

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


class CloudCopyOut(BaseModel):
    key: str
    size: int
    modified: datetime
    dataset_id: int | None  # the version this is the copy of; None = no version of this project matches (deleted, or from an earlier database)
    dataset_name: str | None
    in_use: bool  # a run on that version is waiting for a worker or training: its pod may still need the zip


class CloudCopiesOut(BaseModel):
    enabled: bool  # the bucket is set up (MT_DATASET_BUCKET*)
    copies: list[CloudCopyOut]
    total_bytes: int


class CloudDeleteIn(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=200)


class CloudDeleteOut(BaseModel):
    deleted: int
    freed_bytes: int


class TrainingRunCreate(BaseModel):
    dataset_id: int
    name: str | None = Field(None, min_length=1, max_length=120)  # defaults to run-<id>
    model: Literal["yolo11n", "yolov8n"] = "yolo11n"  # the only detectors Ultralytics can export for the IMX500
    epochs: int = Field(100, ge=1, le=1000)
    imgsz: int = Field(640, ge=320, le=640, multiple_of=32)  # the IMX500 takes up to 640x640
    batch: int = Field(16, ge=1, le=128)
    export_imx: bool = True  # False: stop after training and keep best.pt only
    # Where the IMX export runs: on this Mac after the pod has gone (the default), or on the pod as before. Runs created
    # before this setting existed have no key and export on the pod (training.exports_on_mac).
    export_where: Literal["mac", "pod"] = "mac"


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


class DisplayStatusIn(BaseModel):
    """What the Pi agent reports about the live view on the Pi's screen."""
    request_id: str = Field(max_length=64)
    state: Literal["starting", "running", "stopped", "failed"]
    message: str = Field("", max_length=500)
    camera_fps: float | None = None
    results_per_s: float | None = None
    detections: int | None = None


class ClaimIn(BaseModel):
    kind: Literal["gpu", "pi"]


class HeartbeatIn(BaseModel):
    progress: dict | None = None
    stage: Literal["exporting"] | None = None  # the GPU worker moves training -> exporting


class CompleteIn(BaseModel):
    metrics: dict | None = None


class FailIn(BaseModel):
    error: str = Field(max_length=4000)
