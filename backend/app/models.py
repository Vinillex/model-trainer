import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Source(str, enum.Enum):
    uploaded = "uploaded"
    scraped = "scraped"
    generated = "generated"


class AnnotationStatus(str, enum.Enum):
    auto = "auto"  # model-proposed, awaiting review
    auto_exact = "auto_exact"  # boxes known by construction (composites)
    reviewed = "reviewed"
    rejected = "rejected"


class ReviewStatus(str, enum.Enum):
    pending = "pending"
    reviewed = "reviewed"
    rejected = "rejected"  # image unusable; excluded from dataset exports


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[str | None] = mapped_column(String(120), default=None)

    classes: Mapped[list["DetClass"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="DetClass.id"
    )


class DetClass(Base):
    """A detection class, or a hard-negative category (is_negative) that has images but no boxes."""

    __tablename__ = "classes"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(64))
    is_negative: Mapped[bool] = mapped_column(Boolean, default=False)
    # Position in the exported YOLO label file; only meaningful for non-negative classes.
    index: Mapped[int | None] = mapped_column(Integer, default=None)

    project: Mapped[Project] = relationship(back_populates="classes")


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (UniqueConstraint("project_id", "sha256"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("classes.id"), index=True)
    source: Mapped[Source] = mapped_column(Enum(Source))
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    # Negative-category images need no boxes, so they start out reviewed.
    review_status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus), default=ReviewStatus.pending, server_default=ReviewStatus.pending.name, index=True
    )
    # Images from the same video/scene share a group so splits don't leak near-duplicates.
    group_key: Mapped[str | None] = mapped_column(String(200), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[str | None] = mapped_column(String(120), default=None)

    annotations: Mapped[list["Annotation"]] = relationship(back_populates="image", cascade="all, delete-orphan")
    hash: Mapped["ImageHash | None"] = relationship(cascade="all, delete-orphan", uselist=False)
    scrape_record: Mapped["ScrapeRecord | None"] = relationship(cascade="all, delete-orphan", uselist=False)


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("classes.id"))
    # Normalized YOLO-style box: center x/y and width/height in [0, 1].
    cx: Mapped[float] = mapped_column(Float)
    cy: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    status: Mapped[AnnotationStatus] = mapped_column(Enum(AnnotationStatus), default=AnnotationStatus.reviewed)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)

    image: Mapped[Image] = relationship(back_populates="annotations")


class ImageHash(Base):
    """64-bit difference hash, for spotting near-duplicates (re-encoded / resized copies of one photo)."""

    __tablename__ = "image_hashes"

    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), primary_key=True)
    dhash: Mapped[str] = mapped_column(String(16), index=True)  # 16 hex chars


class ScrapeRecord(Base):
    """Where a scraped image came from, kept for attribution and licence review."""

    __tablename__ = "scrape_records"

    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), default=None)
    query: Mapped[str] = mapped_column(String(300))
    original_url: Mapped[str] = mapped_column(String(2048))
    page_url: Mapped[str | None] = mapped_column(String(2048), default=None)
    title: Mapped[str | None] = mapped_column(String(500), default=None)


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class Job(Base):
    """A unit of background work. `type` selects the handler; params/progress are handler-defined."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    type: Mapped[str] = mapped_column(String(32))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued, index=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class UrlOutcome(str, enum.Enum):
    added = "added"
    duplicate = "duplicate"  # same bytes as an image already in the project
    near_duplicate = "near_duplicate"
    rejected = "rejected"  # not decodable / unsupported / below the minimum size once downloaded
    download_failed = "download_failed"  # retried by later jobs: failures are often transient


class ScrapedUrl(Base):
    """Every image URL a scrape job has tried, whatever happened. Later jobs skip these without downloading.

    Deliberately NOT removed when the image is deleted: a deleted image is usually one you didn't want.
    """

    __tablename__ = "scraped_urls"
    __table_args__ = (UniqueConstraint("project_id", "url_sha"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    url_sha: Mapped[str] = mapped_column(String(64))  # of the normalized URL; keeps the unique index small
    url: Mapped[str] = mapped_column(Text)
    outcome: Mapped[UrlOutcome] = mapped_column(Enum(UrlOutcome))
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SearchLog(Base):
    """One row per search-result page fetched, so we know which pages of a query were already spent."""

    __tablename__ = "search_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    query: Mapped[str] = mapped_column(String(300), index=True)  # normalized: lowercase, single spaces
    page: Mapped[int] = mapped_column(Integer)  # SerpAPI `ijn`, 0-based
    results: Mapped[int] = mapped_column(Integer)  # 0 means the query ran out of results
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
