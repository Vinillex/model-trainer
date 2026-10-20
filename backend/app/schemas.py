from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .models import Source

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
