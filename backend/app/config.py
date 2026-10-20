from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MT_")

    database_url: str = "sqlite:///./data/model_trainer.db"

    # "local" stores under local_storage_dir; "s3" uses any S3-compatible store (R2, MinIO, S3).
    storage_backend: str = "local"
    local_storage_dir: str = "./data/storage"
    s3_bucket: str = ""
    s3_endpoint_url: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "auto"

    min_image_side: int = 32


settings = Settings()
