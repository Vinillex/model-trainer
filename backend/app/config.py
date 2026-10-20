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

    # Background jobs run on a worker thread inside the API process.
    run_worker: bool = True
    worker_poll_seconds: float = 1.0

    # Scraping (SerpAPI Google Images). Set MT_SERPAPI_KEY in backend/.env.
    serpapi_key: str | None = None
    serpapi_url: str = "https://serpapi.com/search"
    serpapi_account_url: str = "https://serpapi.com/account.json"  # free to call; doesn't use quota
    scrape_max_bytes: int = 15 * 1024 * 1024
    scrape_download_workers: int = 8
    # Dev/test only: allow fetching images from loopback/private addresses (normally blocked as SSRF).
    scrape_allow_private_hosts: bool = False


settings = Settings()
