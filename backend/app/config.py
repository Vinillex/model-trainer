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
    # Collecting pictures from web pages (app/collect.py): at most one request per host per this many seconds, pages included.
    collect_request_delay: float = 1.0
    collect_page_max_bytes: int = 3 * 1024 * 1024
    video_max_bytes: int = 400 * 1024 * 1024  # a video larger than this is not downloaded (MT_VIDEO_MAX_BYTES)
    collect_max_side: int = 2560  # collected pictures larger than this (longest side, px) are scaled down; the detector sees 640

    # RunPod (GPU pods for training, serverless for auto-labeling). Set MT_RUNPOD_API_KEY in backend/.env.
    runpod_api_key: str | None = None
    runpod_graphql_url: str = "https://api.runpod.io/graphql"
    runpod_rest_url: str = "https://rest.runpod.io/v1"  # pods are started and deleted here

    # Optional private bucket (Cloudflare R2 or any S3-compatible store) holding a copy of each training zip, so pods download
    # it from there instead of through this Mac's slow upload link (app/dataset_bucket.py). All four set = used.
    dataset_bucket: str = ""
    dataset_bucket_endpoint: str | None = None  # R2: https://<account id>.r2.cloudflarestorage.com
    dataset_bucket_access_key: str | None = None
    dataset_bucket_secret_key: str | None = None

    # Remote workers (GPU PC for training/export, Raspberry Pi for .rpk packaging) call /workers/* with this token.
    # Unset = the worker API is disabled. Set MT_WORKER_TOKEN in backend/.env.
    worker_token: str | None = None
    worker_lease_seconds: int = 900  # a worker silent for this long loses its run (heartbeats renew the lease)
    # Setup lines a RunPod pod reports before its agent runs (POST /workers/log), one file per worker. The launcher prints them, so
    # a pod that stalls while installing says where; its own logs are gone once the pod is deleted.
    worker_log_dir: str = "./data/worker_logs"

    # Where the UI's RunPod launches keep their state file and logs (app/api/runpod.py). The launcher runs as its own process, so this is
    # how the page finds it again after the backend restarts.
    runpod_launch_dir: str = "./data/runpod"

    # What the Pi's screen should show (app/pi_display.py): one small file, so it survives a backend restart.
    pi_display_file: str = "./data/pi_display.json"

    # AI check (app/ai/): SAM 2.1 refines boxes, the trained YOLO is the detector. Needs `pip install -r requirements-ai.txt`; the weights live
    # under ai_dir (git-ignored with the rest of data/). Official weights names are downloaded there on first use: sam2.1_t.pt, _s.pt, _b.pt, _l.pt.
    ai_dir: str = "./data/ai"
    ai_sam_weights: str = "sam2.1_s.pt"
    ai_device: str | None = None  # None = the Mac's GPU (mps) when torch has it, else the CPU
    ai_detector_weights: str | None = None  # a .pt path; None = best.pt of the newest finished training run


settings = Settings()
