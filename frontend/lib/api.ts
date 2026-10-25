export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Source = "uploaded" | "scraped" | "generated" | "imported";
export const SOURCES: Source[] = ["uploaded", "scraped", "generated", "imported"];

export interface DetClass {
  id: number;
  name: string;
  is_negative: boolean;
  index: number | null;
  is_mixed: boolean; // the auto-managed folder for reviewed images showing several classes; never a label
}

export interface Project {
  id: number;
  name: string;
  created_at: string;
  classes: DetClass[];
}

export type ReviewStatus = "pending" | "reviewed" | "rejected";

export type AiVerdict = "ok" | "adjust" | "suspect" | "missing" | "too_small" | "none";
export type AiStatusValue = AiVerdict | "stale" | "dismissed";
/** Boxes in the AI check are normalized x1, y1, x2, y2. */
export type AiBox = [number, number, number, number];
export interface AiLabelItem {
  annotation_id: number | null;
  box: AiBox;
  side_px: number; // size on the 640 px network input
  verdict: AiVerdict;
  reason: string;
  suggested: AiBox | null;
  detector_iou: number | null;
  detector_score: number | null;
  sam_object: boolean | null;
}
export interface AiMissingItem {
  box: AiBox;
  detector_box: AiBox;
  score: number;
  class_name: string | null;
  side_px: number;
  reason: string;
}
export interface AiCheck {
  image_id: number;
  checked_at: string;
  model: string;
  stale: boolean;
  verdict: AiVerdict;
  dismissed: boolean; // a person reviewed these boxes despite the verdict: it no longer asks for a look
  result: { labels: AiLabelItem[]; missing: AiMissingItem[]; summary: Record<string, number> };
}
export interface AiServiceStatus {
  available: boolean;
  reason: string | null;
  sam_weights: string;
  detector: string | null;
  device: string | null; // what the loaded models run on (the setting until a check has loaded them)
  fallbacks: number; // GPU calls that failed and were redone on the CPU
  last_fallback: string | null;
}
export interface AiBoxResult {
  found: boolean;
  box: AiBox | null;
  score: number | null;
}
export type AiSamSize = "t" | "s" | "b" | "l";
export interface AiDetector {
  id: string; // "run:5", "file:name.pt" or "yoloe:26s" (the text-prompt detector)
  label: string;
  kind: "run" | "file" | "prompt";
  map50: number | null; // a training run's validation mAP50
  downloaded?: boolean | null; // the text-prompt detector only: false = its files (about 285 MB) are fetched the first time it is used
}
export interface AiPrompts {
  drone_prompt: string; // the words that find a drone by default
  negative_classes: { class_id: number; name: string; prompt: string }[]; // words for what each hard-negative class holds ("" = nothing to look for)
}
export interface AiSam {
  size: AiSamSize;
  file: string;
  hint: string;
  downloaded: boolean; // false = downloaded (free, once) the first time it is used
}
export interface AiModels {
  detectors: AiDetector[];
  default_detector: string | null; // what "default" means right now
  sams: AiSam[];
  default_sam: string;
}
export interface AiJobParams {
  mode?: "check" | "autolabel" | "negatives"; // check = the AI's opinion of existing boxes; autolabel = draw boxes on unlabelled pending images; negatives = look inside the hard-negative classes
  detector?: string;
  sam?: AiSamSize;
  min_score?: number; // autolabel: least sure detection that becomes a box; negatives: least sure drone-like hit that is reported
  prompt?: string; // text-prompt detector: the words that find a drone
  object_prompts?: Record<string, string>; // negatives: class id -> words for what the class holds ("bird"); blank = no size check
  min_size?: number; // negatives: the smallest the class's thing may be, as a share of the picture's short side
  class_id?: number;
  source?: Source;
  dataset?: string;
  folder?: string; // an upload folder
  image_ids?: number[]; // exactly these images (the Images page's selection)
  review_status?: ReviewStatus;
  limit?: number;
  force?: boolean;
}

export interface ImageRow {
  id: number;
  class_id: number;
  source: Source;
  dataset: string | null; // the imported dataset the image came from (null unless source is "imported")
  folder: string | null; // the folder an uploaded image was uploaded in (null for loose uploads and other sources)
  ai_labelled: boolean; // boxes drawn by the auto-label job that nobody has confirmed yet
  manually_checked: boolean | null; // imported images only: has the person opened it in the review window?
  ai_status: AiStatusValue | null; // the AI check's verdict, "stale" after the boxes were edited, "dismissed" = flagged but reviewed by a person, null = never checked
  sha256: string;
  width: number;
  height: number;
  group_key: string | null;
  review_status: ReviewStatus;
  created_at: string;
}

/** One imported dataset and how many of its images are in each review state. */
export interface ImportedDataset {
  name: string;
  total: number;
  reviewed: number;
  pending: number;
  rejected: number;
  checked: number; // opened in the review window
}

/** Normalized YOLO-style box: centre and size as fractions of the image. */
export interface BoxIn {
  class_id: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface Annotation extends BoxIn {
  id: number;
  status: "auto" | "auto_exact" | "reviewed" | "rejected";
  confidence: number | null;
}

export interface ImageDetail extends ImageRow {
  annotations: Annotation[];
}

export interface BulkClassResult {
  added: DetClass[];
  skipped: string[];
  invalid: string[];
}

export interface UploadResult {
  added: ImageRow[];
  duplicates: string[];
  rejected: string[];
  folder: string | null; // the cleaned folder name the images went into
}

/** One place images live: a class's source and, inside it, an imported dataset or an upload folder (null = loose). */
export interface FolderRow {
  class_id: number;
  source: Source;
  folder: string | null;
  total: number;
  reviewed: number;
  pending: number;
  rejected: number;
  ai_ok: number; // the AI agrees with the boxes
  ai_flagged: number; // "needs a look": the same rule as the gallery filter
  ai_unchecked: number; // the AI has never looked at them
}

export interface StatRow {
  class_id: number;
  class_name: string;
  is_negative: boolean;
  is_mixed: boolean;
  uploaded: number;
  scraped: number;
  generated: number;
  imported: number;
  pending: number;
  rejected: number;
  boxes: number; // boxes drawn with this class on any image; a class that has any can't be removed
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface ScrapeParams {
  class_id: number;
  query: string;
  max_images: number;
  max_pages: number;
  min_side: number;
  near_dup_distance: number;
  start_page: number; // 0-based (SerpAPI ijn)
  skip_known_urls: boolean;
  group_by: GroupBy;
}

/** What keeps pictures together in one train/val/test split: their website, or the page (article) they were on. */
export type GroupBy = "site" | "article";

/** A job that reads the pictures off web pages (type "collect_pages"). */
export interface CollectParams {
  class_id: number;
  urls: string[];
  follow_links: boolean;
  link_contains: string[];
  include_start_pages: boolean;
  max_pages: number;
  max_images: number;
  images_per_page: number;
  min_side: number;
  near_dup_distance: number;
  skip_known_urls: boolean;
  group_by: GroupBy;
  query?: string;
}

/** A job that cuts still frames out of videos (type "video_frames"). */
export interface VideoParams {
  class_id: number;
  urls: string[];
  frames_per_second: number;
  max_frames_per_video: number;
  max_images: number;
  min_side: number;
  near_dup_distance: number;
  skip_known_urls: boolean;
  query?: string;
}

export interface SearchHistory {
  query: string;
  pages_searched: number[]; // 0-based
  next_page: number;
  exhausted_at: number | null;
}

export interface SerpApiAccount {
  plan_name: string | null;
  searches_per_month: number | null;
  this_month_usage: number | null;
  plan_searches_left: number | null;
  total_searches_left: number | null;
  account_rate_limit_per_hour: number | null;
}

export interface Job {
  id: number;
  project_id: number;
  type: string;
  status: JobStatus;
  params: Partial<ScrapeParams>;
  progress: Partial<Record<string, number>>;
  error: string | null;
  cancel_requested: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export type SelectableSource = "uploaded" | "scraped" | "generated";
export const SELECTABLE_SOURCES: SelectableSource[] = ["uploaded", "scraped", "generated"];

/** Which data a dataset version uses: the three plain sources, and imported datasets by name (they are chosen one by one). */
export interface DataSelection {
  sources: SelectableSource[];
  imported: string[];
}

export interface DatasetPreview {
  reviewed_images: number; // for the current selection
  reviewed_with_boxes: number;
  reviewed_empty: number;
  negative_images: number;
  pending: number;
  rejected: number;
  reviewed_total: number; // all reviewed images, whatever is selected
  not_selected: number;
  /** Everything that can be chosen, with its number of reviewed images. */
  available: { sources: Record<SelectableSource, number>; imported_datasets: { name: string; reviewed: number }[] };
  boxes_per_class: Record<string, number>;
}

type SplitName = "train" | "val" | "test";

/** A dataset zip in the cloud bucket (Cloudflare R2): what RunPod pods download. */
export interface CloudCopy {
  key: string;
  size: number;
  modified: string;
  dataset_id: number | null; // null: no version of this project matches it any more
  dataset_name: string | null;
  in_use: boolean; // a run on that version is waiting or training, so its copy cannot be deleted yet
}

export interface CloudCopies {
  enabled: boolean; // the bucket is set up
  copies: CloudCopy[];
  total_bytes: number;
}

export interface DatasetVersion {
  id: number;
  project_id: number;
  name: string;
  created_at: string;
  config: {
    val_ratio: number;
    test_ratio: number;
    seed: number;
    include_negatives: boolean;
    sources?: SelectableSource[] | null; // null/absent = all
    imported_datasets?: string[] | null;
    classes: { index: number; name: string }[];
    merged?: { name: string; classes: string[] }[]; // classes this version treats as one (absent when nothing was merged)
  };
  stats: {
    images: Record<SplitName, number>;
    boxes: Record<SplitName, number>;
    background_images: Partial<Record<SplitName, number>>;
    per_class: Record<string, Record<SplitName, { images: number; boxes: number }>>;
    excluded: { pending: number; rejected: number; not_selected?: number };
    by_source?: Record<string, number>; // "scraped", "uploaded", "imported:<dataset>"
    warnings: string[];
  };
}

export type RunStage = "queued" | "training" | "exporting" | "awaiting_package" | "packaging" | "done" | "failed" | "cancelled";

export interface RunMetrics {
  precision?: number | null;
  recall?: number | null;
  map50?: number | null;
  map50_95?: number | null;
}

export interface TrainingRun {
  id: number;
  project_id: number;
  dataset_id: number;
  name: string;
  config: { model: string; epochs: number; imgsz: number; batch: number; export_imx: boolean };
  stage: RunStage;
  progress: RunMetrics & { message?: string; epoch?: number; epochs?: number };
  metrics: RunMetrics & { per_class?: Record<string, { ap50: number | null; ap50_95: number | null }> };
  artifacts: Partial<Record<ArtifactName, { size: number | null }>>;
  error: string | null;
  cancel_requested: boolean;
  worker: string | null;
  attempts: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export type ArtifactName = "best_pt" | "imx_zip" | "labels" | "rpk";

export interface TrainingParams {
  dataset_id: number;
  model: "yolo11n" | "yolov8n";
  epochs: number;
  imgsz: number;
  batch: number;
  export_imx: boolean;
}

export interface WorkerInfo {
  name: string;
  kind: "gpu" | "pi";
  online: boolean;
  last_seen: string;
  run_id: number | null;
}

export interface RunPodPod {
  id: string;
  name: string | null;
  status: string | null;
  cost_per_hr: number;
}

export interface RunPodStatus {
  key_set: boolean;
  token_set: boolean;
  cloudflared: boolean;
  bucket: boolean;
  balance: number | null;
  spend_per_hr: number | null;
  pods: RunPodPod[];
  other_pods: number;
  error: string | null;
}

export type RunPodCloud = "secure" | "community";

/** What starting a run on RunPod would do and cost: every check the launcher makes before it spends anything. */
export interface RunPodPlan {
  run_id: number;
  run_name: string;
  images: number;
  cloud: RunPodCloud;
  max_hours: number;
  gpus: { id: string; price: number | null }[];
  worst_price: number;
  worst_case: number;
  balance: number;
  bucket: boolean;
  export_imx: boolean;
}

/** The latest RunPod launch (a separate process on the Mac): running, or finished with the end of its log. */
export interface RunPodLaunch {
  state: "none" | "running" | "finished";
  run_id?: number;
  started_at?: string;
  max_cost?: number;
  exit_code?: number | null;
  stopping?: boolean;
  lost_track?: boolean; // the backend restarted while it ran, so how it ended is only in the log
  lines: string[];
}

/** The live view of a finished run on the Raspberry Pi's own screen (the Pi agent shows it; this is what it last reported). */
export interface PiDisplay {
  state: "none" | "waiting" | "starting" | "running" | "stopped" | "failed" | "ended";
  pi_online: boolean;
  request_id: string | null;
  run_id: number | null;
  run_name: string | null;
  message: string;
  camera_fps: number | null;
  results_per_s: number | null;
  detections: number | null;
  threshold: number | null;
  fps: number | null;
}

/** The backend stores UTC but its timestamps carry no zone suffix, which `new Date` would read as local time. */
export const parseUtc = (s: string) => new Date(/(Z|[+-]\d\d:?\d\d)$/.test(s) ? s : s + "Z");

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    const err = new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : res.statusText) as Error & { status?: number };
    err.status = res.status; // callers that need to tell "refused, ask the person" (409) from a failure can
    throw err;
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const datasetDownloadUrl = (id: number) => `${API}/datasets/${id}/download`;
export const imageUrl = (id: number) => `${API}/images/${id}/file`;
export const artifactUrl = (runId: number, name: ArtifactName) => `${API}/training-runs/${runId}/artifacts/${name}`;

/** The filters the Images page offers; `listImages` adds the paging, `countImages` takes them as they are. */
type ImageFilterParams = { classId?: number; boxClassId?: number; excludeRejected?: boolean; source?: Source; dataset?: string; folder?: string; manuallyChecked?: boolean; ai?: string; reviewStatus?: ReviewStatus };

export const api = {
  listProjects: () => req<Project[]>("/projects"),
  createProject: (name: string) => req<Project>("/projects", json("POST", { name })),
  getProject: (id: number) => req<Project>(`/projects/${id}`),
  createClass: (projectId: number, name: string, isNegative: boolean) =>
    req<DetClass>(`/projects/${projectId}/classes`, json("POST", { name, is_negative: isNegative })),
  createClasses: (projectId: number, names: string[], isNegative: boolean) =>
    req<BulkClassResult>(`/projects/${projectId}/classes/bulk`, json("POST", { names, is_negative: isNegative })),
  deleteClass: (projectId: number, classId: number) => req<void>(`/projects/${projectId}/classes/${classId}`, { method: "DELETE" }),
  /** New name for a class; its images' files move to the folder of the new name (`moved` of them). */
  renameClass: (projectId: number, classId: number, name: string) =>
    req<DetClass & { moved: number }>(`/projects/${projectId}/classes/${classId}`, json("PATCH", { name })),
  stats: (projectId: number) => req<StatRow[]>(`/projects/${projectId}/stats`),
  listImages: (projectId: number, p: ImageFilterParams & { limit: number; offset: number }) => {
    const q = new URLSearchParams({ limit: String(p.limit), offset: String(p.offset) });
    if (p.classId != null) q.set("class_id", String(p.classId));
    if (p.boxClassId != null) q.set("box_class_id", String(p.boxClassId));
    if (p.source) q.set("source", p.source);
    if (p.dataset) q.set("dataset", p.dataset);
    if (p.folder) q.set("folder", p.folder);
    if (p.manuallyChecked !== undefined) q.set("manually_checked", String(p.manuallyChecked));
    if (p.ai) q.set("ai", p.ai);
    if (p.reviewStatus) q.set("review_status", p.reviewStatus);
    if (p.excludeRejected) q.set("exclude_rejected", "true");
    return req<ImageRow[]>(`/projects/${projectId}/images?${q}`);
  },
  /** How many images `listImages` would list with the same filters. */
  countImages: (projectId: number, p: ImageFilterParams) => {
    const q = new URLSearchParams();
    if (p.classId != null) q.set("class_id", String(p.classId));
    if (p.boxClassId != null) q.set("box_class_id", String(p.boxClassId));
    if (p.source) q.set("source", p.source);
    if (p.dataset) q.set("dataset", p.dataset);
    if (p.folder) q.set("folder", p.folder);
    if (p.manuallyChecked !== undefined) q.set("manually_checked", String(p.manuallyChecked));
    if (p.ai) q.set("ai", p.ai);
    if (p.reviewStatus) q.set("review_status", p.reviewStatus);
    if (p.excludeRejected) q.set("exclude_rejected", "true");
    return req<{ count: number }>(`/projects/${projectId}/images/count?${q}`);
  },
  /** The review window calls this for an imported image it shows: opening it there is what counts as a manual check. */
  markManuallyChecked: (id: number) => req<ImageRow>(`/images/${id}/manual-check`, { method: "POST" }),
  importedDatasets: (projectId: number) => req<ImportedDataset[]>(`/projects/${projectId}/imported-datasets`),
  /** `folder` files the images under <class>/uploaded/<folder>; `paths` (one per file) is where each sat inside it. */
  upload: (projectId: number, classId: number, files: File[], folder?: string, paths?: string[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    paths?.forEach((p) => form.append("relative_paths", p));
    const q = folder ? `?${new URLSearchParams({ folder })}` : "";
    return req<UploadResult>(`/projects/${projectId}/classes/${classId}/images${q}`, { method: "POST", body: form });
  },
  folders: (projectId: number) => req<FolderRow[]>(`/projects/${projectId}/folders`),
  /** Take rejected images out of the trash: they become "not reviewed" again, boxes as they were. */
  restoreImages: (projectId: number, imageIds: number[]) => req<{ restored: number }>(`/projects/${projectId}/images/restore`, json("POST", { image_ids: imageIds })),
  /** Delete rejected images for good. A 409 means a dataset version contains some of them: ask the person, then repeat with allowBreakingVersions. */
  purgeImages: (projectId: number, imageIds: number[], allowBreakingVersions = false) =>
    req<{ deleted: number }>(`/projects/${projectId}/images/delete`, json("POST", { image_ids: imageIds, allow_breaking_versions: allowBreakingVersions })),
  /** Reject several images exactly as the review screen's Reject does for one. There is no bulk accept: accepting means a person looked. */
  rejectImages: (projectId: number, imageIds: number[]) => req<{ rejected: number }>(`/projects/${projectId}/images/reject`, json("POST", { image_ids: imageIds })),
  getImage: (id: number) => req<ImageDetail>(`/images/${id}`),
  /** `aiDismiss`: these boxes are the person's decision on the AI's doubts (Accept / Reject): a flagged image leaves "AI: needs a look". */
  putAnnotations: (id: number, boxes: BoxIn[], reviewStatus?: ReviewStatus, aiDismiss?: boolean) =>
    req<ImageDetail>(`/images/${id}/annotations`, json("PUT", { boxes, review_status: reviewStatus ?? null, ai_dismiss: aiDismiss ?? false })),
  createScrapeJob: (projectId: number, p: Partial<ScrapeParams> & Pick<ScrapeParams, "class_id" | "query">) =>
    req<Job>(`/projects/${projectId}/jobs/scrape`, json("POST", p)),
  createCollectJob: (projectId: number, p: Partial<CollectParams> & Pick<CollectParams, "class_id" | "urls">) =>
    req<Job>(`/projects/${projectId}/jobs/collect-pages`, json("POST", p)),
  createVideoJob: (projectId: number, p: Partial<VideoParams> & Pick<VideoParams, "class_id" | "urls">) =>
    req<Job>(`/projects/${projectId}/jobs/video-frames`, json("POST", p)),
  searchHistory: (projectId: number, query: string) =>
    req<SearchHistory>(`/projects/${projectId}/searches?${new URLSearchParams({ query })}`),
  serpapiAccount: () => req<SerpApiAccount>("/serpapi/account"),
  listJobs: (projectId: number, type?: string) => req<Job[]>(`/projects/${projectId}/jobs?limit=10${type ? `&type=${type}` : ""}`),
  aiStatus: () => req<AiServiceStatus>("/ai/status"),
  /** Check one image now. The first call loads the models, so it can take a minute. Stores a suggestion; never changes boxes. */
  aiCheck: (id: number) => req<AiCheck>(`/images/${id}/ai-check`, { method: "POST" }),
  /** Put a dismissed image back in "AI: needs a look". */
  aiUndismiss: (id: number) => req<AiCheck>(`/images/${id}/ai-dismiss`, { method: "DELETE" }),
  /** The stored check of an image, or null if it has none. */
  getAiCheck: async (id: number) => {
    const res = await fetch(`${API}/images/${id}/ai-check`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText);
    return (await res.json()) as AiCheck;
  },
  /** Click-to-box: SAM's tight box around a click and/or a rough box (all normalized). */
  aiBox: (id: number, p: { x?: number; y?: number; box?: AiBox }) => req<AiBoxResult>(`/images/${id}/ai-box`, json("POST", p)),
  aiModels: () => req<AiModels>("/ai/models"),
  aiPrompts: (projectId: number) => req<AiPrompts>(`/projects/${projectId}/ai/prompts`),
  /** Keep images as they are although the AI flagged them (what a review Save does with the AI's doubts, without saving). */
  dismissAi: (projectId: number, imageIds: number[]) => req<{ dismissed: number }>(`/projects/${projectId}/images/ai-dismiss`, json("POST", { image_ids: imageIds })),
  /** The weights are only stored, never opened, when they arrive. The header is what the server checks to know the upload came from this app. */
  aiUploadModel: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return req<AiDetector>("/ai/models/upload", { method: "POST", body: form, headers: { "X-MT-Upload": "1" } });
  },
  aiPreview: (projectId: number, p: AiJobParams) => req<{ count: number }>(`/projects/${projectId}/jobs/ai-check/preview`, json("POST", p)),
  createAiJob: (projectId: number, p: AiJobParams) => req<Job>(`/projects/${projectId}/jobs/ai-check`, json("POST", p)),
  cancelJob: (id: number) => req<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  datasetPreview: (projectId: number, sel?: DataSelection) =>
    req<DatasetPreview>(
      `/projects/${projectId}/datasets/preview` +
        (sel ? `?${new URLSearchParams({ sources: sel.sources.join(","), imported: sel.imported.join(",") })}` : ""),
    ),
  listDatasets: (projectId: number) => req<DatasetVersion[]>(`/projects/${projectId}/datasets`),
  createDataset: (
    projectId: number,
    body: {
      name?: string;
      val_ratio: number;
      test_ratio: number;
      include_negatives: boolean;
      sources?: SelectableSource[];
      imported_datasets?: string[];
      merge_classes?: { class_ids: number[]; name?: string }[]; // the version treats each group as one class; the project's labels are unchanged
    },
  ) =>
    req<DatasetVersion>(`/projects/${projectId}/datasets`, json("POST", body)),
  deleteDataset: (id: number) => req<void>(`/datasets/${id}`, { method: "DELETE" }),
  cloudCopies: (projectId: number) => req<CloudCopies>(`/projects/${projectId}/datasets/cloud`),
  deleteCloudCopies: (projectId: number, keys: string[]) =>
    req<{ deleted: number; freed_bytes: number }>(`/projects/${projectId}/datasets/cloud/delete`, json("POST", { keys })),
  deleteImage: (id: number) => req<void>(`/images/${id}`, { method: "DELETE" }),
  createTrainingRun: (projectId: number, body: TrainingParams) => req<TrainingRun>(`/projects/${projectId}/training-runs`, json("POST", body)),
  listTrainingRuns: (projectId: number) => req<TrainingRun[]>(`/projects/${projectId}/training-runs`),
  cancelTrainingRun: (id: number) => req<TrainingRun>(`/training-runs/${id}/cancel`, { method: "POST" }),
  deleteTrainingRun: (id: number) => req<void>(`/training-runs/${id}`, { method: "DELETE" }),
  trainingWorkers: () => req<WorkerInfo[]>("/training-workers"),
  piDisplay: () => req<PiDisplay>("/pi-display"),
  // These switch the Pi's camera and screen on: the header makes the browser ask the backend's CORS allow-list first.
  showOnPi: (runId: number) =>
    req<PiDisplay>(`/training-runs/${runId}/pi-display`, { ...json("POST", {}), headers: { "content-type": "application/json", "X-MT-Pi": "1" } }),
  stopOnPi: () => req<PiDisplay>("/pi-display", { method: "DELETE", headers: { "X-MT-Pi": "1" } }),
  runpodStatus: () => req<RunPodStatus>("/runpod/status"),
  runpodPlan: (runId: number, maxHours: number, cloud: RunPodCloud) => req<RunPodPlan>(`/runpod/plan?run_id=${runId}&max_hours=${maxHours}&cloud=${cloud}`),
  runpodLaunch: () => req<RunPodLaunch>("/runpod/launch"),
  // These spend money or end pods: the header makes the browser ask the backend's CORS allow-list first, so a page that is not this app cannot send them.
  startRunpodLaunch: (body: { run_id: number; max_hours: number; cloud: RunPodCloud; max_cost: number }) =>
    req<RunPodLaunch>("/runpod/launch", { ...json("POST", body), headers: { "content-type": "application/json", "X-MT-Spend": "1" } }),
  stopRunpodLaunch: () => req<RunPodLaunch>("/runpod/launch/stop", { method: "POST", headers: { "X-MT-Spend": "1" } }),
  deleteRunpodPod: (podId: string) => req<{ deleted: boolean }>(`/runpod/pods/${encodeURIComponent(podId)}/delete`, { method: "POST", headers: { "X-MT-Spend": "1" } }),
  /** The one-time command that sets up a Raspberry Pi as an always-on worker (it contains the worker token: copy it, don't show it). */
  piSetup: () => req<{ server: string; command: string }>("/pi-setup"),
};
