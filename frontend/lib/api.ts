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
  id: string; // "run:5" or "file:name.pt"
  label: string;
  kind: "run" | "file";
  map50: number | null; // a training run's validation mAP50
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
  mode?: "check" | "autolabel"; // check = the AI's opinion of existing boxes; autolabel = draw boxes on unlabelled pending images
  detector?: string;
  sam?: AiSamSize;
  min_score?: number;
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

/** The backend stores UTC but its timestamps carry no zone suffix, which `new Date` would read as local time. */
export const parseUtc = (s: string) => new Date(/(Z|[+-]\d\d:?\d\d)$/.test(s) ? s : s + "Z");

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : res.statusText);
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

export const api = {
  listProjects: () => req<Project[]>("/projects"),
  createProject: (name: string) => req<Project>("/projects", json("POST", { name })),
  getProject: (id: number) => req<Project>(`/projects/${id}`),
  createClass: (projectId: number, name: string, isNegative: boolean) =>
    req<DetClass>(`/projects/${projectId}/classes`, json("POST", { name, is_negative: isNegative })),
  createClasses: (projectId: number, names: string[], isNegative: boolean) =>
    req<BulkClassResult>(`/projects/${projectId}/classes/bulk`, json("POST", { names, is_negative: isNegative })),
  deleteClass: (projectId: number, classId: number) => req<void>(`/projects/${projectId}/classes/${classId}`, { method: "DELETE" }),
  stats: (projectId: number) => req<StatRow[]>(`/projects/${projectId}/stats`),
  listImages: (
    projectId: number,
    p: { classId?: number; source?: Source; dataset?: string; folder?: string; manuallyChecked?: boolean; ai?: string; reviewStatus?: ReviewStatus; limit: number; offset: number },
  ) => {
    const q = new URLSearchParams({ limit: String(p.limit), offset: String(p.offset) });
    if (p.classId != null) q.set("class_id", String(p.classId));
    if (p.source) q.set("source", p.source);
    if (p.dataset) q.set("dataset", p.dataset);
    if (p.folder) q.set("folder", p.folder);
    if (p.manuallyChecked !== undefined) q.set("manually_checked", String(p.manuallyChecked));
    if (p.ai) q.set("ai", p.ai);
    if (p.reviewStatus) q.set("review_status", p.reviewStatus);
    return req<ImageRow[]>(`/projects/${projectId}/images?${q}`);
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
  /** Reject several images exactly as the review screen's Reject does for one. There is no bulk accept: accepting means a person looked. */
  rejectImages: (projectId: number, imageIds: number[]) => req<{ rejected: number }>(`/projects/${projectId}/images/reject`, json("POST", { image_ids: imageIds })),
  getImage: (id: number) => req<ImageDetail>(`/images/${id}`),
  /** `aiDismiss`: these boxes are the person's decision on the AI's doubts (Accept / Reject): a flagged image leaves "AI: needs a look". */
  putAnnotations: (id: number, boxes: BoxIn[], reviewStatus?: ReviewStatus, aiDismiss?: boolean) =>
    req<ImageDetail>(`/images/${id}/annotations`, json("PUT", { boxes, review_status: reviewStatus ?? null, ai_dismiss: aiDismiss ?? false })),
  createScrapeJob: (projectId: number, p: Partial<ScrapeParams> & Pick<ScrapeParams, "class_id" | "query">) =>
    req<Job>(`/projects/${projectId}/jobs/scrape`, json("POST", p)),
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
    body: { name?: string; val_ratio: number; test_ratio: number; include_negatives: boolean; sources?: SelectableSource[]; imported_datasets?: string[] },
  ) =>
    req<DatasetVersion>(`/projects/${projectId}/datasets`, json("POST", body)),
  deleteDataset: (id: number) => req<void>(`/datasets/${id}`, { method: "DELETE" }),
  deleteImage: (id: number) => req<void>(`/images/${id}`, { method: "DELETE" }),
  createTrainingRun: (projectId: number, body: TrainingParams) => req<TrainingRun>(`/projects/${projectId}/training-runs`, json("POST", body)),
  listTrainingRuns: (projectId: number) => req<TrainingRun[]>(`/projects/${projectId}/training-runs`),
  cancelTrainingRun: (id: number) => req<TrainingRun>(`/training-runs/${id}/cancel`, { method: "POST" }),
  deleteTrainingRun: (id: number) => req<void>(`/training-runs/${id}`, { method: "DELETE" }),
  trainingWorkers: () => req<WorkerInfo[]>("/training-workers"),
};
