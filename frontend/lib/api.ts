export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Source = "uploaded" | "scraped" | "generated";
export const SOURCES: Source[] = ["uploaded", "scraped", "generated"];

export interface DetClass {
  id: number;
  name: string;
  is_negative: boolean;
  index: number | null;
}

export interface Project {
  id: number;
  name: string;
  created_at: string;
  classes: DetClass[];
}

export type ReviewStatus = "pending" | "reviewed" | "rejected";

export interface ImageRow {
  id: number;
  class_id: number;
  source: Source;
  sha256: string;
  width: number;
  height: number;
  group_key: string | null;
  review_status: ReviewStatus;
  created_at: string;
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
}

export interface StatRow {
  class_id: number;
  class_name: string;
  is_negative: boolean;
  uploaded: number;
  scraped: number;
  generated: number;
  pending: number;
  rejected: number;
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

export const imageUrl = (id: number) => `${API}/images/${id}/file`;

export const api = {
  listProjects: () => req<Project[]>("/projects"),
  createProject: (name: string) => req<Project>("/projects", json("POST", { name })),
  getProject: (id: number) => req<Project>(`/projects/${id}`),
  createClass: (projectId: number, name: string, isNegative: boolean) =>
    req<DetClass>(`/projects/${projectId}/classes`, json("POST", { name, is_negative: isNegative })),
  createClasses: (projectId: number, names: string[], isNegative: boolean) =>
    req<BulkClassResult>(`/projects/${projectId}/classes/bulk`, json("POST", { names, is_negative: isNegative })),
  stats: (projectId: number) => req<StatRow[]>(`/projects/${projectId}/stats`),
  listImages: (projectId: number, p: { classId?: number; source?: Source; reviewStatus?: ReviewStatus; limit: number; offset: number }) => {
    const q = new URLSearchParams({ limit: String(p.limit), offset: String(p.offset) });
    if (p.classId !== undefined) q.set("class_id", String(p.classId));
    if (p.source) q.set("source", p.source);
    if (p.reviewStatus) q.set("review_status", p.reviewStatus);
    return req<ImageRow[]>(`/projects/${projectId}/images?${q}`);
  },
  upload: (projectId: number, classId: number, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return req<UploadResult>(`/projects/${projectId}/classes/${classId}/images`, { method: "POST", body: form });
  },
  getImage: (id: number) => req<ImageDetail>(`/images/${id}`),
  putAnnotations: (id: number, boxes: BoxIn[], reviewStatus?: ReviewStatus) =>
    req<ImageDetail>(`/images/${id}/annotations`, json("PUT", { boxes, review_status: reviewStatus ?? null })),
  createScrapeJob: (projectId: number, p: Partial<ScrapeParams> & Pick<ScrapeParams, "class_id" | "query">) =>
    req<Job>(`/projects/${projectId}/jobs/scrape`, json("POST", p)),
  searchHistory: (projectId: number, query: string) =>
    req<SearchHistory>(`/projects/${projectId}/searches?${new URLSearchParams({ query })}`),
  serpapiAccount: () => req<SerpApiAccount>("/serpapi/account"),
  listJobs: (projectId: number) => req<Job[]>(`/projects/${projectId}/jobs?limit=10`),
  cancelJob: (id: number) => req<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  deleteImage: (id: number) => req<void>(`/images/${id}`, { method: "DELETE" }),
};
