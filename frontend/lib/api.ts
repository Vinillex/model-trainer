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

export interface ImageRow {
  id: number;
  class_id: number;
  source: Source;
  sha256: string;
  width: number;
  height: number;
  group_key: string | null;
  created_at: string;
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
  stats: (projectId: number) => req<StatRow[]>(`/projects/${projectId}/stats`),
  listImages: (projectId: number, p: { classId?: number; source?: Source; limit: number; offset: number }) => {
    const q = new URLSearchParams({ limit: String(p.limit), offset: String(p.offset) });
    if (p.classId !== undefined) q.set("class_id", String(p.classId));
    if (p.source) q.set("source", p.source);
    return req<ImageRow[]>(`/projects/${projectId}/images?${q}`);
  },
  upload: (projectId: number, classId: number, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return req<UploadResult>(`/projects/${projectId}/classes/${classId}/images`, { method: "POST", body: form });
  },
  deleteImage: (id: number) => req<void>(`/images/${id}`, { method: "DELETE" }),
};
