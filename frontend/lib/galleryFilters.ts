import type { ReviewStatus, Source } from "@/lib/api";

/** The main page's gallery filters. The review screen has no filters of its own: it shows the queue these describe. */
export interface GalleryFilters {
  classId?: number;
  /** One choice, so a source, an imported dataset and an upload folder can't be combined: "", "source:scraped", "dataset:anti-uav" or "folder:flight-1". */
  scope: string;
  reviewStatus?: ReviewStatus;
  /** Imported images only: true = opened in the review window, false = not yet. */
  manualCheck?: boolean;
  /** The AI check: "unchecked", "stale", "ok", "flagged", "dismissed", "too_small" or "autolabelled"; undefined = any. */
  ai?: string;
}

export const NO_FILTERS: GalleryFilters = { scope: "" };

const key = (projectId: number) => `gallery-filters:${projectId}`;

/** Per tab (sessionStorage), so the review screen and a return to the main page see the same filters. Never throws. */
export function loadGalleryFilters(projectId: number): GalleryFilters {
  try {
    const raw = sessionStorage.getItem(key(projectId));
    if (raw) return { ...NO_FILTERS, ...JSON.parse(raw) };
  } catch {
    // storage blocked or the value is not JSON: fall back to no filters
  }
  return NO_FILTERS;
}

export function saveGalleryFilters(projectId: number, filters: GalleryFilters) {
  try {
    sessionStorage.setItem(key(projectId), JSON.stringify(filters));
  } catch {
    // filters then just don't carry over to the review screen
  }
}

export function scopeFilter(scope: string): { source?: Source; dataset?: string; folder?: string } {
  if (scope.startsWith("source:")) return { source: scope.slice("source:".length) as Source };
  if (scope.startsWith("dataset:")) return { dataset: scope.slice("dataset:".length) };
  if (scope.startsWith("folder:")) return { folder: scope.slice("folder:".length) };
  return {};
}

/** What the gallery and the review screen both send to list images: one place, so the review queue is exactly the gallery's list. */
export function imageQuery(f: GalleryFilters) {
  return { classId: f.classId, ...scopeFilter(f.scope), manuallyChecked: f.manualCheck, ai: f.ai, reviewStatus: f.reviewStatus };
}

export const AI_FILTERS: { value: string; label: string }[] = [
  { value: "flagged", label: "AI: needs a look" },
  { value: "dismissed", label: "AI: dismissed" },
  { value: "autolabelled", label: "AI drew the boxes (not confirmed)" },
  { value: "ok", label: "AI: ok" },
  { value: "too_small", label: "AI: too small to judge" },
  { value: "stale", label: "AI: out of date" },
  { value: "unchecked", label: "AI: not checked" },
];

const STATUS_LABEL: Record<ReviewStatus, string> = { reviewed: "Reviewed", pending: "Not reviewed", rejected: "Rejected" };

/** A short read-only description of the filters, for the review screen's heading. */
export function describeFilters(f: GalleryFilters, className: (id: number) => string): string {
  const { source, dataset, folder } = scopeFilter(f.scope);
  const parts = [
    f.classId !== undefined ? className(f.classId) : null,
    dataset ?? (folder ? `folder ${folder}` : source) ?? null,
    f.manualCheck === undefined ? null : f.manualCheck ? "manually checked" : "not manually checked",
    f.ai ? AI_FILTERS.find((a) => a.value === f.ai)?.label ?? f.ai : null,
    f.reviewStatus ? STATUS_LABEL[f.reviewStatus] : null,
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "all images";
}
