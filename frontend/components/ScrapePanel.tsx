"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type DetClass, type Job } from "@/lib/api";

const ACTIVE = ["queued", "running"];
const STATUS_STYLE: Record<Job["status"], string> = {
  queued: "text-neutral-500",
  running: "text-blue-500",
  succeeded: "text-green-500",
  failed: "text-red-500",
  cancelled: "text-amber-500",
};
const COUNTER_LABELS: [string, string][] = [
  ["pages", "searches"],
  ["candidates", "found"],
  ["added", "added"],
  ["duplicates", "duplicates"],
  ["near_duplicates", "near-duplicates"],
  ["already_scraped", "already scraped"],
  ["too_small", "too small"],
  ["download_failed", "download failed"],
  ["rejected", "rejected"],
];

export default function ScrapePanel({ projectId, classes, onImagesChanged }: { projectId: number; classes: DetClass[]; onImagesChanged: () => void }) {
  const [classId, setClassId] = useState<number | "">("");
  const [query, setQuery] = useState("");
  const [maxImages, setMaxImages] = useState(50);
  const [maxPages, setMaxPages] = useState(3);
  const [minSide, setMinSide] = useState(200);
  const [nearDup, setNearDup] = useState(4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const lastAdded = useRef(new Map<number, number>());

  const refresh = useCallback(async () => {
    try {
      const next = await api.listJobs(projectId);
      setJobs(next);
      // Tell the page to reload stats/gallery whenever a job has added images since we last looked.
      let changed = false;
      for (const j of next) {
        const added = j.progress.added ?? 0;
        if (lastAdded.current.has(j.id) && lastAdded.current.get(j.id) !== added) changed = true;
        lastAdded.current.set(j.id, added);
      }
      if (changed) onImagesChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [projectId, onImagesChanged]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const anyActive = jobs.some((j) => ACTIVE.includes(j.status));
  useEffect(() => {
    if (!anyActive) return;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [anyActive, refresh]);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    if (classId === "") return;
    setBusy(true);
    setError(null);
    try {
      await api.createScrapeJob(projectId, { class_id: classId, query: query.trim(), max_images: maxImages, max_pages: maxPages, min_side: minSide, near_dup_distance: nearDup });
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function cancel(id: number) {
    try {
      await api.cancelJob(id);
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const className = (id?: number) => classes.find((c) => c.id === id)?.name ?? "?";
  const num = "input w-20";

  return (
    <div className="space-y-4">
      <form onSubmit={start} className="space-y-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">File images under</span>
            <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))} required>
              <option value="">Select class…</option>
              {classes.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                  {c.is_negative ? " (negative)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="min-w-64 flex-1 text-sm">
            <span className="mb-1 block text-neutral-500">Google Images search</span>
            <input className="input w-full" placeholder="e.g. DJI quadcopter flying in sky" value={query} onChange={(e) => setQuery(e.target.value)} maxLength={300} required />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">Max images</span>
            <input type="number" className={num} min={1} max={500} value={maxImages} onChange={(e) => setMaxImages(Number(e.target.value))} />
          </label>
          <button className="btn" disabled={busy || classId === "" || !query.trim()}>
            {busy ? "Starting…" : "Start scrape"}
          </button>
        </div>
        <details className="text-sm">
          <summary className="cursor-pointer text-neutral-500">Advanced</summary>
          <div className="mt-2 flex flex-wrap gap-4">
            <label>
              <span className="mb-1 block text-neutral-500">Max searches (pages)</span>
              <input type="number" className={num} min={1} max={20} value={maxPages} onChange={(e) => setMaxPages(Number(e.target.value))} />
            </label>
            <label>
              <span className="mb-1 block text-neutral-500">Min side (px)</span>
              <input type="number" className={num} min={32} max={4000} value={minSide} onChange={(e) => setMinSide(Number(e.target.value))} />
            </label>
            <label>
              <span className="mb-1 block text-neutral-500">Near-duplicate distance (0 = keep all)</span>
              <input type="number" className={num} min={0} max={16} value={nearDup} onChange={(e) => setNearDup(Number(e.target.value))} />
            </label>
          </div>
        </details>
        <p className="text-xs text-neutral-500">Each search page is one SerpAPI search (about 100 results); this job will use at most {maxPages}. Scraped images arrive as pending, so review them before training, and check each source&apos;s licence.</p>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="space-y-2">
        {jobs.map((j) => {
          const added = j.progress.added ?? 0;
          const target = j.progress.target ?? j.params.max_images ?? 0;
          return (
            <li key={j.id} className="rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid={`job-${j.id}`}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">#{j.id}</span>
                <span>
                  “{j.params.query}” → {className(j.params.class_id)}
                </span>
                <span className={STATUS_STYLE[j.status]} data-testid="job-status">
                  {j.status}
                  {j.status === "running" && j.cancel_requested ? " (stopping…)" : ""}
                </span>
                {ACTIVE.includes(j.status) && !j.cancel_requested && (
                  <button className="btn ml-auto" onClick={() => cancel(j.id)}>
                    Cancel
                  </button>
                )}
              </div>
              {target > 0 && (
                <div className="mt-2 h-1.5 overflow-hidden rounded bg-neutral-200 dark:bg-neutral-800">
                  <div className="h-full bg-green-600 transition-all" style={{ width: `${Math.min(100, (added / target) * 100)}%` }} />
                </div>
              )}
              <p className="mt-1 text-neutral-500" data-testid="job-counters">
                {COUNTER_LABELS.filter(([k]) => (j.progress[k] ?? 0) > 0 || k === "added").map(([k, label]) => `${j.progress[k] ?? 0} ${label}`).join(" · ")}
                {target > 0 && ` (target ${target})`}
              </p>
              {j.error && <p className="mt-1 text-red-600">{j.error}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
