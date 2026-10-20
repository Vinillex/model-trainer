"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type DetClass, type Job, type SearchHistory, type SerpApiAccount } from "@/lib/api";

const ACTIVE = ["queued", "running"];
const STATUS_STYLE: Record<Job["status"], string> = {
  queued: "text-neutral-500",
  running: "text-blue-500",
  succeeded: "text-green-500",
  failed: "text-red-500",
  cancelled: "text-amber-500",
};
const COUNTER_LABELS: [string, string][] = [
  ["candidates", "found"],
  ["added", "added"],
  ["already_scraped", "already seen"],
  ["duplicates", "duplicates"],
  ["near_duplicates", "near-duplicates"],
  ["too_small", "too small"],
  ["download_failed", "download failed"],
  ["rejected", "rejected"],
];

/** [0,1,2,4] -> "1–3, 5" (pages are shown 1-based; SerpAPI counts from 0). */
function pageRanges(pages: number[]): string {
  const out: string[] = [];
  for (let i = 0; i < pages.length; ) {
    let j = i;
    while (j + 1 < pages.length && pages[j + 1] === pages[j] + 1) j++;
    out.push(i === j ? `${pages[i] + 1}` : `${pages[i] + 1}–${pages[j] + 1}`);
    i = j + 1;
  }
  return out.join(", ");
}

export default function ScrapePanel({ projectId, classes, onImagesChanged }: { projectId: number; classes: DetClass[]; onImagesChanged: () => void }) {
  const [classId, setClassId] = useState<number | "">("");
  const [query, setQuery] = useState("");
  const [maxImages, setMaxImages] = useState(50);
  const [startPage, setStartPage] = useState(1); // 1-based in the UI
  const [startTouched, setStartTouched] = useState(false);
  const [maxPages, setMaxPages] = useState(3);
  const [minSide, setMinSide] = useState(200);
  const [nearDup, setNearDup] = useState(4);
  const [skipKnown, setSkipKnown] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [history, setHistory] = useState<SearchHistory | null>(null);
  const [account, setAccount] = useState<SerpApiAccount | null>(null);
  const [accountError, setAccountError] = useState<string | null>(null);
  const lastAdded = useRef(new Map<number, number>());
  const lastStatus = useRef(new Map<number, string>());
  const queryRef = useRef(query); // read by refresh(), so typing doesn't re-create it and refetch the job list
  queryRef.current = query;

  const loadAccount = useCallback(() => {
    api.serpapiAccount().then((a) => { setAccount(a); setAccountError(null); }).catch((e) => { setAccount(null); setAccountError(e.message); });
  }, []);

  const loadHistory = useCallback(async (q: string) => {
    try {
      const h = await api.searchHistory(projectId, q);
      setHistory(h);
      return h;
    } catch {
      setHistory(null);
      return null;
    }
  }, [projectId]);

  const refresh = useCallback(async () => {
    try {
      const next = await api.listJobs(projectId);
      setJobs(next);
      // Reload stats/gallery when a job has added images, and the quota when a job finishes (it spent searches).
      let changed = false;
      let finished = false;
      for (const j of next) {
        const added = j.progress.added ?? 0;
        if (lastAdded.current.has(j.id) && lastAdded.current.get(j.id) !== added) changed = true;
        lastAdded.current.set(j.id, added);
        if (lastStatus.current.has(j.id) && ACTIVE.includes(lastStatus.current.get(j.id)!) && !ACTIVE.includes(j.status)) finished = true;
        lastStatus.current.set(j.id, j.status);
      }
      if (changed) onImagesChanged();
      if (finished) {
        loadAccount();
        if (queryRef.current.trim()) loadHistory(queryRef.current.trim());
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }, [projectId, onImagesChanged, loadAccount, loadHistory]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { loadAccount(); }, [loadAccount]);

  const anyActive = jobs.some((j) => ACTIVE.includes(j.status));
  useEffect(() => {
    if (!anyActive) return;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [anyActive, refresh]);

  // Look up which pages of this query were already searched, and suggest the next one.
  useEffect(() => {
    const q = query.trim();
    if (!q) { setHistory(null); return; }
    const t = setTimeout(async () => {
      const h = await loadHistory(q);
      if (h && !startTouched) setStartPage(h.next_page + 1);
    }, 400);
    return () => clearTimeout(t);
  }, [query, startTouched, loadHistory]);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    if (classId === "") return;
    setBusy(true);
    setError(null);
    try {
      await api.createScrapeJob(projectId, {
        class_id: classId, query: query.trim(), max_images: maxImages, start_page: startPage - 1, max_pages: maxPages,
        min_side: minSide, near_dup_distance: nearDup, skip_known_urls: skipKnown,
      });
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
  const left = account?.total_searches_left ?? null;
  const alreadySearched = history?.pages_searched.includes(startPage - 1) ?? false;
  const lastPage = startPage - 1 + maxPages; // 1-based last page the job may reach

  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-500" data-testid="quota">
        {account
          ? <>SerpAPI {account.plan_name}: <b>{left}</b> of {account.searches_per_month} searches left this month. One search = one page of Google Images results, not one image; downloading the images costs nothing.</>
          : accountError ? `SerpAPI quota unavailable: ${accountError}` : "Checking SerpAPI quota…"}
      </p>
      <form onSubmit={start} className="space-y-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">File images under</span>
            <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))} required>
              <option value="">Select class…</option>
              {classes.filter((c) => !c.is_mixed).map((c) => (
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
        <p className="text-xs text-neutral-500">The class only decides the folder and whether the images count as hard negatives. A search can return other kinds of drone too: in review, give each box its own class.</p>

        <div className="flex flex-wrap items-end gap-4 text-sm">
          <label>
            <span className="mb-1 block text-neutral-500">Start at results page</span>
            <input type="number" className={num} min={1} max={100} value={startPage} onChange={(e) => { setStartPage(Math.max(1, Number(e.target.value))); setStartTouched(true); }} data-testid="start-page" />
          </label>
          <label>
            <span className="mb-1 block text-neutral-500">Pages to search</span>
            <input type="number" className={num} min={1} max={20} value={maxPages} onChange={(e) => setMaxPages(Number(e.target.value))} />
          </label>
          <p className="pb-2 text-neutral-500" data-testid="page-plan">
            Searches pages {startPage}–{lastPage}: up to {maxPages} {maxPages === 1 ? "search" : "searches"} (stops early once it has {maxImages} images).
          </p>
        </div>
        {history && history.pages_searched.length > 0 && (
          <div className="space-y-1 text-sm text-neutral-500" data-testid="history-hint">
            <p>This query was already searched on pages {pageRanges(history.pages_searched)}.</p>
            {history.exhausted_at !== null ? (
              <p className="text-amber-500">Google ran out of results at page {history.exhausted_at + 1}. For more images, try a different search query.</p>
            ) : (
              startPage !== history.next_page + 1 && (
                <p>
                  <button type="button" className="underline" onClick={() => { setStartPage(history.next_page + 1); setStartTouched(true); }}>
                    Start at page {history.next_page + 1} instead
                  </button>{" "}
                  to continue where the last search stopped.
                </p>
              )
            )}
            {alreadySearched && <p className="text-amber-500">Page {startPage} was already searched: its images are skipped, but this still spends a search.</p>}
          </div>
        )}
        {left !== null && maxPages > left && <p className="text-sm text-amber-500">This job may need {maxPages} searches but only {left} are left this month.</p>}

        <details className="text-sm">
          <summary className="cursor-pointer text-neutral-500">Advanced</summary>
          <div className="mt-2 flex flex-wrap items-end gap-4">
            <label>
              <span className="mb-1 block text-neutral-500">Min side (px)</span>
              <input type="number" className={num} min={32} max={4000} value={minSide} onChange={(e) => setMinSide(Number(e.target.value))} />
            </label>
            <label>
              <span className="mb-1 block text-neutral-500">Near-duplicate distance (0 = keep all)</span>
              <input type="number" className={num} min={0} max={16} value={nearDup} onChange={(e) => setNearDup(Number(e.target.value))} />
            </label>
            <label className="flex items-center gap-1 pb-2">
              <input type="checkbox" checked={skipKnown} onChange={(e) => setSkipKnown(e.target.checked)} /> Skip image URLs already tried
            </label>
          </div>
        </details>
        <p className="text-xs text-neutral-500">Scraped images arrive as pending, so review them before training, and check each source&apos;s licence.</p>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="space-y-2">
        {jobs.map((j) => {
          const added = j.progress.added ?? 0;
          const target = j.progress.target ?? j.params.max_images ?? 0;
          const pages = j.progress.pages ?? 0;
          const first = (j.progress.start_page ?? j.params.start_page ?? 0) + 1;
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
                {pages > 0 ? `page${pages > 1 ? "s" : ""} ${pages > 1 ? `${first}–${first + pages - 1}` : first} (${pages} ${pages === 1 ? "search" : "searches"}) · ` : ""}
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
