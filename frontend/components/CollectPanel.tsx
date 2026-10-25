"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type CollectParams, type DetClass, type Job } from "@/lib/api";

const ACTIVE = ["queued", "running"];
const STATUS_STYLE: Record<Job["status"], string> = {
  queued: "text-neutral-500",
  running: "text-blue-500",
  succeeded: "text-green-500",
  failed: "text-red-500",
  cancelled: "text-amber-500",
};
const COUNTER_LABELS: [string, string][] = [
  ["pages", "pages read"],
  ["pages_failed", "pages failed"],
  ["robots_blocked", "blocked by robots.txt"],
  ["candidates", "pictures found"],
  ["added", "added"],
  ["already_scraped", "already seen"],
  ["duplicates", "duplicates"],
  ["near_duplicates", "near-duplicates"],
  ["too_small", "too small"],
  ["download_failed", "download failed"],
  ["rejected", "rejected"],
];
const MAX_URLS = 50;

/** One web address per line (spaces and commas also separate them). */
const parseUrls = (text: string) => [...new Set(text.split(/[\s,]+/).filter(Boolean))];
const isWebAddress = (u: string) => /^https?:\/\/[^\s/]+/i.test(u);

/** Read the pictures off article and gallery pages: news sites, company press pages, Wikimedia. Costs no SerpAPI search. */
export default function CollectPanel({ projectId, classes, onImagesChanged }: { projectId: number; classes: DetClass[]; onImagesChanged: () => void }) {
  const [classId, setClassId] = useState<number | "">("");
  const [text, setText] = useState("");
  const [follow, setFollow] = useState(false);
  const [linkWords, setLinkWords] = useState("");
  const [startPages, setStartPages] = useState(true);
  const [maxImages, setMaxImages] = useState(100);
  const [maxPages, setMaxPages] = useState(30);
  const [perPage, setPerPage] = useState(12);
  const [byArticle, setByArticle] = useState(true);
  const [minSide, setMinSide] = useState(300);
  const [nearDup, setNearDup] = useState(4);
  const [skipKnown, setSkipKnown] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const lastAdded = useRef(new Map<number, number>());

  const urls = parseUrls(text);
  const bad = urls.find((u) => !isWebAddress(u));

  const refresh = useCallback(async () => {
    try {
      const next = await api.listJobs(projectId, "collect_pages");
      setJobs(next);
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

  useEffect(() => { refresh(); }, [refresh]);

  const anyActive = jobs.some((j) => ACTIVE.includes(j.status));
  useEffect(() => {
    if (!anyActive) return;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [anyActive, refresh]);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    if (classId === "" || urls.length === 0) return;
    if (bad) {
      setError(`“${bad.slice(0, 80)}” is not a web address. Each line must start with http:// or https://`);
      return;
    }
    if (urls.length > MAX_URLS) {
      setError(`At most ${MAX_URLS} addresses per job; this has ${urls.length}. Turn on “follow links” to reach more pages from an index page.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.createCollectJob(projectId, {
        class_id: classId,
        urls,
        follow_links: follow,
        link_contains: follow ? linkWords.split(",").map((w) => w.trim()).filter(Boolean) : [],
        include_start_pages: !follow || startPages,
        max_pages: maxPages,
        max_images: maxImages,
        images_per_page: perPage,
        min_side: minSide,
        near_dup_distance: nearDup,
        skip_known_urls: skipKnown,
        group_by: byArticle ? "article" : "site",
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

  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-500">
        Paste addresses of articles or galleries (news sites, company press pages, Wikimedia). The pictures on each page are downloaded, checked for size and
        duplicates, and filed as pending images. This costs no SerpAPI search.
      </p>
      <form onSubmit={start} className="space-y-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">File pictures under</span>
            <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))} required data-testid="collect-class">
              <option value="">Select class…</option>
              {classes.filter((c) => !c.is_mixed).map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                  {c.is_negative ? " (negative)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">Max pictures</span>
            <input type="number" className={num} min={1} max={1000} value={maxImages} onChange={(e) => setMaxImages(Number(e.target.value))} />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">Max pages to read</span>
            <input type="number" className={num} min={1} max={300} value={maxPages} onChange={(e) => setMaxPages(Number(e.target.value))} data-testid="collect-max-pages" />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">Pictures per page</span>
            <input type="number" className={num} min={1} max={50} value={perPage} onChange={(e) => setPerPage(Number(e.target.value))} />
          </label>
          <button className="btn" disabled={busy || classId === "" || urls.length === 0} data-testid="collect-start">
            {busy ? "Starting…" : "Collect pictures"}
          </button>
        </div>

        <label className="block text-sm">
          <span className="mb-1 block text-neutral-500">
            Page addresses, one per line ({urls.length} of {MAX_URLS})
          </span>
          <textarea
            className="input h-28 w-full font-mono text-xs"
            placeholder={"https://example.com/news/kizilelma-first-flight\nhttps://example.com/gallery/anka-3"}
            value={text}
            onChange={(e) => setText(e.target.value)}
            data-testid="collect-urls"
          />
        </label>
        {bad && <p className="text-sm text-amber-500">“{bad.slice(0, 80)}” is not a web address: each one must start with http:// or https://.</p>}

        <div className="space-y-2 text-sm">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={follow} onChange={(e) => { setFollow(e.target.checked); setStartPages(!e.target.checked); }} data-testid="collect-follow" />
            Also read the pages these link to (same website, one level deep)
          </label>
          {follow && (
            <div className="ml-6 space-y-2">
              <label className="block">
                <span className="mb-1 block text-neutral-500">Only links containing these words (comma-separated, optional)</span>
                <input className="input w-full max-w-md" placeholder="e.g. /news/2025/, kizilelma" value={linkWords} onChange={(e) => setLinkWords(e.target.value)} data-testid="collect-words" />
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={startPages} onChange={(e) => setStartPages(e.target.checked)} />
                Also take the pictures on the start pages themselves
              </label>
              <p className="text-xs text-neutral-500">An index page&apos;s own pictures are mostly small thumbnails of the articles it links to, so by default it is used only to find them. The pages read, start pages included, never exceed “Max pages to read”.</p>
            </div>
          )}
          <label className="flex items-center gap-2" title="Pictures of one article stay together in one split, so a near-identical photo cannot be in training and in testing.">
            <input type="checkbox" checked={byArticle} onChange={(e) => setByArticle(e.target.checked)} />
            Keep each page&apos;s pictures together in one train/val/test split
          </label>
        </div>

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
              <input type="checkbox" checked={skipKnown} onChange={(e) => setSkipKnown(e.target.checked)} /> Skip picture addresses already tried
            </label>
          </div>
        </details>
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-neutral-500">
          <li>Polite on purpose: robots.txt is obeyed, and each website gets one request a second, so a big job takes a while. Jobs run one at a time, so a long crawl holds up AI jobs.</li>
          <li>Many news sites turn automated readers away (HTTP 403); those pages show up under “pages failed” with the reason.</li>
          <li>Pictures on news and company sites belong to their owners: they are for private training here, and each one keeps its source page for attribution. Review them (the review window has Accept and Reject) before training.</li>
        </ul>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="space-y-2">
        {jobs.map((j) => {
          const p = j.params as Partial<CollectParams>;
          const added = j.progress.added ?? 0;
          const target = j.progress.target ?? p.max_images ?? 0;
          const failure = (j.progress as Record<string, unknown>).last_failure as string | undefined;
          const waiting = j.progress.queued_pages ?? 0;
          return (
            <li key={j.id} className="rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid={`collect-job-${j.id}`}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">#{j.id}</span>
                <span>
                  {(p.urls?.length ?? 0) === 1 ? p.urls?.[0] : `${p.urls?.length ?? 0} pages`}
                  {p.follow_links ? " + links" : ""} → {className(p.class_id)}
                </span>
                <span className={STATUS_STYLE[j.status]} data-testid="collect-job-status">
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
              <p className="mt-1 text-neutral-500" data-testid="collect-job-counters">
                {COUNTER_LABELS.filter(([k]) => (j.progress[k] ?? 0) > 0 || k === "added").map(([k, label]) => `${j.progress[k] ?? 0} ${label}`).join(" · ")}
                {waiting > 0 && ACTIVE.includes(j.status) && ` · ${waiting} waiting`}
                {target > 0 && ` (target ${target})`}
              </p>
              {failure && <p className="mt-1 text-amber-500" data-testid="collect-job-failure">Last problem: {failure}</p>}
              {j.error && <p className="mt-1 text-red-600">{j.error}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
