"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type DetClass, type Job, type VideoParams } from "@/lib/api";

const ACTIVE = ["queued", "running"];
const STATUS_STYLE: Record<Job["status"], string> = {
  queued: "text-neutral-500",
  running: "text-blue-500",
  succeeded: "text-green-500",
  failed: "text-red-500",
  cancelled: "text-amber-500",
};
const COUNTER_LABELS: [string, string][] = [
  ["videos", "videos read"],
  ["videos_failed", "videos failed"],
  ["frames_cut", "frames cut"],
  ["added", "added"],
  ["already_scraped", "already seen"],
  ["duplicates", "duplicates"],
  ["near_duplicates", "near-duplicates"],
  ["too_small", "too small"],
  ["download_failed", "download failed"],
  ["rejected", "rejected"],
];
const MAX_URLS = 20;

/** One web address per line (spaces and commas also separate them). */
const parseUrls = (text: string) => [...new Set(text.split(/[\s,]+/).filter(Boolean))];
const isWebAddress = (u: string) => /^https?:\/\/[^\s/]+/i.test(u);

/** Cut still frames out of videos. Small targets seen from the air are rare as photographs but common in video. */
export default function VideoPanel({ projectId, classes, onImagesChanged }: { projectId: number; classes: DetClass[]; onImagesChanged: () => void }) {
  const [classId, setClassId] = useState<number | "">("");
  const [text, setText] = useState("");
  const [fps, setFps] = useState(1);
  const [perVideo, setPerVideo] = useState(120);
  const [maxImages, setMaxImages] = useState(300);
  const [minSide, setMinSide] = useState(240);
  const [nearDup, setNearDup] = useState(5);
  const [skipKnown, setSkipKnown] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const lastAdded = useRef(new Map<number, number>());

  const urls = parseUrls(text);
  const bad = urls.find((u) => !isWebAddress(u));

  const refresh = useCallback(async () => {
    try {
      const next = await api.listJobs(projectId, "video_frames");
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
      setError(`At most ${MAX_URLS} videos per job; this has ${urls.length}.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.createVideoJob(projectId, {
        class_id: classId,
        urls,
        frames_per_second: fps,
        max_frames_per_video: perVideo,
        max_images: maxImages,
        min_side: minSide,
        near_dup_distance: nearDup,
        skip_known_urls: skipKnown,
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
        Paste addresses of videos: a Wikimedia Commons file page (commons.wikimedia.org/wiki/File:…) or a direct link to a video file. A few still frames are cut from each
        video, checked for size and duplicates, and filed as pending images. All frames of one video stay together in one train/val/test split.
      </p>
      <form onSubmit={start} className="space-y-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">File frames under</span>
            <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))} required data-testid="video-class">
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
            <span className="mb-1 block text-neutral-500">Frames per second</span>
            <input type="number" className={num} min={0.1} max={5} step={0.1} value={fps} onChange={(e) => setFps(Number(e.target.value))} data-testid="video-fps" />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">Max frames per video</span>
            <input type="number" className={num} min={1} max={600} value={perVideo} onChange={(e) => setPerVideo(Number(e.target.value))} />
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-neutral-500">Max pictures in total</span>
            <input type="number" className={num} min={1} max={2000} value={maxImages} onChange={(e) => setMaxImages(Number(e.target.value))} />
          </label>
          <button className="btn" disabled={busy || classId === "" || urls.length === 0} data-testid="video-start">
            {busy ? "Starting…" : "Cut frames"}
          </button>
        </div>

        <label className="block text-sm">
          <span className="mb-1 block text-neutral-500">
            Video addresses, one per line ({urls.length} of {MAX_URLS})
          </span>
          <textarea
            className="input h-28 w-full font-mono text-xs"
            placeholder={"https://commons.wikimedia.org/wiki/File:Shahed_drone_in_Dnipropetrovsk_Oblast_(2024-02-09).webm"}
            value={text}
            onChange={(e) => setText(e.target.value)}
            data-testid="video-urls"
          />
        </label>
        {bad && <p className="text-sm text-amber-500">“{bad.slice(0, 80)}” is not a web address: each one must start with http:// or https://.</p>}

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
              <input type="checkbox" checked={skipKnown} onChange={(e) => setSkipKnown(e.target.checked)} /> Skip frames already cut
            </label>
          </div>
        </details>
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-neutral-500">
          <li>
            A long video gets a lower rate so “Max frames per video” frames cover all of it, not just the start. Frames of a slow scene look alike, so most are dropped as
            near-duplicates.
          </li>
          <li>Needs ffmpeg on the computer running the backend. Videos over 400 MB are skipped. robots.txt is obeyed and each website gets one request a second.</li>
          <li>YouTube and other video sites are not supported: only Commons file pages and direct video files.</li>
        </ul>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="space-y-2">
        {jobs.map((j) => {
          const p = j.params as Partial<VideoParams>;
          const added = j.progress.added ?? 0;
          const target = j.progress.target ?? p.max_images ?? 0;
          const failure = (j.progress as Record<string, unknown>).last_failure as string | undefined;
          const waiting = j.progress.queued_videos ?? 0;
          return (
            <li key={j.id} className="rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid={`video-job-${j.id}`}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">#{j.id}</span>
                <span>
                  {(p.urls?.length ?? 0) === 1 ? p.urls?.[0] : `${p.urls?.length ?? 0} videos`} → {className(p.class_id)}
                </span>
                <span className={STATUS_STYLE[j.status]} data-testid="video-job-status">
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
              <p className="mt-1 text-neutral-500" data-testid="video-job-counters">
                {COUNTER_LABELS.filter(([k]) => (j.progress[k] ?? 0) > 0 || k === "added").map(([k, label]) => `${j.progress[k] ?? 0} ${label}`).join(" · ")}
                {waiting > 0 && ACTIVE.includes(j.status) && ` · ${waiting} waiting`}
                {target > 0 && ` (target ${target})`}
              </p>
              {failure && <p className="mt-1 text-amber-500" data-testid="video-job-failure">Last problem: {failure}</p>}
              {j.error && <p className="mt-1 text-red-600">{j.error}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
