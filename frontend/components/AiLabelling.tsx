"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type AiDetector, type AiJobParams, type AiModels, type AiSamSize, type AiServiceStatus, type DetClass, type FolderRow, type Job } from "@/lib/api";
import { takeAiSelection } from "@/lib/aiSelection";
import { NO_FILTERS, loadGalleryFilters, saveGalleryFilters, scopeFilter } from "@/lib/galleryFilters";

const ACTIVE = ["queued", "running"];
const STATUS_STYLE: Record<Job["status"], string> = {
  queued: "text-neutral-500",
  running: "text-blue-500",
  succeeded: "text-green-500",
  failed: "text-red-500",
  cancelled: "text-amber-500",
};
const SECONDS_PER_IMAGE = 0.5; // measured on the Mac mini M2 (954 images in 7.6 min)

const MODES = [
  { id: "check", title: "Check my labels", text: "A second opinion on boxes you or the datasets already have. Changes nothing: it only flags what looks wrong." },
  { id: "autolabel", title: "Draw boxes on images with none", text: "Finds drones in images nobody has labelled. The boxes are marked “not confirmed” and stay out of training until you accept them in review." },
] as const;

/** What a batch reports about the device it ran on (the Job type only knows numeric progress). */
const deviceOf = (j: Job) => (j.progress as unknown as { device?: { device: string | null; fallbacks: number; last_error: string | null } }).device;
const paramsOf = (j: Job) => j.params as unknown as AiJobParams;
const scopeOf = (p: AiJobParams) => (p.dataset ? `dataset:${p.dataset}` : p.folder ? `folder:${p.folder}` : p.source ? `source:${p.source}` : "");

/** One AI run: what to do, which models, which images. Results show as tags in the Images page. */
export default function AiLabelling({ projectId, classes, onImagesChanged }: { projectId: number; classes: DetClass[]; onImagesChanged: () => void }) {
  const router = useRouter();
  const [status, setStatus] = useState<AiServiceStatus | null>(null);
  const [models, setModels] = useState<AiModels | null>(null);
  const [folders, setFolders] = useState<FolderRow[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const initial = useRef(loadGalleryFilters(projectId)); // start from what the Images page is showing; every choice is visible below
  const [mode, setMode] = useState<"check" | "autolabel">("check");
  const [detector, setDetector] = useState(""); // "" = the default
  const [sam, setSam] = useState<"" | AiSamSize>("");
  const [minScore, setMinScore] = useState(0.5);
  const [scope, setScope] = useState(initial.current.scope);
  const [classId, setClassId] = useState<number | "">(initial.current.classId ?? "");
  const [selection, setSelection] = useState<number[] | null>(null); // images picked on the Images page: they replace the place and class below
  const [pendingOnly, setPendingOnly] = useState(false);
  const [limit, setLimit] = useState(5000);
  const [force, setForce] = useState(false);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const lastDone = useRef(new Map<number, number>());
  const picker = useRef<HTMLInputElement>(null);

  const autolabel = mode === "autolabel";
  // The request both the estimate and the run use. Auto-labelling only ever takes pending images, so that box is ticked and locked for it.
  const params: AiJobParams = useMemo(
    () => ({
      mode,
      ...(detector ? { detector } : {}),
      ...(sam ? { sam } : {}),
      ...(autolabel ? { min_score: minScore } : {}),
      ...(selection ? { image_ids: selection } : { class_id: classId === "" ? undefined : classId, ...scopeFilter(scope) }),
      review_status: autolabel || pendingOnly ? "pending" : undefined,
      limit,
      force: !autolabel && force,
    }),
    [mode, detector, sam, minScore, classId, scope, pendingOnly, limit, force, autolabel, selection],
  );

  const refreshJobs = useCallback(async () => {
    try {
      const next = await api.listJobs(projectId, "ai_check");
      setJobs(next);
      let changed = false;
      for (const j of next) {
        const done = Number(j.progress.done ?? 0);
        if (lastDone.current.has(j.id) && lastDone.current.get(j.id) !== done) changed = true;
        lastDone.current.set(j.id, done);
      }
      if (changed) onImagesChanged(); // the gallery's tags have moved on
    } catch (e) {
      setError((e as Error).message);
    }
  }, [projectId, onImagesChanged]);
  const loadModels = useCallback(() => api.aiModels().then(setModels).catch((e) => setError(e.message)), []);

  useEffect(() => {
    const picked = takeAiSelection(projectId); // read once; set only if there was one (dev mode runs effects twice)
    if (picked) setSelection(picked);
  }, [projectId]);
  useEffect(() => {
    api.aiStatus().then(setStatus).catch((e) => setError(e.message));
    api.folders(projectId).then(setFolders).catch((e) => setError(e.message));
    loadModels();
    refreshJobs();
  }, [projectId, loadModels, refreshJobs]);

  const anyActive = jobs.some((j) => ACTIVE.includes(j.status));
  useEffect(() => {
    if (!anyActive) return;
    const t = setInterval(refreshJobs, 2000);
    return () => clearInterval(t);
  }, [anyActive, refreshJobs]);

  // How many images this would take, asked of the server (the same selection the run uses) a moment after the last change.
  useEffect(() => {
    setCount(null);
    const t = setTimeout(() => api.aiPreview(projectId, params).then((r) => setCount(r.count)).catch(() => setCount(null)), 300);
    return () => clearTimeout(t);
  }, [projectId, params]);

  const places = useMemo(() => {
    const sum = (rows: FolderRow[]) => rows.reduce((n, r) => n + r.total, 0);
    const by = <K extends string>(rows: FolderRow[], key: (r: FolderRow) => K | null) => {
      const m = new Map<K, FolderRow[]>();
      for (const r of rows) {
        const k = key(r);
        if (k) m.set(k, [...(m.get(k) ?? []), r]);
      }
      return [...m].map(([k, rs]) => ({ value: k, total: sum(rs) }));
    };
    return {
      sources: by(folders.filter((r) => !r.folder || r.source !== "imported"), (r) => r.source).map((s) => ({ ...s, value: `source:${s.value}` })),
      folders: by(folders.filter((r) => r.source === "uploaded" && r.folder), (r) => r.folder).map((s) => ({ ...s, value: `folder:${s.value}` })),
      datasets: by(folders.filter((r) => r.source === "imported" && r.folder), (r) => r.folder).map((s) => ({ ...s, value: `dataset:${s.value}` })),
    };
  }, [folders]);

  const detectorLabel = (id: string | undefined) => (id ? models?.detectors.find((d) => d.id === id)?.label ?? id : models?.default_detector ?? "default detector");

  async function start() {
    setStarting(true);
    setError(null);
    try {
      await api.createAiJob(projectId, params);
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function cancel(id: number) {
    try {
      await api.cancelJob(id);
      await refreshJobs();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function upload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError(null);
    setNotice(null);
    try {
      const d: AiDetector = await api.aiUploadModel(file);
      await loadModels();
      setDetector(d.id);
      setNotice(`${d.label} stored. It is only opened when a run uses it.`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
      if (picker.current) picker.current.value = "";
    }
  }

  /** Open the images this run was about, with the filter that shows its results. */
  function viewResults(j: Job) {
    const p = paramsOf(j);
    saveGalleryFilters(projectId, { ...NO_FILTERS, scope: scopeOf(p), classId: p.class_id ?? undefined, ai: p.mode === "autolabel" ? "autolabelled" : "flagged" });
    router.push(`/projects/${projectId}/images`);
  }

  const minutes = count === null ? null : Math.max(1, Math.round((Math.min(count, limit) * SECONDS_PER_IMAGE) / 60));
  const canRun = !!status?.available && !anyActive && !starting && (count ?? 0) > 0;

  return (
    <div className="space-y-5" data-testid="ai-panel">
      <p className="text-sm">
        {status &&
          (status.available ? (
            <span className="text-green-600" data-testid="ai-available">
              Ready on this Mac{status.device ? ` · ${status.device}` : ""}
            </span>
          ) : (
            <span className="text-amber-600" data-testid="ai-unavailable">
              {status.reason}
            </span>
          ))}
      </p>

      <section className="card space-y-3">
        <h2 className="font-medium">1 · What should the AI do?</h2>
        <div className="grid gap-3 sm:grid-cols-2" role="radiogroup">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="radio"
              aria-checked={mode === m.id}
              onClick={() => setMode(m.id)}
              className={`rounded-lg border p-3 text-left text-sm transition-colors ${mode === m.id ? "border-blue-500 bg-blue-500/10" : "border-neutral-300 hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-900"}`}
            >
              <span className="block font-medium">{m.title}</span>
              <span className="text-neutral-500">{m.text}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="card space-y-3">
        <h2 className="font-medium">2 · Which models?</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-sm">
            <span className="text-neutral-500">Detector (finds the drones)</span>
            <select className="input w-full" value={detector} onChange={(e) => setDetector(e.target.value)} aria-label="Detector">
              <option value="">Default: {models?.default_detector ?? "none yet"}</option>
              {models?.detectors.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                  {d.map50 != null ? ` · mAP50 ${d.map50.toFixed(2)}` : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-neutral-500">SAM size (tightens the boxes)</span>
            <select className="input w-full" value={sam} onChange={(e) => setSam(e.target.value as "" | AiSamSize)} aria-label="SAM size">
              <option value="">Default: {models?.default_sam ?? "…"}</option>
              {models?.sams.map((s) => (
                <option key={s.size} value={s.size}>
                  {s.hint}
                  {s.downloaded ? "" : " (downloads on first use)"}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <button className="btn" onClick={() => picker.current?.click()} disabled={uploading}>
            {uploading ? "Uploading…" : "Upload a .pt model"}
          </button>
          <input ref={picker} type="file" accept=".pt" hidden onChange={(e) => upload(e.target.files?.[0])} data-testid="model-input" />
          <span className="text-neutral-500">Only upload weights you trust: a .pt file can run code on this Mac when it is loaded.</span>
        </div>
        {autolabel && (
          <label className="flex items-center gap-2 text-sm">
            Draw a box when the detector is at least
            <select className="input" value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} aria-label="Least sure detection">
              {[0.4, 0.5, 0.6, 0.7, 0.8].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </select>
            sure
          </label>
        )}
        {notice && <p className="text-sm text-green-600">{notice}</p>}
      </section>

      <section className="card space-y-3">
        <h2 className="font-medium">3 · Which images?</h2>
        {selection && (
          <p className="flex flex-wrap items-center gap-3 rounded border border-blue-500/50 bg-blue-500/10 px-3 py-2 text-sm" data-testid="ai-selection">
            <span>
              {selection.length} image{selection.length === 1 ? "" : "s"} picked on the Images page. The rules still apply: {autolabel ? "only images that are not reviewed and have no boxes are drawn on" : "images in hard-negative classes are skipped"}.
            </span>
            <button className="btn" onClick={() => setSelection(null)}>
              Use a place and class instead
            </button>
          </p>
        )}
        <div className={`flex flex-wrap gap-3 ${selection ? "hidden" : ""}`}>
          <select className="input" value={scope} onChange={(e) => setScope(e.target.value)} aria-label="Where">
            <option value="">All images</option>
            {places.folders.length > 0 && (
              <optgroup label="Uploaded folders">
                {places.folders.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.value.slice("folder:".length)} ({p.total.toLocaleString()})
                  </option>
                ))}
              </optgroup>
            )}
            {places.datasets.length > 0 && (
              <optgroup label="Imported datasets">
                {places.datasets.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.value.slice("dataset:".length)} ({p.total.toLocaleString()})
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="By source">
              {places.sources.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.value.slice("source:".length)} ({p.total.toLocaleString()})
                </option>
              ))}
            </optgroup>
          </select>
          <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))} aria-label="Class">
            <option value="">All classes</option>
            {classes.filter((c) => !c.is_negative).map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-sm">
            <input type="checkbox" checked={autolabel || pendingOnly} disabled={autolabel} onChange={(e) => setPendingOnly(e.target.checked)} />
            only images not reviewed yet{autolabel ? " (always, for drawing boxes)" : ""}
          </label>
          {!autolabel && (
            <label className="flex items-center gap-1 text-sm">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              also redo images already checked
            </label>
          )}
          <label className="flex items-center gap-1 text-sm">
            at most
            <input className="input w-24" type="number" min={1} max={20000} value={limit} onChange={(e) => setLimit(Math.max(1, Math.min(20000, Number(e.target.value) || 1)))} />
          </label>
        </div>
        <p className="text-sm text-neutral-500" data-testid="ai-estimate">
          {count === null ? "Counting…" : count === 0 ? (autolabel ? "No unlabelled, not-reviewed images here." : "No images match.") : `${Math.min(count, limit).toLocaleString()} images · about ${minutes} min on this Mac`}
        </p>
      </section>

      <div className="flex flex-wrap items-center gap-3">
        <button className="btn bg-blue-600 text-white hover:bg-blue-700" onClick={start} disabled={!canRun} data-testid="ai-start">
          {autolabel ? "Draw boxes" : "Run AI check"}
          {count ? ` on ${Math.min(count, limit).toLocaleString()} images` : ""}
        </button>
        {anyActive && <span className="text-sm text-neutral-500">A run is already going: wait for it or cancel it below.</span>}
      </div>
      {status && status.fallbacks > 0 && (
        <p className="text-sm text-amber-600" data-testid="ai-fallbacks">
          {status.fallbacks} calls the GPU refused were redone on the CPU{status.last_fallback ? ` (last: ${status.last_fallback})` : ""}. If runs are slow, this is why.
        </p>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {jobs.length > 0 && <h2 className="font-medium">Runs</h2>}
      <ul className="space-y-2">
        {jobs.map((j) => {
          const p = paramsOf(j);
          const total = Number(j.progress.total ?? 0);
          const done = Number(j.progress.done ?? 0);
          const draws = p.mode === "autolabel";
          return (
            <li key={j.id} className="card text-sm" data-testid={`ai-job-${j.id}`}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">#{j.id}</span>
                <span>{draws ? "Draw boxes" : "Check labels"}</span>
                <span className={STATUS_STYLE[j.status]}>
                  {j.status}
                  {j.status === "running" && j.cancel_requested ? " (stopping…)" : ""}
                </span>
                <span className="text-neutral-500">
                  {detectorLabel(p.detector)} · SAM {p.sam ? models?.sams.find((s) => s.size === p.sam)?.hint.split(":")[0] : models?.default_sam} ·{" "}
                  {p.image_ids ? `${p.image_ids.length} picked images` : scopeOf(p) || "all images"}
                </span>
                <span className="ml-auto flex gap-2">
                  {done > 0 && (
                    <button className="btn" onClick={() => viewResults(j)} title={p.image_ids ? "Opens every image with this kind of result, not only the ones you picked" : undefined}>
                      {p.image_ids ? "Open images with these results" : "View results"}
                    </button>
                  )}
                  {ACTIVE.includes(j.status) && !j.cancel_requested && (
                    <button className="btn" onClick={() => cancel(j.id)}>
                      Cancel
                    </button>
                  )}
                </span>
              </div>
              {total > 0 && (
                <div className="mt-2 h-1.5 overflow-hidden rounded bg-neutral-200 dark:bg-neutral-800">
                  <div className="h-full bg-green-600 transition-all" style={{ width: `${Math.min(100, (done / total) * 100)}%` }} />
                </div>
              )}
              <p className="mt-1 text-neutral-500">
                {done} of {total} images ·{" "}
                {draws
                  ? `${Number(j.progress.boxes ?? 0)} boxes on ${Number(j.progress.labelled ?? 0)} images · ${Number(j.progress.nothing ?? 0)} with nothing found · ${Number(j.progress.unmapped ?? 0)} detections of unknown classes left out · ${Number(j.progress.skipped ?? 0)} changed meanwhile`
                  : `${Number(j.progress.flagged ?? 0)} need a look · ${Number(j.progress.skipped ?? 0)} already checked`}{" "}
                · {Number(j.progress.failed ?? 0)} failed
              </p>
              {j.progress.last_error && <p className="text-amber-600">{String(j.progress.last_error)}</p>}
              {(deviceOf(j)?.fallbacks ?? 0) > 0 && <p className="text-amber-600">GPU fallbacks: {deviceOf(j)?.fallbacks} ({deviceOf(j)?.last_error})</p>}
              {j.error && <p className="mt-1 text-red-600">{j.error}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
