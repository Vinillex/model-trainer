"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import CloudCopies from "@/components/CloudCopies";
import { SELECTABLE_SOURCES, api, datasetDownloadUrl, parseUtc, type DataSelection, type DatasetPreview, type DatasetVersion, type DetClass } from "@/lib/api";

const SPLITS = ["train", "val", "test"] as const;
const MAX_MERGE_ROWS = 5; // the backend allows up to 5 groups; each needs two classes, so the form shows (drone classes / 2) of them
const SIMILAR_MIN = 0.5; // the backend's limits for "leave out similar pictures" (schemas.SIMILAR_PCT_MIN / MAX)
const SIMILAR_MAX = 25;

type MergeRow = { ids: Set<number>; name: string };

function VersionCard({ v, onDeleted }: { v: DatasetVersion; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const st = v.stats;
  const total = SPLITS.reduce((n, s) => n + st.images[s], 0);
  const classes = Object.keys(st.per_class);

  async function remove() {
    try {
      await api.deleteDataset(v.id);
      onDeleted();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <li className="space-y-2 rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid={`dataset-${v.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{v.name}</span>
        <span className="text-neutral-500">
          {total} images · requested split {Math.round((1 - v.config.val_ratio - v.config.test_ratio) * 100)}/{Math.round(v.config.val_ratio * 100)}/{Math.round(v.config.test_ratio * 100)} · {parseUtc(v.created_at).toLocaleString()}
        </span>
        <span className="ml-auto flex gap-2">
          <a className="btn" href={datasetDownloadUrl(v.id)} download>
            Download YOLO zip
          </a>
          {confirming ? (
            <>
              <button className="btn bg-red-700/40" onClick={remove}>
                Confirm delete
              </button>
              <button className="btn" onClick={() => setConfirming(false)}>
                Cancel
              </button>
            </>
          ) : (
            <button className="btn" onClick={() => setConfirming(true)}>
              Delete
            </button>
          )}
        </span>
      </div>

      <table className="w-full max-w-xl text-xs">
        <thead>
          <tr className="border-b border-neutral-300 text-left dark:border-neutral-700">
            <th className="py-1 pr-3">Class</th>
            {SPLITS.map((s) => (
              <th key={s} className="px-2 text-right capitalize">
                {s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => (
            <tr key={c} className="border-b border-neutral-200 dark:border-neutral-800">
              <td className="py-1 pr-3">{c}</td>
              {SPLITS.map((s) => (
                <td key={s} className="px-2 text-right tabular-nums text-neutral-500">
                  {st.per_class[c][s].images} img · {st.per_class[c][s].boxes} box
                </td>
              ))}
            </tr>
          ))}
          <tr className="font-medium">
            <td className="py-1 pr-3">All images</td>
            {SPLITS.map((s) => (
              <td key={s} className="px-2 text-right tabular-nums">
                {st.images[s]}
                {(st.background_images[s] ?? 0) > 0 && <span className="font-normal text-neutral-500"> ({st.background_images[s]} background)</span>}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      {v.config.merged && v.config.merged.length > 0 && (
        <p className="text-xs text-neutral-500" data-testid={`dataset-${v.id}-merged`}>
          Treated as one class: {v.config.merged.map((m) => `${m.classes.join(" + ")} → ${m.name}`).join("; ")}. The labels in your project are unchanged.
        </p>
      )}
      {st.by_source && Object.keys(st.by_source).length > 0 && (
        <p className="text-xs text-neutral-500" data-testid={`dataset-${v.id}-data`}>
          Data used:{" "}
          {Object.entries(st.by_source)
            .map(([k, n]) => `${k.startsWith("imported:") ? `imported (${k.slice(9)})` : k} ${n.toLocaleString()}`)
            .join(" · ")}
        </p>
      )}
      <p className="text-xs text-neutral-500">
        Left out: {st.excluded.pending} pending, {st.excluded.rejected} rejected
        {(st.excluded.not_selected ?? 0) > 0 && <>, {st.excluded.not_selected!.toLocaleString()} reviewed but not selected</>}
        {v.config.skip_similar && (
          <span data-testid={`dataset-${v.id}-similar`}>
            , {(st.excluded.similar ?? 0).toLocaleString()} similar pictures (box moved under {v.config.skip_similar.box_pct} %)
          </span>
        )}
        . Images from the same
        source page or scene always share a split.
      </p>
      {v.config.group_scenes && (
        <p className="text-xs text-neutral-500" data-testid={`dataset-${v.id}-scenes`}>
          Look-alike scenes kept in one split ({(st.scene_groups_joined ?? 0).toLocaleString()} groups joined), so no test picture is a near-copy of a training one.
        </p>
      )}
      {st.warnings.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-amber-500" data-testid="dataset-warnings">
          {st.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
      {error && <p className="text-xs text-red-600">{error}</p>}
    </li>
  );
}

export default function DatasetsPanel({ projectId, classes, refreshKey }: { projectId: number; classes: DetClass[]; refreshKey: number }) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [name, setName] = useState("");
  const [valPct, setValPct] = useState(10);
  const [testPct, setTestPct] = useState(10);
  const [negatives, setNegatives] = useState(true);
  const [off, setOff] = useState<Set<string>>(new Set()); // data the person unticked: "source:scraped", "imported:anti-uav"
  // Groups of classes this version should treat as one, each with its own (optional) new name; `rows` below shows the usable ones
  const [merges, setMerges] = useState<MergeRow[]>(() => Array.from({ length: MAX_MERGE_ROWS }, () => ({ ids: new Set<number>(), name: "" })));
  const [skipSimilar, setSkipSimilar] = useState(false);
  const [similarPct, setSimilarPct] = useState(5);
  const [groupScenes, setGroupScenes] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Everything is selected until unticked. The picker's options come from the preview itself, so the first load asks for all.
  const selection: DataSelection | undefined = preview
    ? {
        sources: SELECTABLE_SOURCES.filter((s) => !off.has(`source:${s}`)),
        imported: preview.available.imported_datasets.map((d) => d.name).filter((n) => !off.has(`imported:${n}`)),
      }
    : undefined;
  const similarOk = similarPct >= SIMILAR_MIN && similarPct <= SIMILAR_MAX;
  const similarAsked = skipSimilar && similarOk ? similarPct : undefined;
  const selectionKey = JSON.stringify([selection ?? null, similarAsked ?? null]);
  const selectionRef = useRef({ selection, similarAsked });
  selectionRef.current = { selection, similarAsked };

  const loadPreview = useCallback(() => {
    api.datasetPreview(projectId, selectionRef.current.selection, selectionRef.current.similarAsked).then(setPreview).catch((e) => setError(e.message));
  }, [projectId]);
  const loadVersions = useCallback(() => {
    api.listDatasets(projectId).then(setVersions).catch((e) => setError(e.message));
  }, [projectId]);
  useEffect(loadVersions, [loadVersions, refreshKey]);
  useEffect(loadPreview, [loadPreview, refreshKey, selectionKey]);
  const load = useCallback(() => {
    loadPreview();
    loadVersions();
  }, [loadPreview, loadVersions]);

  const toggle = (key: string) =>
    setOff((prev) => {
      const next = new Set(prev);
      if (!next.delete(key)) next.add(key);
      return next;
    });

  const updateMerge = (row: number, change: (m: MergeRow) => MergeRow) => setMerges((prev) => prev.map((m, i) => (i === row ? change(m) : m)));
  const toggleMerge = (row: number, id: number) =>
    updateMerge(row, (m) => {
      const ids = new Set(m.ids);
      if (!ids.delete(id)) ids.add(id);
      return { ...m, ids };
    });
  const drone = classes.filter((c) => !c.is_negative && !c.is_mixed);
  const rows = merges.slice(0, Math.min(MAX_MERGE_ROWS, Math.floor(drone.length / 2))); // 4 classes: 2 groups, 6: 3, ...
  const picked = rows.map((m) => drone.filter((c) => m.ids.has(c.id)));
  const nameOk = (n: string) => n.trim() === "" || /^[a-z0-9][a-z0-9_-]*$/.test(n.trim());

  async function create(e: React.FormEvent) {
    e.preventDefault();
    const problem = picked.some((p) => p.length === 1)
      ? "Tick at least two classes in a “treat as one” row, or none."
      : rows.some((m) => !nameOk(m.name))
        ? "A merged class name uses lowercase letters, digits, - and _."
        : skipSimilar && !similarOk
          ? `The distance for similar pictures must be between ${SIMILAR_MIN} and ${SIMILAR_MAX} %.`
          : null;
    if (problem) {
      setError(problem);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.createDataset(projectId, {
        name: name.trim() || undefined,
        val_ratio: valPct / 100,
        test_ratio: testPct / 100,
        include_negatives: negatives,
        sources: selection?.sources,
        imported_datasets: selection?.imported,
        merge_classes: rows
          .map((m, i) => ({ class_ids: picked[i].map((c) => c.id), name: m.name.trim() || undefined }))
          .filter((m) => m.class_ids.length >= 2),
        ...(skipSimilar ? { skip_similar: true, similar_box_pct: similarPct } : {}),
        group_scenes: groupScenes,
      });
      setName("");
      load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const num = "input w-20";
  return (
    <div className="space-y-4">
      {preview && (
        <p className="text-sm text-neutral-500" data-testid="dataset-preview">
          A new version would contain <b>{preview.reviewed_images.toLocaleString()}</b> reviewed images: {preview.reviewed_with_boxes} with boxes, {preview.reviewed_empty} with none, {preview.negative_images} hard negatives.{" "}
          {preview.not_selected > 0 && (
            <>
              <b>{preview.not_selected.toLocaleString()}</b> reviewed images are not selected below.{" "}
            </>
          )}
          {preview.pending + preview.rejected > 0 && (
            <>
              Left out: <b>{preview.pending}</b> pending (review them to include them), {preview.rejected} rejected.{" "}
            </>
          )}
          Boxes per class: {Object.entries(preview.boxes_per_class).map(([c, n]) => `${c} ${n}`).join(", ")}.
        </p>
      )}
      <form onSubmit={create} className="flex flex-wrap items-end gap-3 text-sm">
        <label>
          <span className="mb-1 block text-neutral-500">Name (optional)</span>
          <input className="input w-40" placeholder="v1, v2, …" value={name} onChange={(e) => setName(e.target.value)} maxLength={120} />
        </label>
        <label>
          <span className="mb-1 block text-neutral-500">Validation %</span>
          <input type="number" className={num} min={0} max={50} value={valPct} onChange={(e) => setValPct(Number(e.target.value))} />
        </label>
        <label>
          <span className="mb-1 block text-neutral-500">Test %</span>
          <input type="number" className={num} min={0} max={50} value={testPct} onChange={(e) => setTestPct(Number(e.target.value))} />
        </label>
        <label className="flex items-center gap-1 pb-2">
          <input type="checkbox" checked={negatives} onChange={(e) => setNegatives(e.target.checked)} /> Include hard negatives as background images
        </label>
        {preview && (
          <fieldset className="basis-full space-y-1" data-testid="data-picker">
            <legend className="mb-1 text-neutral-500">Data to include (reviewed images only)</legend>
            <div className="flex flex-wrap gap-x-5 gap-y-1">
              {SELECTABLE_SOURCES.map((src) => {
                const n = preview.available.sources[src];
                return (
                  <label key={src} className={`flex items-center gap-1 ${n === 0 ? "text-neutral-500" : ""}`}>
                    <input type="checkbox" checked={!off.has(`source:${src}`)} onChange={() => toggle(`source:${src}`)} disabled={n === 0} /> {src}{" "}
                    <span className="text-neutral-500">({n.toLocaleString()})</span>
                  </label>
                );
              })}
            </div>
            {preview.available.imported_datasets.length > 0 && (
              <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
                <span className="text-neutral-500">Imported datasets:</span>
                {preview.available.imported_datasets.map((d) => (
                  <label key={d.name} className="flex items-center gap-1">
                    <input type="checkbox" checked={!off.has(`imported:${d.name}`)} onChange={() => toggle(`imported:${d.name}`)} /> {d.name}{" "}
                    <span className="text-neutral-500">({d.reviewed.toLocaleString()})</span>
                  </label>
                ))}
              </div>
            )}
          </fieldset>
        )}
        <fieldset className="basis-full space-y-1" data-testid="scene-picker">
          <legend className="mb-1 text-neutral-500">Train / test separation</legend>
          <label className="flex items-center gap-1">
            <input type="checkbox" checked={groupScenes} onChange={(e) => setGroupScenes(e.target.checked)} data-testid="group-scenes" /> Keep look-alike scenes in
            the same split (recommended)
          </label>
          <p className="text-xs text-neutral-500">
            Frames of one video can sit in two imported datasets (cmy-uav copies Anti-UAV&apos;s videos) or in different scene groups. Without this, a test picture
            can be a near-copy of a training picture and the scores look better than they are. Your scraped pictures are only joined with each other.
          </p>
        </fieldset>
        <fieldset className="basis-full space-y-1" data-testid="similar-picker">
          <legend className="mb-1 text-neutral-500">Similar pictures (optional)</legend>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <label className="flex items-center gap-1">
              <input type="checkbox" checked={skipSimilar} onChange={(e) => setSkipSimilar(e.target.checked)} data-testid="skip-similar" /> Leave out similar pictures: same scene, drone
              box moved less than
            </label>
            <label className="flex items-center gap-1">
              <input
                type="number"
                className={`input w-20 ${skipSimilar && !similarOk ? "border-red-500" : ""}`}
                min={SIMILAR_MIN}
                max={SIMILAR_MAX}
                step={0.5}
                value={similarPct}
                onChange={(e) => setSimilarPct(Number(e.target.value))}
                disabled={!skipSimilar}
                data-testid="similar-pct"
              />
              % of the picture
            </label>
            {skipSimilar && preview?.similar != null && (
              <span className="font-medium" data-testid="similar-count">
                {preview.similar.toLocaleString()} pictures would be left out
              </span>
            )}
          </div>
          <p className="text-xs text-neutral-500">
            One picture of each run of repeats is kept (the first). Compared by the picture&apos;s fingerprint and where its boxes are, so the same scene with the
            drone somewhere else stays in. Pictures without boxes are always kept. Nothing in your project changes.
          </p>
        </fieldset>
        {drone.length >= 2 && (
          <fieldset className="basis-full space-y-1" data-testid="merge-picker">
            <legend className="mb-1 text-neutral-500">Treat classes as one (optional)</legend>
            {rows.map((m, row) => (
              <div key={row} className="flex flex-wrap items-center gap-x-5 gap-y-1" data-testid={`merge-row-${row}`}>
                <span className="w-16 text-neutral-500">Group {row + 1}</span>
                {drone.map((c) => {
                  const elsewhere = rows.some((o, j) => j !== row && o.ids.has(c.id)); // a class can be in one group only
                  return (
                    <label key={c.id} className={`flex items-center gap-1 ${elsewhere ? "text-neutral-500" : ""}`}>
                      <input type="checkbox" checked={m.ids.has(c.id)} disabled={elsewhere} onChange={() => toggleMerge(row, c.id)} data-testid={`merge-${row}-${c.name}`} /> {c.name}
                    </label>
                  );
                })}
                {picked[row].length >= 2 && (
                  <label className="flex items-center gap-1">
                    <span className="text-neutral-500">new label</span>
                    <input
                      className={`input w-44 ${nameOk(m.name) ? "" : "border-red-500"}`}
                      placeholder={picked[row].map((c) => c.name).join("+")}
                      value={m.name}
                      onChange={(e) => updateMerge(row, (x) => ({ ...x, name: e.target.value }))}
                      maxLength={64}
                      data-testid={`merge-name-${row}`}
                    />
                  </label>
                )}
              </div>
            ))}
            <p className="text-xs text-neutral-500">
              Each group trains as one class under its new label, e.g. multirotor + interceptor as “rotor” and fixed-wing + delta-wing as “wing”. Your labels
              and boxes are not changed; make another version without it to compare.
            </p>
          </fieldset>
        )}
        <button className="btn" disabled={busy || (preview?.reviewed_images ?? 0) === 0}>
          {busy ? "Creating…" : "Create dataset version"}
        </button>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {/* Shown only with a bucket set up. Reloads when a version is created or deleted (deleting one also deletes its copy). */}
      <CloudCopies projectId={projectId} refreshKey={refreshKey + versions.length} />
      <ul className="space-y-3">
        {versions.map((v) => (
          <VersionCard key={v.id} v={v} onDeleted={load} />
        ))}
      </ul>
    </div>
  );
}
