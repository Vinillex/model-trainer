"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, imageUrl, type DetClass, type FolderRow, type ImageRow } from "@/lib/api";
import { NO_FILTERS, saveGalleryFilters, scopeFilter } from "@/lib/galleryFilters";
import { useSelection } from "@/lib/useSelection";

const PAGE = 48;

type Step = "idle" | "confirm" | "versions"; // deleting for good: the plain "are you sure", then (only if needed) the dataset-version warning

/** Every rejected image of the project, wherever it came from. Restore puts it back as "not reviewed"; Delete forever removes the file. */
export default function TrashPanel({ projectId, classes, refreshKey, onChanged }: { projectId: number; classes: DetClass[]; refreshKey: number; onChanged: () => void }) {
  const [classId, setClassId] = useState<number | "">("");
  const [scope, setScope] = useState("");
  const [rows, setRows] = useState<FolderRow[]>([]);
  const [images, setImages] = useState<ImageRow[]>([]);
  const [more, setMore] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<Step>("idle");
  const [versionWarning, setVersionWarning] = useState("");
  const { selected, toggle, selectAll, clear, max } = useSelection(images, `${classId}|${scope}`);
  const className = (id: number) => classes.find((c) => c.id === id)?.name ?? id;
  useEffect(() => setStep("idle"), [classId, scope]);

  const load = useCallback(
    async (offset: number) => {
      try {
        const page = await api.listImages(projectId, { reviewStatus: "rejected", classId: classId === "" ? undefined : classId, ...scopeFilter(scope), limit: PAGE, offset });
        setImages((prev) => (offset === 0 ? page : [...prev, ...page]));
        setMore(page.length === PAGE);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoaded(true);
      }
    },
    [projectId, classId, scope],
  );
  useEffect(() => {
    load(0);
  }, [load, refreshKey]);
  useEffect(() => {
    api.folders(projectId).then(setRows).catch((e) => setError(e.message));
  }, [projectId, refreshKey]);

  // The places the rejected images came from, with how many each.
  const places = useMemo(() => {
    const byKey = new Map<string, number>();
    for (const r of rows) {
      if (r.rejected === 0) continue;
      const key = r.source === "imported" && r.folder ? `dataset:${r.folder}` : r.folder ? `folder:${r.folder}` : `source:${r.source}`;
      byKey.set(key, (byKey.get(key) ?? 0) + r.rejected);
    }
    return [...byKey].map(([value, n]) => ({ value, n, label: value.slice(value.indexOf(":") + 1) })).sort((a, b) => b.n - a.n);
  }, [rows]);
  const inTrash = rows.reduce((n, r) => n + r.rejected, 0);

  async function restore() {
    setBusy(true);
    try {
      await api.restoreImages(projectId, [...selected]);
      clear();
      setError(null);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function purge(allowBreakingVersions: boolean) {
    setBusy(true);
    try {
      await api.purgeImages(projectId, [...selected], allowBreakingVersions);
      clear();
      setStep("idle");
      setError(null);
      onChanged();
    } catch (e) {
      const err = e as Error & { status?: number };
      if (err.status === 409) {
        setVersionWarning(err.message); // a dataset version contains some of them: say so and let the person decide
        setStep("versions");
      } else {
        setError(err.message);
        setStep("idle");
      }
    } finally {
      setBusy(false);
    }
  }

  /** Opening a card reviews the trash: the review screen shows the queue these filters describe. */
  const openFilters = () => saveGalleryFilters(projectId, { ...NO_FILTERS, scope, classId: classId === "" ? undefined : classId, reviewStatus: "rejected" });

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <select className="input" value={scope} onChange={(e) => setScope(e.target.value)} aria-label="Where from">
          <option value="">All places ({inTrash.toLocaleString()})</option>
          {places.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label} ({p.n.toLocaleString()})
            </option>
          ))}
        </select>
        <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))} aria-label="Class">
          <option value="">All classes</option>
          {classes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        <span className="text-sm text-neutral-500" data-testid="trash-count">
          {inTrash.toLocaleString()} rejected image{inTrash === 1 ? "" : "s"} in the trash
        </span>
      </div>

      {selected.size > 0 && (
        <div className="sticky top-2 z-10 space-y-2 rounded-lg border border-blue-500/50 bg-white/95 px-3 py-2 text-sm shadow dark:bg-neutral-900/95" data-testid="trash-bar">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{selected.size} selected</span>
            <button className="btn" onClick={selectAll}>
              Select {images.length > max ? `first ${max}` : `all ${images.length}`} shown
            </button>
            {step === "idle" && (
              <>
                <button className="btn" onClick={restore} disabled={busy} data-testid="trash-restore">
                  Restore
                </button>
                <button className="btn" onClick={() => setStep("confirm")} data-testid="trash-delete">
                  Delete forever…
                </button>
              </>
            )}
            <button className="ml-auto text-neutral-500 hover:text-current" onClick={clear}>
              Clear selection
            </button>
          </div>
          {step === "confirm" && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-red-500">
                Delete {selected.size === 1 ? "this image and its file" : `${selected.size} images and their files`} for good? This cannot be undone.
              </span>
              <button className="btn bg-red-600 text-white hover:bg-red-700" onClick={() => purge(false)} disabled={busy} data-testid="trash-confirm-delete">
                Delete forever
              </button>
              <button className="btn" onClick={() => setStep("idle")}>
                Cancel
              </button>
            </div>
          )}
          {step === "versions" && (
            <div className="space-y-2" data-testid="trash-versions-warning">
              <p className="text-amber-500">{versionWarning}</p>
              <div className="flex gap-2">
                <button className="btn bg-red-600 text-white hover:bg-red-700" onClick={() => purge(true)} disabled={busy} data-testid="trash-delete-anyway">
                  Delete anyway
                </button>
                <button className="btn" onClick={() => setStep("idle")}>
                  Keep them
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}
      {loaded && images.length === 0 && !error && (
        <p className="rounded border border-dashed border-neutral-400 p-6 text-center text-sm text-neutral-500 dark:border-neutral-600" data-testid="trash-empty">
          {inTrash === 0 ? "The trash is empty. Images you reject in the review screen or on the Images page end up here." : "No rejected images match."}
        </p>
      )}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {images.map((img) => (
          <figure key={img.id} className={`group relative overflow-hidden rounded border ${selected.has(img.id) ? "border-blue-500 ring-2 ring-blue-500" : "border-neutral-300 dark:border-neutral-700"}`} data-testid={`trash-card-${img.id}`}>
            <label className={`absolute left-1 top-1 items-center rounded bg-black/70 p-1 ${selected.size > 0 ? "flex" : "flex md:hidden md:group-hover:flex"}`} title="Select this image">
              <input type="checkbox" checked={selected.has(img.id)} onChange={() => toggle(img.id)} aria-label={`Select image ${img.id}`} data-testid={`trash-select-${img.id}`} />
            </label>
            <Link href={`/projects/${projectId}/review?image=${img.id}`} onClick={openFilters} title="Open in the review screen">
              {/* eslint-disable-next-line @next/next/no-img-element -- served by the API, not optimizable by next/image */}
              <img src={imageUrl(img.id)} alt={`${className(img.class_id)} #${img.id}`} loading="lazy" className="aspect-square w-full object-cover opacity-80" />
            </Link>
            <figcaption className="px-2 py-1 text-xs text-neutral-500">
              {className(img.class_id)} · {img.dataset ?? img.folder ?? img.source}
            </figcaption>
          </figure>
        ))}
      </div>
      {more && (
        <button className="btn" onClick={() => load(images.length)}>
          Load more
        </button>
      )}
    </div>
  );
}
