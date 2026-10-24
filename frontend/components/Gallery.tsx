"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { SELECTABLE_SOURCES, api, imageUrl, type DetClass, type FolderRow, type ImageRow, type ImportedDataset, type ReviewStatus } from "@/lib/api";
import { setAiSelection } from "@/lib/aiSelection";
import { useSelection } from "@/lib/useSelection";
import { AI_FILTERS, NO_FILTERS, imageQuery, loadGalleryFilters, saveGalleryFilters, scopeFilter, type GalleryFilters } from "@/lib/galleryFilters";

const PAGE = 48;
const scopeLabel = (s: string) => (s.startsWith("folder:") ? `folder ${s.slice(7)}` : s.startsWith("dataset:") ? s.slice(8) : s.slice(7));
const STATUS_STYLE = { pending: "text-amber-500", reviewed: "text-green-500", rejected: "text-red-500" } as const;
// "pending" is shown as "not reviewed": it means nobody has looked at the image yet.
const REVIEW_FILTERS: { value: Exclude<ReviewStatus, "rejected">; label: string }[] = [
  { value: "reviewed", label: "Reviewed" },
  { value: "pending", label: "Not reviewed" },
];

// How the AI check shows on a card: [text, colour]. Mirrors the manual-check tag.
const AI_TAG: Record<string, [string, string]> = {
  unchecked: ["AI: not checked", "text-neutral-500"],
  ok: ["AI ✓", "text-green-500"],
  none: ["AI: nothing found", "text-neutral-500"],
  stale: ["AI: out of date", "text-neutral-500"],
  too_small: ["AI: too small to judge", "text-neutral-500"],
  adjust: ["AI: needs a look", "text-amber-500"],
  suspect: ["AI: needs a look", "text-amber-500"],
  missing: ["AI: needs a look", "text-amber-500"],
  dismissed: ["AI: dismissed", "text-neutral-500"], // flagged, but a person reviewed those boxes
};

export default function Gallery({ projectId, classes, refreshKey, onChanged }: { projectId: number; classes: DetClass[]; refreshKey: number; onChanged: () => void }) {
  // These filters are also the review screen's: it opens the queue they describe, so they are kept for the tab.
  const [filters, setFilters] = useState<GalleryFilters>(() => loadGalleryFilters(projectId));
  const { classId, boxClassId, scope, reviewStatus, manualCheck, ai } = filters;
  const { dataset } = scopeFilter(scope);
  const update = (patch: Partial<GalleryFilters>) => setFilters((f) => ({ ...f, ...patch }));
  const [datasets, setDatasets] = useState<ImportedDataset[]>([]);
  const [folders, setFolders] = useState<FolderRow[]>([]);
  const [images, setImages] = useState<ImageRow[]>([]);
  const { selected, toggle, selectAll, clear: clearSelection, max: MAX_SELECTED } = useSelection(images, filters);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);
  const router = useRouter();
  const [rejecting, setRejecting] = useState(false); // the inline "are you sure" for a bulk reject
  const [bulkBusy, setBulkBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const className = (id: number) => classes.find((c) => c.id === id)?.name ?? id;

  useEffect(() => {
    saveGalleryFilters(projectId, filters);
  }, [projectId, filters]);
  useEffect(() => {
    setRejecting(false);
    setNotice(null);
  }, [filters]);

  const load = useCallback(
    async (offset: number, limit = PAGE) => {
      try {
        const rows = await api.listImages(projectId, { ...imageQuery(filters), limit, offset });
        setImages((prev) => (offset === 0 ? rows : [...prev, ...rows]));
        setMore(rows.length === limit);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [projectId, filters],
  );
  useEffect(() => {
    load(0);
  }, [load, refreshKey]);
  // Coming back to this tab (the review screen may have been opened in another one) shows the list as it is now, with as many rows as are loaded.
  const loaded = images.length;
  useEffect(() => {
    const onBack = () => {
      if (document.visibilityState === "visible") load(0, Math.min(500, Math.max(1, Math.ceil(loaded / PAGE)) * PAGE));
    };
    document.addEventListener("visibilitychange", onBack);
    window.addEventListener("focus", onBack);
    return () => {
      document.removeEventListener("visibilitychange", onBack);
      window.removeEventListener("focus", onBack);
    };
  }, [load, loaded]);
  useEffect(() => {
    api.importedDatasets(projectId).then(setDatasets).catch((e) => setError(e.message));
    api.folders(projectId).then(setFolders).catch((e) => setError(e.message));
  }, [projectId, refreshKey]);
  // One entry per upload folder, however many classes it was filed under.
  const uploadFolders = [...folders.filter((f) => f.source === "uploaded" && f.folder).reduce((m, f) => m.set(f.folder!, (m.get(f.folder!) ?? 0) + f.total), new Map<string, number>())];


  async function rejectSelected() {
    setBulkBusy(true);
    try {
      await api.rejectImages(projectId, [...selected]);
      clearSelection();
      setError(null);
      onChanged(); // reloads the list and the counts
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRejecting(false);
      setBulkBusy(false);
    }
  }

  /** Keep the picked images although the AI flagged them: they leave "AI: needs a look" (the review screen only shows hard-negative pictures). */
  async function keepSelected() {
    setBulkBusy(true);
    try {
      const { dismissed } = await api.dismissAi(projectId, [...selected]);
      const none = selected.size - dismissed;
      setNotice(`Kept ${dismissed} image(s) as they are${none > 0 ? `; ${none} had nothing flagged` : ""}. Filter "AI: dismissed" to see them again.`);
      clearSelection();
      setError(null);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBulkBusy(false);
    }
  }

  function runAiOnSelected() {
    setAiSelection(projectId, [...selected]);
    router.push(`/projects/${projectId}/ai`);
  }

  // The active filters as removable chips (the selects above stay the way to change them).
  const chips: { key: string; label: string; clear: () => void }[] = [];
  if (classId !== undefined) chips.push({ key: "class", label: String(className(classId)), clear: () => update({ classId: undefined }) });
  if (boxClassId !== undefined) chips.push({ key: "box", label: `${className(boxClassId)} boxes`, clear: () => update({ boxClassId: undefined }) });
  if (scope) chips.push({ key: "scope", label: scopeLabel(scope), clear: () => update({ scope: "", manualCheck: undefined }) }); // the manual-check filter only exists inside one dataset
  if (manualCheck !== undefined) chips.push({ key: "manual", label: manualCheck ? "manual check ✓" : "not checked yet", clear: () => update({ manualCheck: undefined }) });
  if (ai) chips.push({ key: "ai", label: AI_FILTERS.find((f) => f.value === ai)?.label ?? ai, clear: () => update({ ai: undefined }) });
  if (reviewStatus) chips.push({ key: "status", label: REVIEW_FILTERS.find((f) => f.value === reviewStatus)?.label ?? reviewStatus, clear: () => update({ reviewStatus: undefined }) });

  async function remove(id: number) {
    try {
      await api.deleteImage(id);
      setConfirming(null);
      setImages((prev) => prev.filter((i) => i.id !== id));
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <select className="input" value={classId ?? ""} onChange={(e) => update({ classId: e.target.value === "" ? undefined : Number(e.target.value) })}>
          <option value="">All classes</option>
          {classes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        {/* The manual-check filter only exists inside one imported dataset, so changing the scope clears it. */}
        <select className="input" value={scope} onChange={(e) => update({ scope: e.target.value, manualCheck: undefined })} aria-label="Source">
          <option value="">All sources</option>
          {SELECTABLE_SOURCES.map((s) => (
            <option key={s} value={`source:${s}`}>
              {s}
            </option>
          ))}
          {uploadFolders.length > 0 && (
            <optgroup label="Uploaded folders">
              {uploadFolders.map(([name, total]) => (
                <option key={name} value={`folder:${name}`}>
                  {name} ({total.toLocaleString()})
                </option>
              ))}
            </optgroup>
          )}
          {datasets.length > 0 && (
            <optgroup label="Imported datasets">
              {datasets.map((d) => (
                <option key={d.name} value={`dataset:${d.name}`}>
                  {d.name} ({d.checked.toLocaleString()} of {d.total.toLocaleString()} checked)
                </option>
              ))}
            </optgroup>
          )}
        </select>
        {dataset && (
          <select
            className="input"
            value={manualCheck === undefined ? "" : String(manualCheck)}
            onChange={(e) => update({ manualCheck: e.target.value === "" ? undefined : e.target.value === "true" })}
            aria-label="Manual check"
          >
            <option value="">Any manual check</option>
            <option value="true">Manual check ✓</option>
            <option value="false">Not checked yet</option>
          </select>
        )}
        <select className="input" value={ai ?? ""} onChange={(e) => update({ ai: e.target.value || undefined })} aria-label="AI check">
          <option value="">Any AI check</option>
          {AI_FILTERS.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </select>
        <select
          className="input"
          value={reviewStatus ?? ""}
          onChange={(e) => update({ reviewStatus: (e.target.value || undefined) as GalleryFilters["reviewStatus"] })}
          aria-label="Review status"
        >
          <option value="">All images</option>
          {REVIEW_FILTERS.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </select>
      </div>
      <div className="flex flex-wrap items-center gap-2" data-testid="filter-chips">
        {chips.map((c) => (
          <span key={c.key} className="inline-flex items-center gap-1 rounded-full border border-blue-500/50 bg-blue-500/10 py-0.5 pl-2.5 pr-1 text-xs">
            {c.label}
            <button className="rounded-full px-1 text-neutral-500 hover:text-current" onClick={c.clear} aria-label={`Remove filter ${c.label}`}>
              ✕
            </button>
          </span>
        ))}
        {chips.length > 0 && (
          <button className="text-xs text-neutral-500 hover:underline" onClick={() => setFilters(NO_FILTERS)}>
            Clear filters
          </button>
        )}
        {/* The review screen opens the queue these filters describe. */}
        {images.length > 0 && (
          <Link href={`/projects/${projectId}/review`} className="btn ml-auto" data-testid="review-these">
            Review these →
          </Link>
        )}
      </div>
      {selected.size > 0 && (
        <div className="sticky top-2 z-10 flex flex-wrap items-center gap-2 rounded-lg border border-blue-500/50 bg-white/95 px-3 py-2 text-sm shadow dark:bg-neutral-900/95" data-testid="bulk-bar">
          <span className="font-medium">{selected.size} selected</span>
          <button className="btn" onClick={selectAll}>
            Select {images.length > MAX_SELECTED ? `first ${MAX_SELECTED}` : `all ${images.length}`} shown
          </button>
          {rejecting ? (
            <>
              <span className="text-red-500">Reject {selected.size} image(s)? Their boxes stay as they are.</span>
              <button className="btn bg-red-600 text-white hover:bg-red-700" onClick={rejectSelected} disabled={bulkBusy} data-testid="confirm-reject">
                Confirm reject
              </button>
              <button className="btn" onClick={() => setRejecting(false)}>
                Cancel
              </button>
            </>
          ) : (
            <>
              <button className="btn" onClick={() => setRejecting(true)} data-testid="bulk-reject">
                Reject…
              </button>
              <button className="btn" onClick={runAiOnSelected} data-testid="bulk-ai">
                Run AI on these
              </button>
              <button className="btn" onClick={keepSelected} disabled={bulkBusy} data-testid="bulk-keep" title="Set the AI's doubts about these images aside: they leave “AI: needs a look”. Nothing else changes.">
                Keep as is
              </button>
            </>
          )}
          <button className="ml-auto text-neutral-500 hover:text-current" onClick={clearSelection}>
            Clear selection
          </button>
        </div>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}
      {notice && (
        <p className="text-sm text-green-600" data-testid="gallery-notice">
          {notice}
        </p>
      )}
      {images.length === 0 && !error && <p className="text-sm text-neutral-500">No images match.</p>}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {images.map((img) => (
          <figure key={img.id} className={`group relative overflow-hidden rounded border ${selected.has(img.id) ? "border-blue-500 ring-2 ring-blue-500" : "border-neutral-300 dark:border-neutral-700"}`}>
            <label className={`absolute left-1 top-1 items-center rounded bg-black/70 p-1 ${selected.size > 0 ? "flex" : "flex md:hidden md:group-hover:flex"}`} title="Select this image">
              <input type="checkbox" checked={selected.has(img.id)} onChange={() => toggle(img.id)} aria-label={`Select image ${img.id}`} data-testid={`select-${img.id}`} />
            </label>
            <Link href={`/projects/${projectId}/review?image=${img.id}`} title="Open in review screen">
              {/* eslint-disable-next-line @next/next/no-img-element -- served by the API, not optimizable by next/image */}
              <img src={imageUrl(img.id)} alt={`${className(img.class_id)} #${img.id}`} loading="lazy" className="aspect-square w-full object-cover" />
            </Link>
            <figcaption className="flex items-center justify-between px-2 py-1 text-xs">
              <span>
                {className(img.class_id)} · {img.dataset ?? img.folder ?? img.source}
              </span>
              <span className={STATUS_STYLE[img.review_status]}>{img.review_status === "pending" ? "not reviewed" : img.review_status}</span>
            </figcaption>
            {/* The tag only exists on imported images (null elsewhere). */}
            {img.manually_checked !== null && (
              <p className={`px-2 pb-1 text-xs ${img.manually_checked ? "text-green-500" : "text-neutral-500"}`}>
                {img.manually_checked ? "Manual check ✓" : "Manual check: not yet"}
              </p>
            )}
            <p className={`px-2 pb-1 text-xs ${AI_TAG[img.ai_status ?? "unchecked"]?.[1] ?? "text-neutral-500"}`} data-testid={`ai-tag-${img.id}`}>
              {AI_TAG[img.ai_status ?? "unchecked"]?.[0] ?? img.ai_status}
            </p>
            {img.ai_labelled && (
              <p className="px-2 pb-1 text-xs text-blue-500" data-testid={`ai-labelled-${img.id}`}>
                AI drew the boxes · not confirmed
              </p>
            )}
            {/* Sibling of the link, so clicking it never navigates. Stays visible while confirming. */}
            <div className={`absolute right-1 top-1 gap-1 ${confirming === img.id ? "flex" : "hidden group-hover:flex"}`}>
              {confirming === img.id ? (
                <>
                  <button onClick={() => remove(img.id)} className="rounded bg-red-600 px-1.5 py-0.5 text-xs text-white">
                    Confirm delete
                  </button>
                  <button onClick={() => setConfirming(null)} className="rounded bg-black/70 px-1.5 py-0.5 text-xs text-white">
                    Cancel
                  </button>
                </>
              ) : (
                <button onClick={() => setConfirming(img.id)} className="rounded bg-black/70 px-1.5 py-0.5 text-xs text-white" aria-label="Delete image">
                  Delete
                </button>
              )}
            </div>
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
