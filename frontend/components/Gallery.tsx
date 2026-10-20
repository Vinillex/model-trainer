"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { SOURCES, api, imageUrl, type DetClass, type ImageRow, type ReviewStatus, type Source } from "@/lib/api";

const PAGE = 48;
const STATUS_STYLE = { pending: "text-amber-500", reviewed: "text-green-500", rejected: "text-red-500" } as const;
// "pending" is shown as "not reviewed": it means nobody has looked at the image yet.
const REVIEW_FILTERS: { value: ReviewStatus; label: string }[] = [
  { value: "reviewed", label: "Reviewed" },
  { value: "pending", label: "Not reviewed" },
  { value: "rejected", label: "Rejected" },
];

export default function Gallery({ projectId, classes, refreshKey, onChanged }: { projectId: number; classes: DetClass[]; refreshKey: number; onChanged: () => void }) {
  const [classId, setClassId] = useState<number | undefined>();
  const [source, setSource] = useState<Source | undefined>();
  const [reviewStatus, setReviewStatus] = useState<ReviewStatus | undefined>();
  const [images, setImages] = useState<ImageRow[]>([]);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);
  const className = (id: number) => classes.find((c) => c.id === id)?.name ?? id;

  const load = useCallback(
    async (offset: number) => {
      try {
        const rows = await api.listImages(projectId, { classId, source, reviewStatus, limit: PAGE, offset });
        setImages((prev) => (offset === 0 ? rows : [...prev, ...rows]));
        setMore(rows.length === PAGE);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [projectId, classId, source, reviewStatus],
  );
  useEffect(() => {
    load(0);
  }, [load, refreshKey]);

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
        <select className="input" value={classId ?? ""} onChange={(e) => setClassId(e.target.value === "" ? undefined : Number(e.target.value))}>
          <option value="">All classes</option>
          {classes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        <select className="input" value={source ?? ""} onChange={(e) => setSource((e.target.value || undefined) as Source | undefined)}>
          <option value="">All sources</option>
          {SOURCES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select
          className="input"
          value={reviewStatus ?? ""}
          onChange={(e) => setReviewStatus((e.target.value || undefined) as ReviewStatus | undefined)}
          aria-label="Review status"
        >
          <option value="">Reviewed or not</option>
          {REVIEW_FILTERS.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </select>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {images.length === 0 && !error && <p className="text-sm text-neutral-500">No images match.</p>}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {images.map((img) => (
          <figure key={img.id} className="group relative overflow-hidden rounded border border-neutral-300 dark:border-neutral-700">
            <Link href={`/projects/${projectId}/review?image=${img.id}`} title="Open in review screen">
              {/* eslint-disable-next-line @next/next/no-img-element -- served by the API, not optimizable by next/image */}
              <img src={imageUrl(img.id)} alt={`${className(img.class_id)} #${img.id}`} loading="lazy" className="aspect-square w-full object-cover" />
            </Link>
            <figcaption className="flex items-center justify-between px-2 py-1 text-xs">
              <span>
                {className(img.class_id)} · {img.source}
              </span>
              <span className={STATUS_STYLE[img.review_status]}>{img.review_status === "pending" ? "not reviewed" : img.review_status}</span>
            </figcaption>
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
