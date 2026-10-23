"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { api, type FolderRow } from "@/lib/api";
import { NO_FILTERS, saveGalleryFilters } from "@/lib/galleryFilters";

interface Place {
  key: string;
  label: string;
  kind: string;
  scope: string; // the gallery scope value that shows exactly this place
  total: number;
  reviewed: number;
  rejected: number;
  ai_ok: number;
  ai_flagged: number;
  ai_unchecked: number;
}

const pct = (n: number, of: number) => (of === 0 ? "–" : `${Math.round((100 * n) / of)}%`);

/** Where the images are (an imported dataset, an upload folder, scraped, loose uploads): how much is reviewed, what the AI thinks, what needs a look. */
export default function PlacesTable({ projectId, refreshKey }: { projectId: number; refreshKey: number }) {
  const router = useRouter();
  const [rows, setRows] = useState<FolderRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.folders(projectId).then(setRows).catch((e) => setError(e.message));
  }, [projectId, refreshKey]);

  const places = useMemo(() => {
    const byKey = new Map<string, Place>();
    for (const r of rows ?? []) {
      const dataset = r.source === "imported" && r.folder;
      const key = dataset ? `dataset:${r.folder}` : r.folder ? `folder:${r.folder}` : `source:${r.source}`;
      const p = byKey.get(key) ?? {
        key,
        label: dataset ? String(r.folder) : r.folder ? String(r.folder) : r.source === "uploaded" ? "uploaded (loose files)" : r.source,
        kind: dataset ? "imported dataset" : r.folder ? "upload folder" : "source",
        scope: key,
        total: 0, reviewed: 0, rejected: 0, ai_ok: 0, ai_flagged: 0, ai_unchecked: 0,
      };
      p.total += r.total;
      p.reviewed += r.reviewed;
      p.rejected += r.rejected;
      p.ai_ok += r.ai_ok;
      p.ai_flagged += r.ai_flagged;
      p.ai_unchecked += r.ai_unchecked;
      byKey.set(key, p);
    }
    return [...byKey.values()].sort((a, b) => b.total - a.total);
  }, [rows]);

  /** Open the Images page (or the review queue) on one place, optionally only the images the AI wants looked at. */
  function open(p: Place, flaggedOnly: boolean, review: boolean) {
    saveGalleryFilters(projectId, { ...NO_FILTERS, scope: p.scope, ai: flaggedOnly ? "flagged" : undefined });
    router.push(`/projects/${projectId}/${review ? "review" : "images"}`);
  }

  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (rows && places.length === 0) return <p className="text-sm text-neutral-500">No images yet. Add some in Add data.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="places-table">
        <thead>
          <tr className="border-b border-neutral-300 text-left dark:border-neutral-700">
            <th className="py-1.5 pr-4">Place</th>
            <th className="px-2 text-right">Images</th>
            <th className="px-2 text-right" title="Reviewed by a person (rejected ones are not counted)">Reviewed</th>
            <th className="px-2 text-right" title="The AI agrees with the boxes (or found nothing on an unlabelled image)">AI ok</th>
            <th className="px-2 text-right" title="Boxes the AI doubts and nobody has decided on yet">Needs a look</th>
            <th className="px-2 text-right" title="The AI has never looked at these">Not checked</th>
            <th className="pl-2" />
          </tr>
        </thead>
        <tbody>
          {places.map((p) => (
            <tr key={p.key} className="border-b border-neutral-200 dark:border-neutral-800" data-testid={`place-${p.key}`}>
              <td className="py-1.5 pr-4">
                <span className="font-medium">{p.label}</span> <span className="text-xs text-neutral-500">{p.kind}</span>
              </td>
              <td className="px-2 text-right tabular-nums">{p.total.toLocaleString()}</td>
              <td className="px-2 text-right tabular-nums">{pct(p.reviewed, p.total)}</td>
              <td className="px-2 text-right tabular-nums">{pct(p.ai_ok, p.total)}</td>
              <td className={`px-2 text-right tabular-nums ${p.ai_flagged > 0 ? "font-medium text-amber-500" : "text-neutral-500"}`}>{p.ai_flagged.toLocaleString()}</td>
              <td className="px-2 text-right tabular-nums text-neutral-500">{p.ai_unchecked.toLocaleString()}</td>
              <td className="space-x-1 pl-2 text-right whitespace-nowrap">
                {p.ai_flagged > 0 && (
                  <button className="btn" onClick={() => open(p, true, true)} title="Review the images the AI wants looked at in this place">
                    Review {p.ai_flagged.toLocaleString()}
                  </button>
                )}
                <button className="btn" onClick={() => open(p, false, false)}>
                  Open
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
