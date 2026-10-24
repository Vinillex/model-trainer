"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { SOURCES, api, type StatRow } from "@/lib/api";
import { NO_FILTERS, saveGalleryFilters } from "@/lib/galleryFilters";

const plural = (n: number, one: string, many: string) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

/** The classes, as two tables: the drone classes the model learns, and the hard negatives (things that look like drones but are not). */
export default function StatsTable({ projectId, rows, onChanged }: { projectId: number; rows: StatRow[]; onChanged: () => void }) {
  const router = useRouter();
  const [confirming, setConfirming] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const total = (r: StatRow) => r.uploaded + r.scraped + r.generated + r.imported;
  const inUse = (r: StatRow) => total(r) > 0 || r.boxes > 0; // the server enforces this too

  async function remove(r: StatRow) {
    setError(null);
    try {
      await api.deleteClass(projectId, r.class_id);
      setConfirming(null);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  /** What still uses a class, so it can be moved, relabelled or deleted: the boxes open in the review screen (to fix them), the images in the gallery. */
  function show(r: StatRow, what: "boxes" | "images") {
    saveGalleryFilters(projectId, what === "boxes" ? { ...NO_FILTERS, boxClassId: r.class_id } : { ...NO_FILTERS, classId: r.class_id });
    router.push(`/projects/${projectId}/${what === "boxes" ? "review" : "images"}`);
  }

  function table(title: string, hint: string, list: StatRow[], negative: boolean) {
    return (
      <section className="space-y-2" data-testid={negative ? "negatives-table" : "classes-table"}>
        <div>
          <h2 className="font-medium">{title}</h2>
          <p className="text-xs text-neutral-500">{hint}</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-neutral-300 text-left dark:border-neutral-700">
              <th className="py-1.5 pr-4">Class</th>
              {SOURCES.map((s) => (
                <th key={s} className="px-2 text-right capitalize">
                  {s}
                </th>
              ))}
              <th className="px-2 text-right">Total</th>
              {/* negative images arrive already reviewed, so there is never anything to review in this table */}
              {!negative && <th className="px-2 text-right">To review</th>}
              <th className="pl-2" />
            </tr>
          </thead>
          <tbody>
            {list.length === 0 && (
              <tr>
                <td colSpan={SOURCES.length + 4} className="py-2 text-neutral-500">
                  {negative ? "No hard negatives yet. Add some below." : "No classes yet."}
                </td>
              </tr>
            )}
            {list.map((r) => (
              <tr key={r.class_id} className="border-b border-neutral-200 dark:border-neutral-800" data-testid={`class-row-${r.class_name}`}>
                <td className="py-1.5 pr-4">
                  {r.class_name}
                  {r.is_mixed && (
                    <span className="ml-2 rounded bg-neutral-200 px-1.5 py-0.5 text-xs dark:bg-neutral-800" title="Reviewed images whose boxes show two or more classes are moved here automatically. It is a folder, not a class the model learns.">
                      mixed folder
                    </span>
                  )}
                </td>
                {SOURCES.map((s) => (
                  <td key={s} className="px-2 text-right tabular-nums">
                    {r[s]}
                  </td>
                ))}
                <td className="px-2 text-right font-medium tabular-nums">{total(r)}</td>
                {!negative && (
                  <td className="px-2 text-right tabular-nums">
                    {r.pending > 0 ? (
                      // The review screen shows whatever the Images page's filters say, so this link sets them to "this class, not reviewed".
                      <Link
                        href={`/projects/${projectId}/review`}
                        onClick={() => saveGalleryFilters(projectId, { scope: "", classId: r.class_id, reviewStatus: "pending" })}
                        className="text-blue-500 hover:underline"
                      >
                        {r.pending} →
                      </Link>
                    ) : (
                      <span className="text-neutral-500">0</span>
                    )}
                  </td>
                )}
                <td className="whitespace-nowrap pl-2 text-right">
                  {confirming === r.class_id ? (
                    <span className="flex justify-end gap-1">
                      <button className="btn bg-red-700/40 px-2 py-0.5 text-xs" onClick={() => remove(r)}>
                        Confirm remove
                      </button>
                      <button className="btn px-2 py-0.5 text-xs" onClick={() => setConfirming(null)}>
                        Keep
                      </button>
                    </span>
                  ) : (
                    <span className="flex items-center justify-end gap-2">
                      {/* Where it is still used, as links; Remove is always here and only switches on once nothing uses the class. */}
                      {r.boxes > 0 && (
                        <button className="text-xs text-blue-500 hover:underline" onClick={() => show(r, "boxes")} title="Open the images that have a box of this class, to relabel or delete those boxes" data-testid={`show-boxes-${r.class_name}`}>
                          show {plural(r.boxes, "box", "boxes")}
                        </button>
                      )}
                      {total(r) > 0 && (
                        <button className="text-xs text-blue-500 hover:underline" onClick={() => show(r, "images")} title="Open the images filed under this class, to move or delete them">
                          show images
                        </button>
                      )}
                      <button
                        className="btn px-2 py-0.5 text-xs"
                        disabled={inUse(r)}
                        onClick={() => setConfirming(r.class_id)}
                        title={inUse(r) ? `Still in use: ${plural(total(r), "image", "images")} filed under it, ${plural(r.boxes, "box", "boxes")}. Move or relabel them first.` : "Remove this class"}
                        data-testid={`remove-${r.class_name}`}
                      >
                        Remove
                      </button>
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    );
  }

  return (
    <div className="space-y-6">
      {table("Drone classes", "What the model learns to find. A class can only be removed once no image is filed under it and no box uses it.", rows.filter((r) => !r.is_negative), false)}
      {table("Hard negatives", "Things that look like drones but are not (birds, balloons, helicopters…). Their images count as background: they never have boxes.", rows.filter((r) => r.is_negative), true)}
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
