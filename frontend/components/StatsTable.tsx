"use client";

import Link from "next/link";
import { useState } from "react";
import { SOURCES, api, type StatRow } from "@/lib/api";
import { saveGalleryFilters } from "@/lib/galleryFilters";

export default function StatsTable({ projectId, rows, onChanged }: { projectId: number; rows: StatRow[]; onChanged: () => void }) {
  const [confirming, setConfirming] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const total = (r: StatRow) => r.uploaded + r.scraped + r.generated + r.imported;
  const removable = (r: StatRow) => total(r) === 0 && r.boxes === 0; // the server enforces this too

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

  return (
    <div className="space-y-2">
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
            <th className="px-2 text-right">To review</th>
            <th className="pl-2" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.class_id} className="border-b border-neutral-200 dark:border-neutral-800" data-testid={`class-row-${r.class_name}`}>
              <td className="py-1.5 pr-4">
                {r.class_name}
                {r.is_negative && <span className="ml-2 rounded bg-neutral-200 px-1.5 py-0.5 text-xs dark:bg-neutral-800">negative</span>}
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
              <td className="px-2 text-right tabular-nums">
                {r.pending > 0 ? (
                  // The review screen shows whatever the main page's filters say, so this link sets them to "this class, not reviewed".
                  <Link
                    href={`/projects/${projectId}/review`}
                    onClick={() => saveGalleryFilters(projectId, { scope: "", classId: r.class_id, reviewStatus: "pending" })}
                    className="text-blue-500 hover:underline"
                  >
                    {r.pending} →
                  </Link>
                ) : (
                  <span className="text-neutral-500">{r.is_negative ? "–" : 0}</span>
                )}
              </td>
              <td className="whitespace-nowrap pl-2 text-right">
                {!removable(r) ? (
                  <span className="text-xs text-neutral-500" title="Delete its images and boxes first">
                    in use
                  </span>
                ) : confirming === r.class_id ? (
                  <span className="flex justify-end gap-1">
                    <button className="btn bg-red-700/40 px-2 py-0.5 text-xs" onClick={() => remove(r)}>
                      Confirm remove
                    </button>
                    <button className="btn px-2 py-0.5 text-xs" onClick={() => setConfirming(null)}>
                      Keep
                    </button>
                  </span>
                ) : (
                  <button className="btn px-2 py-0.5 text-xs" onClick={() => setConfirming(r.class_id)}>
                    Remove
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
}
