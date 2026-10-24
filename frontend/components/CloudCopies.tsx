"use client";

import { useCallback, useEffect, useState } from "react";
import { api, parseUtc, type CloudCopies as CloudCopiesData } from "@/lib/api";

const mb = (n: number) => (n < 2 ** 20 ? `${Math.max(1, Math.round(n / 1024))} KB` : n < 2 ** 30 ? `${(n / 2 ** 20).toFixed(0)} MB` : `${(n / 2 ** 30).toFixed(2)} GB`);

/** The dataset zips this project has in the cloud bucket (Cloudflare R2). A RunPod pod downloads its training data from there; a copy is made
 *  the first time a version is used on RunPod and stays until it is deleted here (or its version is deleted). Shown only when a bucket is set up. */
export default function CloudCopies({ projectId, refreshKey }: { projectId: number; refreshKey: number }) {
  const [data, setData] = useState<CloudCopiesData | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .cloudCopies(projectId)
      .then((d) => {
        setData(d);
        setSelected((prev) => new Set([...prev].filter((k) => d.copies.some((c) => c.key === k && !c.in_use)))); // only what is still there and free
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [projectId]);
  useEffect(load, [load, refreshKey]);

  if (!data?.enabled && !error) return null;
  const copies = data?.copies ?? [];
  const chosen = copies.filter((c) => selected.has(c.key));
  const chosenBytes = chosen.reduce((n, c) => n + c.size, 0);
  const deletable = copies.filter((c) => !c.in_use);
  // The list is newest first. Every copy except the newest one that still belongs to a dataset version; a copy no version matches can never be used.
  const newest = deletable.find((c) => c.dataset_name !== null);
  const older = deletable.filter((c) => c !== newest);

  const toggle = (key: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(key)) next.add(key);
      return next;
    });

  async function remove() {
    setBusy(true);
    setNotice(null);
    try {
      const r = await api.deleteCloudCopies(projectId, chosen.map((c) => c.key));
      setNotice(`Deleted ${r.deleted} cop${r.deleted === 1 ? "y" : "ies"} from the cloud (${mb(r.freed_bytes)}).`);
      setSelected(new Set());
      setConfirming(false);
      load();
    } catch (e) {
      setError((e as Error).message);
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-2 rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid="cloud-copies">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="font-medium">Copies in the cloud</h2>
        <span className="text-neutral-500" data-testid="cloud-total">
          {copies.length} cop{copies.length === 1 ? "y" : "ies"} · {mb(data?.total_bytes ?? 0)} in the Cloudflare bucket
        </span>
      </div>
      <p className="text-xs text-neutral-500">
        RunPod pods download their training data from here. Deleting a copy only frees space in the bucket: the dataset version stays, and the next RunPod run on it
        uploads it again, which is slow from this Mac.
      </p>
      {error && <p className="text-red-600">{error}</p>}
      {notice && (
        <p className="text-green-600" data-testid="cloud-notice">
          {notice}
        </p>
      )}
      {data?.enabled && copies.length === 0 && !error && <p className="text-neutral-500">Nothing is stored in the cloud for this project.</p>}
      {copies.length > 0 && (
        <>
          <ul className="space-y-1">
            {copies.map((c) => (
              <li key={c.key} className="flex flex-wrap items-center gap-x-3 gap-y-0.5" data-testid={`cloud-copy-${c.key}`}>
                <label className="flex cursor-pointer items-center gap-2">
                  <input type="checkbox" className="h-4 w-4" checked={selected.has(c.key)} disabled={c.in_use || busy} onChange={() => toggle(c.key)} aria-label={`Select the cloud copy of ${c.dataset_name ?? c.key}`} />
                  <span className={c.dataset_name ? "font-medium" : "text-neutral-500"}>{c.dataset_name ?? "no dataset version matches this copy"}</span>
                </label>
                <span className="tabular-nums text-neutral-500">{mb(c.size)}</span>
                <span className="text-neutral-500">uploaded {parseUtc(c.modified).toLocaleString()}</span>
                {c.in_use && <span className="text-amber-500">a run is waiting or training: it may still need this</span>}
              </li>
            ))}
          </ul>
          <div className="flex flex-wrap items-center gap-2">
            {!confirming && (
              <>
                <button className="btn" onClick={() => setSelected(new Set(older.map((c) => c.key)))} disabled={older.length === 0 || busy} data-testid="cloud-select-older">
                  Select all but the newest
                </button>
                <button className="btn" onClick={() => setConfirming(true)} disabled={chosen.length === 0 || busy} data-testid="cloud-delete">
                  Delete {chosen.length > 0 ? chosen.length : ""} selected from the cloud…
                </button>
                {selected.size > 0 && (
                  <button className="text-neutral-500 hover:text-current" onClick={() => setSelected(new Set())}>
                    Clear selection
                  </button>
                )}
              </>
            )}
            {confirming && (
              <>
                <span className="text-red-500">
                  Delete {chosen.length === 1 ? "this copy" : `these ${chosen.length} copies`} ({mb(chosenBytes)}) from the cloud? This cannot be undone.
                </span>
                <button className="btn bg-red-600 text-white hover:bg-red-700" onClick={remove} disabled={busy} data-testid="cloud-confirm-delete">
                  Delete from the cloud
                </button>
                <button className="btn" onClick={() => setConfirming(false)} disabled={busy}>
                  Cancel
                </button>
              </>
            )}
          </div>
        </>
      )}
    </section>
  );
}
