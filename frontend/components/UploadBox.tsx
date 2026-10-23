"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { api, type DetClass } from "@/lib/api";
import { droppedEntries, fromEntries, fromInput, plan, type Picked } from "@/lib/dropFiles";
import { NO_FILTERS, saveGalleryFilters } from "@/lib/galleryFilters";
import { folderHint } from "@/lib/uploadHints";
import { setUploadNotice } from "@/lib/uploadNotice";

const BATCH = 20; // files per request, keeps request bodies small

interface Landed {
  folder: string; // the cleaned name the server filed them under
  added: number;
}

/** Upload images or whole folders. A folder goes to <class>/uploaded/<folder> and opens in the gallery when it is done. */
export default function UploadBox({ projectId, classes, onDone }: { projectId: number; classes: DetClass[]; onDone: () => void }) {
  const router = useRouter();
  const [classId, setClassId] = useState<number | "">("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [summary, setSummary] = useState<string[]>([]);
  const [landed, setLanded] = useState<Landed[]>([]);
  const [dragging, setDragging] = useState(false);
  const [check, setCheck] = useState<{ picked: Picked[]; hints: string[] } | null>(null); // a pick the folder name and the class disagree about, waiting for the person
  const files = useRef<HTMLInputElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const ready = classId !== "" && !busy && !check;

  function openFolder(name: string) {
    saveGalleryFilters(projectId, { ...NO_FILTERS, scope: `folder:${name}` });
    router.push(`/projects/${projectId}/images`);
  }

  /** Uploading starts the moment files are picked, and a negative class files them as already reviewed background: so a folder whose name
   *  and class disagree waits here for a yes first. */
  function upload(picked: Picked[]) {
    const cls = classes.find((c) => c.id === classId);
    if (!cls || picked.length === 0) return Promise.resolve();
    const hints = [...plan(picked).groups.keys()].flatMap((name) => (name ? [folderHint(name, cls.name, cls.is_negative)] : [])).filter((h): h is string => h !== null);
    if (hints.length > 0) {
      setCheck({ picked, hints });
      return Promise.resolve();
    }
    return doUpload(picked);
  }

  function cancelCheck() {
    setCheck(null);
    if (files.current) files.current.value = "";
    if (folder.current) folder.current.value = "";
  }

  async function doUpload(picked: Picked[]) {
    if (classId === "" || picked.length === 0) return;
    const { groups, skipped } = plan(picked);
    const total = [...groups.values()].reduce((n, g) => n + g.length, 0);
    setSummary(skipped && total === 0 ? [`No images found: ${skipped} file(s) skipped (only jpg, png and webp are uploaded)`] : []);
    setLanded([]);
    if (total === 0) return;
    setBusy(true);
    let done = 0;
    let added = 0;
    const duplicates: string[] = [];
    const problems: string[] = [];
    const where: Landed[] = [];
    let loose = false;
    try {
      for (const [name, group] of groups) {
        loose ||= name === null;
        let cleaned: string | null = null;
        let addedHere = 0;
        for (let i = 0; i < group.length; i += BATCH) {
          setProgress({ done, total });
          const part = group.slice(i, i + BATCH);
          const r = await api.upload(projectId, classId, part.map((p) => p.file), name ?? undefined, name === null ? undefined : part.map((p) => p.path));
          cleaned = r.folder;
          addedHere += r.added.length;
          duplicates.push(...r.duplicates);
          problems.push(...r.rejected);
          done += part.length;
        }
        added += addedHere;
        if (cleaned && addedHere > 0) where.push({ folder: cleaned, added: addedHere });
      }
    } catch (e) {
      problems.push((e as Error).message);
    }
    setProgress(null);
    setBusy(false);
    if (files.current) files.current.value = "";
    if (folder.current) folder.current.value = "";
    onDone();

    const lines = [
      `${added} added`,
      ...where.map((w) => `${w.added} in ${w.folder}`),
      ...(duplicates.length ? [`${duplicates.length} duplicate(s) skipped`] : []),
      ...(skipped ? [`${skipped} other file(s) skipped (only jpg, png and webp are uploaded)`] : []),
      ...problems.slice(0, 20).map((p) => `Rejected: ${p}`),
      ...(problems.length > 20 ? [`…and ${problems.length - 20} more rejected`] : []),
    ];
    if (where.length === 1 && !loose && problems.length === 0) {
      // One folder and nothing went wrong: open it, with the numbers on the page it opens.
      const extras = [...(duplicates.length ? [`${duplicates.length} duplicate(s) skipped`] : []), ...(skipped ? [`${skipped} other file(s) skipped`] : [])];
      setUploadNotice(projectId, [`${added} image(s) added to ${where[0].folder}`, ...extras].join(" · "));
      openFolder(where[0].folder);
      return;
    }
    setSummary(lines);
    setLanded(where);
  }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (!ready) return;
    const entries = droppedEntries(e.dataTransfer); // read now: the browser empties it when this handler returns
    const loose = Array.from(e.dataTransfer.files);
    try {
      await upload(entries ? await fromEntries(entries) : fromInput(loose));
    } catch (err) {
      setSummary([`Could not read the dropped items: ${(err as Error).message}`]);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-neutral-500">
        The class decides where the images are filed (<code>class/uploaded/folder</code>) and whether they count as hard negatives; each box you draw in review carries its own class.
      </p>
      <select className="input" value={classId} onChange={(e) => {
          setClassId(e.target.value === "" ? "" : Number(e.target.value));
          cancelCheck();
        }}
        aria-label="Target class"
      >
        <option value="">Select target class…</option>
        {classes.filter((c) => !c.is_mixed).map((c) => (
          <option key={c.id} value={c.id}>
            {c.name}
            {c.is_negative ? " (negative)" : ""}
          </option>
        ))}
      </select>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        data-testid="drop-zone"
        className={`flex min-h-32 flex-col items-center justify-center gap-3 rounded border-2 border-dashed p-4 text-sm ${
          classId === "" ? "opacity-50" : ""
        } ${dragging ? "border-blue-500 bg-blue-500/10" : "border-neutral-400 dark:border-neutral-600"}`}
      >
        {busy && progress ? (
          <div className="w-full max-w-sm space-y-1 text-center">
            <p>
              Uploading {progress.done} / {progress.total}
            </p>
            <div className="h-2 overflow-hidden rounded bg-neutral-300 dark:bg-neutral-700">
              <div className="h-full bg-blue-600 transition-all" style={{ width: `${(100 * progress.done) / progress.total}%` }} />
            </div>
          </div>
        ) : (
          <>
            <p className="text-neutral-500">{classId === "" ? "Choose a class first" : "Drop images or a folder here, or"}</p>
            <div className="flex gap-2">
              <button className="btn" disabled={!ready} onClick={() => files.current?.click()}>
                Choose images
              </button>
              <button className="btn" disabled={!ready} onClick={() => folder.current?.click()}>
                Choose folder
              </button>
            </div>
          </>
        )}
      </div>
      <input ref={files} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden onChange={(e) => upload(fromInput(Array.from(e.target.files ?? [])))} />
      {/* webkitdirectory is not in React's input types, but every current browser reads it */}
      <input ref={folder} type="file" multiple hidden data-testid="folder-input" {...{ webkitdirectory: "" }} onChange={(e) => upload(fromInput(Array.from(e.target.files ?? [])))} />
      {check && (
        <div className="space-y-2 rounded border border-amber-500/60 bg-amber-500/10 p-3 text-sm" data-testid="upload-hint">
          {check.hints.map((h) => (
            <p key={h}>{h}</p>
          ))}
          <div className="flex gap-2">
            <button
              className="btn"
              onClick={() => {
                const p = check.picked;
                setCheck(null);
                doUpload(p);
              }}
              data-testid="upload-anyway"
            >
              Upload anyway
            </button>
            <button className="btn" onClick={cancelCheck}>
              Cancel
            </button>
          </div>
        </div>
      )}
      {summary.length > 0 && (
        <ul className="text-sm text-neutral-600 dark:text-neutral-400">
          {summary.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
      )}
      {landed.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {landed.map((w) => (
            <button key={w.folder} className="btn" onClick={() => openFolder(w.folder)}>
              Open {w.folder}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
