"use client";

import { useRef, useState } from "react";
import { api, type DetClass } from "@/lib/api";

const BATCH = 20; // files per request, keeps request bodies small

export default function UploadBox({ projectId, classes, onDone }: { projectId: number; classes: DetClass[]; onDone: () => void }) {
  const [classId, setClassId] = useState<number | "">("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [summary, setSummary] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function upload(files: File[]) {
    if (classId === "" || files.length === 0) return;
    setBusy(true);
    setSummary([]);
    let added = 0;
    const duplicates: string[] = [];
    const problems: string[] = [];
    try {
      for (let i = 0; i < files.length; i += BATCH) {
        setProgress(`Uploading ${Math.min(i + BATCH, files.length)} / ${files.length}`);
        const r = await api.upload(projectId, classId, files.slice(i, i + BATCH));
        added += r.added.length;
        duplicates.push(...r.duplicates);
        problems.push(...r.rejected);
      }
    } catch (e) {
      problems.push((e as Error).message);
    }
    setSummary([`${added} added`, ...(duplicates.length ? [`${duplicates.length} duplicate(s) skipped`] : []), ...problems.map((p) => `Rejected: ${p}`)]);
    setProgress("");
    setBusy(false);
    if (input.current) input.current.value = "";
    onDone();
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-neutral-500">The class only decides the folder and whether the images count as hard negatives; each box you draw in review carries its own class.</p>
      <select className="input" value={classId} onChange={(e) => setClassId(e.target.value === "" ? "" : Number(e.target.value))}>
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
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          upload(Array.from(e.dataTransfer.files));
        }}
        onClick={() => classId !== "" && input.current?.click()}
        className={`flex h-28 items-center justify-center rounded border-2 border-dashed text-sm ${
          classId === "" ? "cursor-not-allowed opacity-50" : "cursor-pointer"
        } ${dragging ? "border-blue-500 bg-blue-500/10" : "border-neutral-400 dark:border-neutral-600"}`}
      >
        {busy ? progress : classId === "" ? "Choose a class first" : "Drop images here or click to browse"}
      </div>
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden onChange={(e) => upload(Array.from(e.target.files ?? []))} />
      {summary.length > 0 && (
        <ul className="text-sm text-neutral-600 dark:text-neutral-400">
          {summary.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
