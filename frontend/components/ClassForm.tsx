"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export default function ClassForm({ projectId, onAdded }: { projectId: number; onAdded: () => void }) {
  const [text, setText] = useState("");
  const [negative, setNegative] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ text: string; error?: boolean }[] | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const names = text.split(/[,\n]+/).map((n) => n.trim()).filter(Boolean);
    if (names.length === 0) return;
    setBusy(true);
    try {
      const r = await api.createClasses(projectId, names, negative);
      const kind = negative ? "hard negative" : "target";
      setMessage([
        { text: `Added ${r.added.length} ${kind} ${r.added.length === 1 ? "class" : "classes"}${r.added.length ? `: ${r.added.map((c) => c.name).join(", ")}` : ""}` },
        ...(r.skipped.length ? [{ text: `Skipped ${r.skipped.length} already in project: ${r.skipped.join(", ")}` }] : []),
        ...(r.invalid.length ? [{ text: `Invalid (use letters, digits, - and _): ${r.invalid.join(", ")}`, error: true }] : []),
      ]);
      // Keep only the entries that need fixing so a retry doesn't resubmit the rest.
      setText(r.invalid.join(", "));
      onAdded();
    } catch (err) {
      setMessage([{ text: (err as Error).message, error: true }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-2">
      <textarea
        className="input block h-20 w-full max-w-md"
        placeholder={"Class names, separated by commas or new lines\ne.g. quadcopter, hexacopter"}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-1 text-sm">
          <input type="checkbox" checked={negative} onChange={(e) => setNegative(e.target.checked)} /> hard negatives
        </label>
        <button className="btn" disabled={busy || !text.trim()}>
          {busy ? "Adding…" : "Add classes"}
        </button>
      </div>
      {message && (
        <ul className="text-sm">
          {message.map((l, i) => (
            <li key={i} className={l.error ? "text-red-600" : "text-neutral-600 dark:text-neutral-400"}>
              {l.text}
            </li>
          ))}
        </ul>
      )}
    </form>
  );
}
