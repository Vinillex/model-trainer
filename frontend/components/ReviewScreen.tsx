"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import BoxEditor, { type Box } from "@/components/BoxEditor";
import { api, imageUrl, type BoxIn, type ImageDetail, type ImageRow, type Project, type ReviewStatus } from "@/lib/api";

const PALETTE = ["#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7", "#f97316", "#14b8a6", "#ec4899"];
const QUEUE_LIMIT = 500;
type StatusFilter = ReviewStatus | "all";

const toBoxes = (d: ImageDetail): Box[] =>
  d.annotations.map((a) => ({
    key: `a${a.id}`,
    classId: a.class_id,
    x1: Math.max(0, a.cx - a.w / 2),
    y1: Math.max(0, a.cy - a.h / 2),
    x2: Math.min(1, a.cx + a.w / 2),
    y2: Math.min(1, a.cy + a.h / 2),
  }));

const toBoxIn = (b: Box): BoxIn => {
  const r = (v: number) => Math.round(v * 1e6) / 1e6;
  return { class_id: b.classId, cx: r((b.x1 + b.x2) / 2), cy: r((b.y1 + b.y2) / 2), w: r(b.x2 - b.x1), h: r(b.y2 - b.y1) };
};

export default function ReviewScreen() {
  const projectId = Number(useParams<{ id: string }>().id);
  const params = useSearchParams();
  const jumpTo = params.get("image") ? Number(params.get("image")) : null;

  const [project, setProject] = useState<Project | null>(null);
  const [classFilter, setClassFilter] = useState<number | undefined>(params.get("class") ? Number(params.get("class")) : undefined);
  const [status, setStatus] = useState<StatusFilter>(jumpTo ? "all" : "pending");
  const [queue, setQueue] = useState<ImageRow[] | null>(null);
  const [index, setIndex] = useState(0);
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [activeClass, setActiveClass] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const jumped = useRef(false);

  const targets = useMemo(() => (project?.classes ?? []).filter((c) => !c.is_negative && !c.is_mixed), [project]);
  const classById = useMemo(() => new Map((project?.classes ?? []).map((c) => [c.id, c])), [project]);
  const colorFor = useCallback((id: number) => PALETTE[(classById.get(id)?.index ?? id) % PALETTE.length], [classById]);
  const nameFor = useCallback((id: number) => classById.get(id)?.name ?? String(id), [classById]);

  const current = queue?.[index] ?? null;
  const imageClass = current ? classById.get(current.class_id) : undefined;
  const readOnly = imageClass?.is_negative ?? false;

  useEffect(() => {
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
  }, [projectId]);

  // (Re)load the queue whenever the filters change.
  useEffect(() => {
    let cancelled = false;
    api
      .listImages(projectId, { classId: classFilter, reviewStatus: status === "all" ? undefined : status, limit: QUEUE_LIMIT, offset: 0 })
      .then(async (rows) => {
        if (cancelled) return;
        let start = 0;
        if (jumpTo && !jumped.current) {
          jumped.current = true;
          start = rows.findIndex((r) => r.id === jumpTo);
          if (start < 0) {
            // Not within the newest QUEUE_LIMIT images: put it at the front.
            const one = await api.getImage(jumpTo);
            rows = [one, ...rows];
            start = 0;
          }
        }
        setQueue(rows);
        setIndex(start);
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [projectId, classFilter, status, jumpTo]);

  // Load the boxes for the current image.
  const currentId = current?.id;
  useEffect(() => {
    if (currentId === undefined) {
      setDetail(null);
      setBoxes([]);
      return;
    }
    let cancelled = false;
    api
      .getImage(currentId)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        setBoxes(toBoxes(d));
        setSelected(null);
        setDirty(false);
        setError(null);
        setActiveClass((prev) => {
          const own = classById.get(d.class_id);
          if (own && !own.is_negative && !own.is_mixed) return own.id; // default to the folder the image was filed under
          return prev ?? targets[0]?.id ?? null;
        });
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [currentId, classById, targets]);

  const change = useCallback((next: Box[]) => {
    setBoxes(next);
    setDirty(true);
  }, []);

  /** Persist the current boxes. Returns false (and shows the error) if the server refused. */
  const save = useCallback(
    async (newStatus?: ReviewStatus): Promise<boolean> => {
      if (!detail) return true;
      setSaving(true);
      try {
        const out = await api.putAnnotations(detail.id, boxes.map(toBoxIn), newStatus);
        setDetail(out);
        setDirty(false);
        setQueue((q) => q && q.map((r) => (r.id === out.id ? { ...r, review_status: out.review_status, class_id: out.class_id } : r)));
        setError(null);
        // A reviewed image moves to the folder its boxes show (or "mixed"); that class may be new, so reload the class list.
        if (out.class_id !== detail.class_id) {
          const p = classById.has(out.class_id) ? project : await api.getProject(projectId);
          if (p && p !== project) setProject(p);
          setNotice(`Moved to the ${p?.classes.find((c) => c.id === out.class_id)?.name ?? "new"} folder`);
        } else {
          setNotice(null);
        }
        return true;
      } catch (e) {
        setError((e as Error).message);
        return false;
      } finally {
        setSaving(false);
      }
    },
    [detail, boxes, classById, project, projectId],
  );

  /** Move to another image, saving unsaved edits first so nothing drawn is lost. */
  const go = useCallback(
    async (delta: number) => {
      if (!queue) return;
      const next = index + delta;
      if (next < 0 || next >= queue.length) return;
      if (dirty && !(await save())) return;
      setIndex(next);
    },
    [queue, index, dirty, save],
  );

  /** Save with a status, then continue: in the pending queue the image leaves it, elsewhere we step forward. */
  const finish = useCallback(
    async (newStatus: ReviewStatus) => {
      if (!(await save(newStatus)) || !queue || !current) return;
      if (status === "pending") {
        const rest = queue.filter((r) => r.id !== current.id);
        setQueue(rest);
        setIndex((i) => Math.min(i, Math.max(0, rest.length - 1)));
      } else {
        setIndex((i) => Math.min(i + 1, queue.length - 1));
      }
    },
    [save, queue, current, status],
  );

  const removeSelected = useCallback(() => {
    if (selected) {
      change(boxes.filter((b) => b.key !== selected));
      setSelected(null);
    }
  }, [selected, boxes, change]);

  const setBoxClass = useCallback(
    (classId: number) => {
      setActiveClass(classId);
      if (selected) change(boxes.map((b) => (b.key === selected ? { ...b, classId } : b)));
    },
    [selected, boxes, change],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        removeSelected();
      } else if (e.key === "Escape") setSelected(null);
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "Enter" && !readOnly) {
        e.preventDefault();
        finish("reviewed");
      } else if (e.key === "x" && !readOnly) finish("rejected");
      else if (/^[1-9]$/.test(e.key) && targets[Number(e.key) - 1]) setBoxClass(targets[Number(e.key) - 1].id);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [removeSelected, go, finish, setBoxClass, targets, readOnly]);

  const selectedBox = boxes.find((b) => b.key === selected);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link href={`/projects/${projectId}`} className="text-sm text-neutral-500 hover:underline">
          ← {project?.name ?? "Project"}
        </Link>
        <h1 className="text-xl font-semibold">Review</h1>
        <select className="input" aria-label="Status filter" value={status} onChange={(e) => setStatus(e.target.value as StatusFilter)}>
          <option value="pending">Pending</option>
          <option value="reviewed">Reviewed</option>
          <option value="rejected">Rejected</option>
          <option value="all">All</option>
        </select>
        <select className="input" aria-label="Class filter" value={classFilter ?? ""} onChange={(e) => setClassFilter(e.target.value === "" ? undefined : Number(e.target.value))}>
          <option value="">All classes</option>
          {project?.classes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        {queue && queue.length > 0 && (
          <span className="text-sm text-neutral-500" data-testid="position">
            {index + 1} / {queue.length}
            {queue.length >= QUEUE_LIMIT && "+"}
          </span>
        )}
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {notice && !error && <p className="text-sm text-neutral-400" data-testid="moved-notice">{notice}</p>}
      {queue === null && !error && <p className="text-sm text-neutral-500">Loading…</p>}
      {queue?.length === 0 && (
        <p className="text-neutral-500">
          Nothing to show with these filters.{" "}
          <button className="underline" onClick={() => setStatus("all")}>
            Show all images
          </button>
        </p>
      )}

      {current && detail && detail.id === current.id && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-neutral-500">
              #{detail.id} · filed under <b>{nameFor(detail.class_id)}</b> · {detail.source} · {detail.width}×{detail.height}
            </span>
            <span className={{ pending: "text-amber-500", reviewed: "text-green-500", rejected: "text-red-500" }[detail.review_status]} data-testid="review-status">
              {detail.review_status}
            </span>
            {dirty && <span className="rounded bg-amber-500/20 px-1.5 text-amber-500">unsaved</span>}
          </div>

          {readOnly ? (
            <p className="text-sm text-neutral-500">This is a hard-negative image: it never has boxes.</p>
          ) : (
            <div className="flex min-h-9 flex-wrap items-center gap-2">
              <span className="text-sm text-neutral-500">Draw as:</span>
              {targets.map((c, i) => (
                <button
                  key={c.id}
                  onClick={() => setBoxClass(c.id)}
                  className={`rounded border px-2 py-0.5 text-sm ${activeClass === c.id ? "border-white" : "border-neutral-600 opacity-70"}`}
                  style={{ background: activeClass === c.id ? colorFor(c.id) : undefined }}
                >
                  {i < 9 && <kbd className="mr-1 opacity-60">{i + 1}</kbd>}
                  {c.name}
                </button>
              ))}
              {/* Always rendered: a button that appears on selection would change this row's height and shift the image mid-drag. */}
              <button className="btn" onClick={removeSelected} disabled={!selectedBox}>
                {selectedBox ? `Delete box (${nameFor(selectedBox.classId)})` : "Delete box"}
              </button>
            </div>
          )}

          <BoxEditor
            src={imageUrl(detail.id)}
            boxes={boxes}
            selectedKey={selected}
            activeClassId={activeClass}
            disabled={readOnly}
            colorFor={colorFor}
            nameFor={nameFor}
            onChange={change}
            onSelect={setSelected}
          />

          <div className="flex flex-wrap items-center gap-2">
            <button className="btn" onClick={() => go(-1)} disabled={saving || index === 0}>
              ← Prev
            </button>
            <button className="btn" onClick={() => go(1)} disabled={saving || index >= queue!.length - 1}>
              Next →
            </button>
            {!readOnly && (
              <>
                <button className="btn bg-green-700/30" onClick={() => finish("reviewed")} disabled={saving}>
                  Accept ({boxes.length} {boxes.length === 1 ? "box" : "boxes"}) &amp; next
                </button>
                <button className="btn" onClick={() => finish("rejected")} disabled={saving}>
                  Reject image
                </button>
                <button className="btn" onClick={() => save()} disabled={saving || !dirty}>
                  Save
                </button>
              </>
            )}
          </div>
          <p className="text-xs text-neutral-500">
            Drag to draw · click a box to select, drag to move, drag handles to resize · Delete removes · 1–9 pick class · Enter accept · X reject · ←/→ navigate
          </p>
        </div>
      )}
    </div>
  );
}
