"use client";

import Link from "next/link";
import ShortcutsHelp from "@/components/ShortcutsHelp";
import { useParams, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import BoxEditor, { type Box } from "@/components/BoxEditor";
import { api, imageUrl, type AiBox, type AiCheck, type AiStatusValue, type BoxIn, type ImageDetail, type ImageRow, type Project, type ReviewStatus } from "@/lib/api";
import { KIND_COLOR, applySuggestion, buildSuggestions, ghostFor, type Suggestion } from "@/lib/aiSuggestions";
import { NO_FILTERS, describeFilters, imageQuery, loadGalleryFilters, loadTrashFilters, saveGalleryFilters, saveTrashFilters, trashQuery, type GalleryFilters } from "@/lib/galleryFilters";

const PALETTE = ["#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7", "#f97316", "#14b8a6", "#ec4899"];
const QUEUE_LIMIT = 500;

const toBoxes = (d: ImageDetail): Box[] =>
  d.annotations.map((a) => ({
    key: `a${a.id}`,
    classId: a.class_id,
    x1: Math.max(0, a.cx - a.w / 2),
    y1: Math.max(0, a.cy - a.h / 2),
    x2: Math.min(1, a.cx + a.w / 2),
    y2: Math.min(1, a.cy + a.h / 2),
  }));

/** What the gallery card and the queue show for a stored check. */
const statusOf = (c: AiCheck): AiStatusValue => (c.stale ? "stale" : c.dismissed ? "dismissed" : c.verdict);

const toBoxIn = (b: Box): BoxIn => {
  const r = (v: number) => Math.round(v * 1e6) / 1e6;
  return { class_id: b.classId, cx: r((b.x1 + b.x2) / 2), cy: r((b.y1 + b.y2) / 2), w: r(b.x2 - b.x1), h: r(b.y2 - b.y1) };
};

export default function ReviewScreen() {
  const projectId = Number(useParams<{ id: string }>().id);
  const params = useSearchParams();
  const jumpTo = params.get("image") ? Number(params.get("image")) : null;
  const fromTrash = params.get("from") === "trash"; // opened from the Trash page: the queue is the trash, and leaving goes back there

  const [project, setProject] = useState<Project | null>(null);
  // No filters of its own: the queue is whatever the main page's gallery filters select (read once the page is in the browser).
  const [filters, setFilters] = useState<GalleryFilters | null>(null);
  const [queue, setQueue] = useState<ImageRow[] | null>(null);
  const [index, setIndex] = useState(0);
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [activeClass, setActiveClass] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [help, setHelp] = useState(false); // the keyboard shortcuts list
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const jumped = useRef(false);
  // The AI check: a stored second opinion on this image's boxes. Suggestions are only drawn and listed; accepting one edits the boxes
  // here (unsaved) like drawing would, and Save / Accept & next persists them.
  const [ai, setAi] = useState<AiCheck | null>(null);
  const [aiBusy, setAiBusy] = useState<string | null>(null); // what the AI is doing right now, shown while it works
  const [aiError, setAiError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const shownId = useRef<number | null>(null);

  const targets = useMemo(() => (project?.classes ?? []).filter((c) => !c.is_negative && !c.is_mixed), [project]);
  const classById = useMemo(() => new Map((project?.classes ?? []).map((c) => [c.id, c])), [project]);
  const colorFor = useCallback((id: number) => PALETTE[(classById.get(id)?.index ?? id) % PALETTE.length], [classById]);
  const nameFor = useCallback((id: number) => classById.get(id)?.name ?? String(id), [classById]);

  const current = queue?.[index] ?? null;
  const imageClass = current ? classById.get(current.class_id) : undefined;
  const readOnly = imageClass?.is_negative ?? false;

  useEffect(() => {
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
    setFilters(fromTrash ? loadTrashFilters(projectId) : loadGalleryFilters(projectId));
  }, [projectId, fromTrash]);

  // Load the queue the main page's filters describe.
  useEffect(() => {
    if (!filters) return;
    let cancelled = false;
    api
      .listImages(projectId, { ...(fromTrash ? trashQuery(filters) : imageQuery(filters)), limit: QUEUE_LIMIT, offset: 0 })
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
  }, [projectId, filters, jumpTo, fromTrash]);

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
        shownId.current = d.id;
        setAi(null);
        setAiError(null);
        setDismissed(new Set());
        api.getAiCheck(d.id).then((c) => !cancelled && setAi(c)).catch(() => undefined); // a stored check, if there is one
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

  // Showing an imported image here is what counts as checking it by hand. Repeating the call is harmless.
  const detailId = detail?.id;
  const unchecked = detail?.manually_checked === false;
  useEffect(() => {
    if (detailId === undefined || !unchecked) return;
    api
      .markManuallyChecked(detailId)
      .then(() => {
        setDetail((d) => (d && d.id === detailId ? { ...d, manually_checked: true } : d));
        setQueue((q) => q && q.map((r) => (r.id === detailId ? { ...r, manually_checked: true } : r)));
      })
      .catch((e) => setError(e.message));
  }, [detailId, unchecked]);

  // After a save the stored check is out of date (the server marks it); fetch it again so the page agrees.
  const refreshAi = useCallback((id: number) => {
    api.getAiCheck(id).then((c) => shownId.current === id && setAi(c)).catch(() => undefined);
  }, []);

  const change = useCallback((next: Box[]) => {
    setBoxes(next);
    setDirty(true);
  }, []);

  /** Persist the current boxes. Returns false (and shows the error) if the server refused. */
  const save = useCallback(
    async (newStatus?: ReviewStatus, dismissAi?: boolean): Promise<boolean> => {
      if (!detail) return true;
      setSaving(true);
      try {
        const out = await api.putAnnotations(detail.id, boxes.map(toBoxIn), newStatus, dismissAi);
        setDetail(out);
        setDirty(false);
        setQueue((q) => q && q.map((r) => (r.id === out.id ? { ...r, review_status: out.review_status, class_id: out.class_id, ai_status: out.ai_status } : r)));
        refreshAi(out.id);
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
    [detail, boxes, classById, project, projectId, refreshAi],
  );

  /** Move to another image, saving unsaved edits first so nothing drawn is lost. */
  const go = useCallback(
    async (delta: number) => {
      if (!queue) return;
      const next = index + delta;
      if (next < 0 || next >= queue.length) return;
      if (dirty && !(await save())) return;
      setNotice(null); // it was about the image just left
      setIndex(next);
    },
    [queue, index, dirty, save],
  );

  /** The same two decisions for a hard-negative picture, which never has boxes: "reviewed" = Keep as is (the same save as Accept with no boxes:
   *  the picture counts as reviewed, so a restored one is used for training again, and the AI's doubts are set aside), "rejected" = into the
   *  trash (the gallery's bulk Reject, which leaves the picture's folder alone). */
  const decideNegative = useCallback(
    async (newStatus: ReviewStatus): Promise<boolean> => {
      if (!detail) return true;
      setSaving(true);
      try {
        if (newStatus === "rejected") await api.rejectImages(projectId, [detail.id]);
        else await api.putAnnotations(detail.id, [], "reviewed", true);
        const out = await api.getImage(detail.id);
        setDetail(out);
        setQueue((q) => q && q.map((r) => (r.id === out.id ? { ...r, review_status: out.review_status, ai_status: out.ai_status } : r)));
        refreshAi(out.id);
        setError(null);
        setNotice(null);
        return true;
      } catch (e) {
        setError((e as Error).message);
        return false;
      } finally {
        setSaving(false);
      }
    },
    [detail, projectId, refreshAi],
  );

  /** Save with a status, then step forward. Accepting or rejecting is also the person's decision on the AI's doubts (a flagged image is then
   *  dismissed for these boxes). The decided image stays in this list so Prev can come back to it; it is gone from the list the next time
   *  the review window is opened. */
  const finish = useCallback(
    async (newStatus: ReviewStatus) => {
      const done = readOnly ? await decideNegative(newStatus) : await save(newStatus, true);
      if (!done || !queue || !current) return;
      if (index >= queue.length - 1) setNotice((n) => n ?? "That was the last image in this list. Open the review again to see what is left.");
      setIndex((i) => Math.min(i + 1, queue.length - 1));
    },
    [readOnly, decideNegative, save, queue, current, index],
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

  const allSuggestions = useMemo(() => buildSuggestions(ai?.dismissed ? null : ai), [ai]); // a dismissed check asks for nothing
  const suggestions = useMemo(() => allSuggestions.filter((x) => !dismissed.has(x.key)), [allSuggestions, dismissed]);
  const ghosts = useMemo(() => suggestions.map(ghostFor), [suggestions]);

  /** Run the AI check on the image on screen. The first call loads the models, so it can take a minute. */
  const runAi = useCallback(async () => {
    if (!detail || aiBusy) return;
    const id = detail.id;
    setAiBusy("Checking with the AI… the first check loads the models and can take a minute");
    setAiError(null);
    try {
      const out = await api.aiCheck(id);
      if (shownId.current !== id) return; // the person moved on while it ran
      setAi(out);
      setDismissed(new Set());
      setDetail((d) => (d && d.id === id ? { ...d, ai_status: statusOf(out) } : d));
    } catch (e) {
      setAiError((e as Error).message);
    } finally {
      setAiBusy(null);
    }
  }, [detail, aiBusy]);

  /** Put a dismissed image back in "AI: needs a look". */
  const undismiss = useCallback(async () => {
    if (!detail) return;
    const id = detail.id;
    try {
      const out = await api.aiUndismiss(id);
      if (shownId.current !== id) return;
      setAi(out);
      setDismissed(new Set());
      setDetail((d) => (d && d.id === id ? { ...d, ai_status: statusOf(out) } : d));
      setQueue((q) => q && q.map((r) => (r.id === id ? { ...r, ai_status: statusOf(out) } : r)));
    } catch (e) {
      setAiError((e as Error).message);
    }
  }, [detail]);

  const classForName = useCallback((name?: string | null) => targets.find((c) => c.name.toLowerCase() === name?.toLowerCase())?.id ?? activeClass ?? targets[0]?.id, [targets, activeClass]);

  const accept = useCallback(
    (list: Suggestion[]) => {
      let next = boxes;
      let last: string | null = null;
      const done: string[] = [];
      for (const sug of list) {
        const classId = classForName(sug.className);
        if (classId === undefined) continue;
        const r = applySuggestion(next, sug, classId, () => crypto.randomUUID());
        if (!r) continue;
        next = r.boxes;
        last = r.selected;
        done.push(sug.key);
      }
      if (!done.length) return;
      change(next);
      setSelected(last);
      setDismissed((d) => new Set([...d, ...done]));
    },
    [boxes, change, classForName],
  );

  const dismiss = (key: string) => setDismissed((d) => new Set([...d, key]));

  /** Ask SAM for the tight box around a click, or around the selected box, and put it in the editor. */
  const snap = useCallback(
    async (point?: { x: number; y: number }) => {
      if (!detail || aiBusy || readOnly) return;
      const sel = boxes.find((b) => b.key === selected);
      if (!point && !sel) return;
      const id = detail.id;
      setAiBusy("SAM is looking…");
      setAiError(null);
      try {
        const body = point ? point : { box: [sel!.x1, sel!.y1, sel!.x2, sel!.y2] as AiBox };
        const out = await api.aiBox(id, body);
        if (shownId.current !== id) return;
        if (!out.found || !out.box) {
          setAiError("SAM found no clear object there. Try clicking the middle of the drone, or draw the box by hand.");
          return;
        }
        const [x1, y1, x2, y2] = out.box;
        if (point || !sel) {
          const classId = activeClass ?? targets[0]?.id;
          if (classId === undefined) return;
          const key = crypto.randomUUID();
          change([...boxes, { key, classId, x1, y1, x2, y2 }]);
          setSelected(key);
        } else {
          change(boxes.map((b) => (b.key === sel.key ? { ...b, x1, y1, x2, y2 } : b)));
        }
      } catch (e) {
        setAiError((e as Error).message);
      } finally {
        setAiBusy(null);
      }
    },
    [detail, aiBusy, readOnly, boxes, selected, activeClass, targets, change],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "?") {
        e.preventDefault();
        setHelp((h) => !h);
        return;
      }
      if (help) {
        // While the list is open nothing else reacts: Enter must not accept the image behind it, nor X reject it, nor Delete remove a box.
        e.preventDefault();
        if (e.key === "Escape") setHelp(false);
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        removeSelected();
      } else if (e.key === "Escape") setSelected(null);
      else if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
      else if (e.key === "Enter") {
        e.preventDefault();
        finish("reviewed");
      } else if (e.key === "x") finish("rejected");
      else if (e.key.toLowerCase() === "a" && !readOnly) runAi();
      else if (e.key.toLowerCase() === "s" && !readOnly) snap();
      else if (/^[1-9]$/.test(e.key) && targets[Number(e.key) - 1]) setBoxClass(targets[Number(e.key) - 1].id);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [removeSelected, go, finish, setBoxClass, targets, readOnly, runAi, snap, help]);

  const selectedBox = boxes.find((b) => b.key === selected);

  // The review screen never scrolls: it is the whole window (the image on the left, every control in the panel on the right), so the page
  // behind it is locked while it is open.
  useEffect(() => {
    const root = document.documentElement;
    const before = root.style.overflow;
    root.style.overflow = "hidden";
    return () => {
      root.style.overflow = before;
    };
  }, []);

  return (
    <div className="fixed inset-0 z-30 flex flex-col gap-2 overflow-hidden bg-[var(--background)] p-3" data-testid="review-root">
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1">
        <Link href={`/projects/${projectId}/${fromTrash ? "trash" : "images"}`} className="text-sm text-neutral-500 hover:underline" data-testid="review-back">
          ← {fromTrash ? "Trash" : project?.name ?? "Project"}
        </Link>
        <h1 className="text-xl font-semibold">Review</h1>
        {/* Read-only: the filters are set on the Images page (or the Trash page). */}
        <span className="text-sm text-neutral-500" data-testid="filter-summary">
          Showing {filters ? (fromTrash ? `the trash${describeFilters(filters, nameFor) === "all images" ? "" : ` · ${describeFilters(filters, nameFor)}`}` : describeFilters(filters, nameFor)) : "…"}
        </span>
        {queue && queue.length > 0 && (
          <span className="text-sm text-neutral-500" data-testid="position">
            {index + 1} / {queue.length}
            {queue.length >= QUEUE_LIMIT && "+"}
          </span>
        )}
        <button className="btn ml-auto" onClick={() => setHelp(true)} title="Keyboard shortcuts" data-testid="shortcuts-button">
          Shortcuts <kbd className="ml-1 opacity-60">?</kbd>
        </button>
      </div>
      {help && <ShortcutsHelp onClose={() => setHelp(false)} />}

      {error && <p className="shrink-0 text-sm text-red-600">{error}</p>}
      {notice && !error && <p className="shrink-0 text-sm text-neutral-400" data-testid="moved-notice">{notice}</p>}
      {queue === null && !error && <p className="shrink-0 text-sm text-neutral-500">Loading…</p>}
      {queue?.length === 0 && (
        <p className="shrink-0 text-neutral-500">
          {fromTrash ? "Nothing in the trash matches the Trash page's filters." : "Nothing matches the Images page's filters."}{" "}
          <button
            className="underline"
            onClick={() => {
              (fromTrash ? saveTrashFilters : saveGalleryFilters)(projectId, NO_FILTERS);
              setFilters(NO_FILTERS);
            }}
          >
            {fromTrash ? "Show the whole trash" : "Show all images"}
          </button>
        </p>
      )}

      {current && detail && detail.id === current.id && (
        <div className="flex min-h-0 flex-1 flex-col gap-3 md:flex-row">
          <div className="min-h-0 min-w-0 flex-1">
            <BoxEditor
              src={imageUrl(detail.id)}
              imageWidth={detail.width}
              imageHeight={detail.height}
              boxes={boxes}
              selectedKey={selected}
              activeClassId={activeClass}
              disabled={readOnly}
              colorFor={colorFor}
              nameFor={nameFor}
              onChange={change}
              onSelect={setSelected}
              ghosts={ghosts}
              onAltClick={readOnly ? undefined : (x, y) => snap({ x, y })}
            />
          </div>

          {/* Everything you press, always on screen. Only the list of AI suggestions can scroll, inside its own box. */}
          <aside className="flex min-h-0 w-full shrink-0 flex-col gap-3 overflow-y-auto short:gap-2 max-md:max-h-[50%] md:w-72 lg:w-[22rem]" data-testid="review-panel">
            <div className="space-y-1 text-sm">
              <p className="text-neutral-500">
                #{detail.id} · filed under <b>{nameFor(detail.class_id)}</b> · {detail.dataset ? `imported · ${detail.dataset}` : detail.folder ? `uploaded · ${detail.folder}` : detail.source} · {detail.width}×{detail.height}
              </p>
              <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className={{ pending: "text-amber-500", reviewed: "text-green-500", rejected: "text-red-500" }[detail.review_status]} data-testid="review-status">
                  {detail.review_status}
                </span>
                {detail.manually_checked && <span className="text-green-500" data-testid="manual-check">Manual check ✓</span>}
                {detail.ai_labelled && (
                  <span className="text-blue-500" data-testid="ai-labelled-note" title="The AI drew these boxes. Fix any that are off, then Accept to keep them; until then they are not used for training.">
                    AI drew these boxes · not confirmed
                  </span>
                )}
                {ai && (
                  <span
                    className={ai.stale || ai.dismissed ? "text-neutral-500" : ai.verdict === "ok" || ai.verdict === "none" ? "text-green-500" : ai.verdict === "too_small" ? "text-neutral-500" : "text-amber-500"}
                    data-testid="ai-status"
                    title={`${ai.model} · ${ai.checked_at}`}
                  >
                    {ai.stale ? "AI check out of date" : ai.dismissed ? "AI: dismissed" : ai.verdict === "ok" ? "AI ✓" : ai.verdict === "none" ? "AI: nothing found" : ai.verdict === "too_small" ? "AI: too small to judge" : "AI: needs a look"}
                  </span>
                )}
                {dirty && <span className="rounded bg-amber-500/20 px-1.5 text-amber-500">unsaved</span>}
              </p>
            </div>

            <div className="grid shrink-0 grid-cols-2 gap-2">
              {readOnly ? (
                <>
                  <button className="btn col-span-2 bg-green-700/30 py-2 font-medium short:py-1" onClick={() => finish("reviewed")} disabled={saving} data-testid="keep-as-is" title="Keep this picture as it is and set the AI's doubts about it aside (Enter)">
                    Keep as is &amp; next
                  </button>
                  <button className="btn col-span-2" onClick={() => finish("rejected")} disabled={saving} data-testid="reject-negative" title="Send this picture to the trash (X)">
                    Reject image
                  </button>
                </>
              ) : (
                <>
                  <button className="btn col-span-2 bg-green-700/30 py-2 font-medium short:py-1" onClick={() => finish("reviewed")} disabled={saving}>
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
              <button className="btn" onClick={() => go(-1)} disabled={saving || index === 0}>
                ← Prev
              </button>
              <button className="btn" onClick={() => go(1)} disabled={saving || index >= queue!.length - 1}>
                Next →
              </button>
            </div>

            {readOnly ? (
              <p className="text-sm text-neutral-500">This is a hard-negative image: it never has boxes, so there is nothing to draw. Keep it or reject it.</p>
            ) : (
              <div className="shrink-0 space-y-1.5">
                <p className="text-sm text-neutral-500">Draw as:</p>
                <div className="flex flex-wrap items-center gap-1.5">
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
                </div>
                {/* Always rendered: a button that appears on selection would change the panel's height and shift things mid-drag. */}
                <button className="btn w-full" onClick={removeSelected} disabled={!selectedBox}>
                  {selectedBox ? `Delete box (${nameFor(selectedBox.classId)})` : "Delete box"}
                </button>
              </div>
            )}

            {!readOnly && (
              <div className="shrink-0 space-y-1.5" data-testid="ai-row">
                <p className="text-sm text-neutral-500">AI:</p>
                <div className="grid grid-cols-2 gap-2">
                  <button className="btn" onClick={runAi} disabled={!!aiBusy} title="Check this image's boxes with the AI (A)">
                    <kbd className="mr-1 opacity-60">A</kbd>
                    {ai && !ai.stale ? "Check again" : "AI check"}
                  </button>
                  <button className="btn" onClick={() => snap()} disabled={!!aiBusy || !selectedBox} title="Snap the selected box tight around the object in it (S)">
                    <kbd className="mr-1 opacity-60">S</kbd>Snap box
                  </button>
                </div>
                <p className="text-xs text-neutral-500 short:hidden">Alt/Option + click on a drone draws a box around it</p>
                {aiBusy && <p className="text-sm text-blue-500" data-testid="ai-busy">{aiBusy}</p>}
                {aiError && <p className="text-sm text-red-600" data-testid="ai-error">{aiError}</p>}
              </div>
            )}

            {ai?.stale && (
              <p className="shrink-0 text-sm text-neutral-500" data-testid="ai-stale">
                The AI check is out of date: the boxes changed after it ran. Press A to check again.
              </p>
            )}
            {ai?.dismissed && (
              <p className="shrink-0 text-sm text-neutral-500" data-testid="ai-dismissed">
                You reviewed these boxes despite the AI&apos;s doubts, so this image is off the &quot;AI: needs a look&quot; list.{" "}
                <button className="underline" onClick={undismiss}>
                  Put it back in the list
                </button>
              </p>
            )}
            {suggestions.length > 0 && (
              <div className="flex min-h-0 flex-1 flex-col gap-2 rounded border border-neutral-300 p-3 dark:border-neutral-700" data-testid="ai-suggestions">
                <div className="flex shrink-0 flex-wrap items-center gap-2 text-sm">
                  <b>AI suggestions</b>
                  <span className="text-xs text-neutral-500">dashed boxes on the image</span>
                  {!readOnly && suggestions.some((x) => x.box) && (
                    <button className="btn ml-auto" onClick={() => accept(suggestions)} data-testid="ai-accept-all">
                      Accept all
                    </button>
                  )}
                </div>
                <ul className="min-h-0 flex-1 space-y-2 overflow-y-auto text-sm">
                  {suggestions.map((x) => (
                    <li key={x.key} className="space-y-1" data-testid={`ai-suggestion-${x.key}`}>
                      <p className="flex items-start gap-2">
                        <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: KIND_COLOR[x.kind] }} />
                        <span>{x.text}</span>
                      </p>
                      <div className="flex flex-wrap gap-1.5 pl-4">
                        {x.boxKey && !x.box && (
                          <button className="btn" onClick={() => setSelected(x.boxKey!)}>
                            Select box
                          </button>
                        )}
                        {x.box && !readOnly && (
                          <button className="btn bg-green-700/30" onClick={() => accept([x])}>
                            {x.kind === "missing" ? "Add box" : "Use this box"}
                          </button>
                        )}
                        <button className="btn" onClick={() => dismiss(x.key)}>
                          Dismiss
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
                <p className="shrink-0 text-xs text-neutral-500" title={ai?.model}>
                  {readOnly
                    ? "Keep as is sets these doubts aside; Reject sends the picture to the trash. The AI is a second opinion, not the answer."
                    : "Nothing is saved until you Save or Accept. The AI is a second opinion, not the answer."}
                </p>
              </div>
            )}
            {ai && !ai.stale && !ai.dismissed && suggestions.length === 0 && (
              <p className="shrink-0 text-sm text-neutral-500" data-testid="ai-clear">
                {allSuggestions.length > 0
                  ? "You have been through all of the AI's suggestions. Save to keep your changes."
                  : ai.verdict === "none"
                    ? "The AI found nothing in this image and it has no boxes."
                    : ai.result.summary.too_small
                      ? "Nothing to review. The AI could not judge some boxes: they are too small."
                      : "Nothing for you to review: the AI agrees with these boxes."}
              </p>
            )}
            <p className="mt-auto shrink-0 text-xs text-neutral-500 short:hidden">Press ? for every keyboard shortcut.</p>
          </aside>
        </div>
      )}
    </div>
  );
}
