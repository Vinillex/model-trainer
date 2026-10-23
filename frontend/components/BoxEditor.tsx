"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

/** A box as normalized corners (0..1 of the image), which is what the pointer maths works in. */
export interface Box {
  key: string;
  classId: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** A suggestion drawn over the image (dashed, not editable): the AI check's tighter box, or a drone it found that has no label. */
export interface Ghost {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  label: string;
}

type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const HANDLE_POS: Record<Handle, [number, number]> = {
  nw: [0, 0], n: [50, 0], ne: [100, 0], e: [100, 50], se: [100, 100], s: [50, 100], sw: [0, 100], w: [0, 50],
};
const HANDLE_CURSOR: Record<Handle, string> = {
  nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
};

// A draw drag shorter than this many screen pixels is a click, not a box. In pixels rather than a share of the image, so zooming
// in is what lets you draw a smaller box.
const MIN_DRAW_PX = 4;
const MIN_SIZE = 0.005;
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

const MIN_ZOOM = 1; // 1 = the whole image fits in the view
const MAX_ZOOM = 12;
const ZOOM_STEP = 1.5;
const MAX_FIT = 2; // a small image may be shown up to twice its size to fill the space it is given (zoom 100% = this fit)
// Space around the image inside the scrolling view: the top one leaves room for a box label above a box at the image's edge.
const PAD_X = 4;
const PAD_TOP = 24;
const PAD_BOTTOM = 4;

type Drag =
  | { mode: "draw"; key: string; sx: number; sy: number }
  | { mode: "move"; key: string; sx: number; sy: number; orig: Box }
  | { mode: "resize"; key: string; handle: Handle; orig: Box };

interface Props {
  src: string;
  imageWidth: number;
  imageHeight: number;
  boxes: Box[];
  selectedKey: string | null;
  activeClassId: number | null;
  disabled?: boolean;
  colorFor: (classId: number) => string;
  nameFor: (classId: number) => string;
  onChange: (boxes: Box[]) => void;
  onSelect: (key: string | null) => void;
  /** Suggestions drawn on top of the boxes. They take no pointer events, so drawing and moving boxes work through them. */
  ghosts?: Ghost[];
  /** Alt/Option + click on the image (a click-to-box request), with the point as shares of the image. */
  onAltClick?: (x: number, y: number) => void;
}

export default function BoxEditor({ src, imageWidth, imageHeight, boxes, selectedKey, activeClassId, disabled, colorFor, nameFor, onChange, onSelect, ghosts, onAltClick }: Props) {
  const measure = useRef<HTMLDivElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const boxesRef = useRef(boxes);
  boxesRef.current = boxes;

  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [avail, setAvail] = useState({ w: 0, h: 0 });
  const availWidth = avail.w;
  const availHeight = avail.h;

  // The editor fills the space its parent gives it (the review screen never scrolls): at zoom 1 the whole image fits that space,
  // shrunk as needed and enlarged at most MAX_FIT times.
  const fit = Math.min(MAX_FIT, (availWidth - 2 * PAD_X) / imageWidth, (availHeight - PAD_TOP - PAD_BOTTOM) / imageHeight);
  const baseW = Math.max(1, imageWidth * fit);
  const baseH = Math.max(1, imageHeight * fit);
  const stageW = baseW * zoom;
  const stageH = baseH * zoom;
  const contentW = stageW + 2 * PAD_X;
  const contentH = stageH + PAD_TOP + PAD_BOTTOM;

  // The zoom maths runs from event handlers, so it reads these instead of closing over stale render values.
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;
  const baseRef = useRef({ w: baseW, h: baseH });
  baseRef.current = { w: baseW, h: baseH };

  useLayoutEffect(() => {
    const el = measure.current!;
    const read = () => setAvail((a) => (a.w === el.clientWidth && a.h === el.clientHeight ? a : { w: el.clientWidth, h: el.clientHeight }));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // A new image starts fitted again.
  useEffect(() => {
    setZoom(MIN_ZOOM);
    zoomRef.current = MIN_ZOOM;
    viewport.current?.scrollTo(0, 0);
  }, [src]);

  /** Zoom keeping the point under (clientX, clientY), or the middle of the view, where it is. */
  const zoomTo = useCallback((next: number, clientX?: number, clientY?: number) => {
    const vp = viewport.current;
    const from = zoomRef.current;
    const to = clamp(next, MIN_ZOOM, MAX_ZOOM);
    if (!vp || to === from) return;
    const r = vp.getBoundingClientRect();
    const ax = (clientX ?? r.left + r.width / 2) - r.left;
    const ay = (clientY ?? r.top + r.height / 2) - r.top;
    const { w, h } = baseRef.current;
    // Where the anchor sits on the image, as a share of it, so it can be put back under the same point after the resize.
    const fx = (vp.scrollLeft + ax - PAD_X) / (w * from);
    const fy = (vp.scrollTop + ay - PAD_TOP) / (h * from);
    zoomRef.current = to;
    // Render now rather than on the next frame: one wheel tick sends several events, and each must measure a view that already
    // has the previous step applied.
    flushSync(() => setZoom(to));
    vp.scrollLeft = PAD_X + fx * w * to - ax;
    vp.scrollTop = PAD_TOP + fy * h * to - ay;
  }, []);

  // Scrolling over the image zooms at the cursor, with no key held; moving around is the middle-button drag below. A trackpad
  // pinch arrives as a ctrl+wheel with much smaller steps, so it gets a bigger factor.
  useEffect(() => {
    const vp = viewport.current!;
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY === 0) return; // a sideways swipe still scrolls the view
      e.preventDefault();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY; // some mice report lines, not pixels
      zoomTo(zoomRef.current * Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0025)), e.clientX, e.clientY);
    };
    vp.addEventListener("wheel", onWheel, { passive: false }); // React's onWheel is passive, so it can't stop the page scrolling
    return () => vp.removeEventListener("wheel", onWheel);
  }, [zoomTo]);

  // Hold the middle (scroll) button and drag to move the image around inside the view.
  const pan = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const [panning, setPanning] = useState(false);
  function onPanStart(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 1) return;
    const vp = e.currentTarget;
    pan.current = { x: e.clientX, y: e.clientY, left: vp.scrollLeft, top: vp.scrollTop };
    try {
      vp.setPointerCapture(e.pointerId); // keep panning if the cursor leaves the view
    } catch {
      // the pointer already ended: the pan simply stops with it
    }
    setPanning(true);
    e.preventDefault();
  }
  function onPanMove(e: React.PointerEvent<HTMLDivElement>) {
    const p = pan.current;
    if (!p) return;
    e.currentTarget.scrollLeft = p.left - (e.clientX - p.x);
    e.currentTarget.scrollTop = p.top - (e.clientY - p.y);
  }
  function onPanEnd() {
    if (!pan.current) return;
    pan.current = null;
    setPanning(false);
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "+" || e.key === "=") zoomTo(zoomRef.current * ZOOM_STEP);
      else if (e.key === "-" || e.key === "_") zoomTo(zoomRef.current / ZOOM_STEP);
      else if (e.key === "0") zoomTo(MIN_ZOOM);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomTo]);

  const norm = (e: React.PointerEvent) => {
    const r = overlay.current!.getBoundingClientRect();
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) };
  };
  // Update the ref now, not on the next render: a quick drag can end before React re-renders, and pointer-up must see the final box.
  const commit = (next: Box[]) => {
    boxesRef.current = next;
    onChange(next);
  };
  const replace = (key: string, patch: Partial<Box>) =>
    commit(boxesRef.current.map((b) => (b.key === key ? { ...b, ...patch } : b)));

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (disabled || e.button !== 0) return;
    const target = e.target as HTMLElement;
    const { x, y } = norm(e);
    if (e.altKey && onAltClick) {
      onAltClick(x, y);
      e.preventDefault();
      return;
    }
    const handle = target.closest<HTMLElement>("[data-handle]")?.dataset.handle as Handle | undefined;
    const boxKey = target.closest<HTMLElement>("[data-box]")?.dataset.box;
    const box = boxesRef.current.find((b) => b.key === boxKey);

    if (handle && box) {
      drag.current = { mode: "resize", key: box.key, handle, orig: box };
    } else if (box) {
      onSelect(box.key);
      drag.current = { mode: "move", key: box.key, sx: x, sy: y, orig: box };
    } else if (activeClassId !== null) {
      const key = crypto.randomUUID();
      drag.current = { mode: "draw", key, sx: x, sy: y };
      commit([...boxesRef.current, { key, classId: activeClassId, x1: x, y1: y, x2: x, y2: y }]);
      onSelect(key);
    } else {
      onSelect(null);
      return;
    }
    overlay.current!.setPointerCapture(e.pointerId); // keep receiving moves if the cursor leaves the image
    e.preventDefault();
  }

  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const d = drag.current;
    if (!d) return;
    const { x, y } = norm(e);
    if (d.mode === "draw") {
      replace(d.key, { x1: Math.min(d.sx, x), y1: Math.min(d.sy, y), x2: Math.max(d.sx, x), y2: Math.max(d.sy, y) });
    } else if (d.mode === "move") {
      const o = d.orig;
      const dx = clamp(x - d.sx, -o.x1, 1 - o.x2);
      const dy = clamp(y - d.sy, -o.y1, 1 - o.y2);
      replace(d.key, { x1: o.x1 + dx, x2: o.x2 + dx, y1: o.y1 + dy, y2: o.y2 + dy });
    } else {
      let { x1, y1, x2, y2 } = d.orig;
      if (d.handle.includes("w")) x1 = x;
      if (d.handle.includes("e")) x2 = x;
      if (d.handle.includes("n")) y1 = y;
      if (d.handle.includes("s")) y2 = y;
      // Dragging an edge past its opposite flips the box instead of inverting it.
      replace(d.key, { x1: Math.min(x1, x2), x2: Math.max(x1, x2), y1: Math.min(y1, y2), y2: Math.max(y1, y2) });
    }
  }

  function onPointerUp() {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    const b = boxesRef.current.find((x) => x.key === d.key);
    if (!b) return;
    const shown = overlay.current!.getBoundingClientRect();
    const tooSmall =
      d.mode === "draw"
        ? (b.x2 - b.x1) * shown.width < MIN_DRAW_PX || (b.y2 - b.y1) * shown.height < MIN_DRAW_PX
        : b.x2 - b.x1 < MIN_SIZE || b.y2 - b.y1 < MIN_SIZE;
    if (tooSmall) {
      // A click on empty space, or a degenerate resize: drop (draw) or restore (resize/move) rather than save a sliver.
      if (d.mode === "draw") {
        commit(boxesRef.current.filter((x) => x.key !== d.key));
        onSelect(null);
      } else {
        replace(d.key, d.orig);
      }
    }
  }

  return (
    <div className="flex h-full min-h-0 w-full select-none flex-col">
      <div className="mb-1 flex shrink-0 flex-wrap items-center gap-1 text-xs text-neutral-500">
        <span>Zoom</span>
        <button className="btn px-2 py-0.5" onClick={() => zoomTo(zoom / ZOOM_STEP)} disabled={zoom <= MIN_ZOOM} aria-label="Zoom out" title="Zoom out (−)">
          −
        </button>
        <span className="w-12 text-center tabular-nums" data-testid="zoom-level">
          {Math.round(zoom * 100)}%
        </span>
        <button className="btn px-2 py-0.5" onClick={() => zoomTo(zoom * ZOOM_STEP)} disabled={zoom >= MAX_ZOOM} aria-label="Zoom in" title="Zoom in (+)">
          +
        </button>
        <button className="btn px-2 py-0.5" onClick={() => zoomTo(MIN_ZOOM)} disabled={zoom === MIN_ZOOM} title="Fit the whole image (0)">
          Fit
        </button>
        <span className="ml-1">Scroll over the image to zoom at the cursor{zoom > MIN_ZOOM ? "; hold the scroll button and drag to move it" : ""}</span>
      </div>
      {/* The space left for the image: measured, so the viewport below is exactly as big as fits (never taller than the screen). */}
      <div ref={measure} className="relative min-h-0 flex-1 overflow-hidden">
      <div
        ref={viewport}
        className="mx-auto overflow-auto"
        style={{ width: Math.min(contentW, availWidth), height: Math.min(contentH, availHeight), colorScheme: "light dark", cursor: panning ? "grabbing" : undefined }}
        onPointerDown={onPanStart}
        onPointerMove={onPanMove}
        onPointerUp={onPanEnd}
        onPointerCancel={onPanEnd}
        onMouseDown={(e) => {
          if (e.button === 1) e.preventDefault(); // otherwise the browser starts its own middle-click auto-scroll
        }}
      >
        <div className="relative" style={{ width: contentW, height: contentH }}>
          <div className="absolute" style={{ left: PAD_X, top: PAD_TOP, width: stageW, height: stageH }}>
            {/* eslint-disable-next-line @next/next/no-img-element -- served by the API */}
            <img src={src} alt="" draggable={false} className="pointer-events-none block h-full w-full select-none" />
            <div
              ref={overlay}
              data-testid="box-overlay"
              className={`absolute inset-0 touch-none ${disabled ? "" : "cursor-crosshair"}`}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            >
              {boxes.map((b) => {
                const color = colorFor(b.classId);
                const selected = b.key === selectedKey;
                return (
                  <div
                    key={b.key}
                    data-box={b.key}
                    className="absolute cursor-move"
                    style={{
                      left: `${b.x1 * 100}%`,
                      top: `${b.y1 * 100}%`,
                      width: `${(b.x2 - b.x1) * 100}%`,
                      height: `${(b.y2 - b.y1) * 100}%`,
                      border: `2px solid ${color}`,
                      background: selected ? `${color}33` : "transparent",
                      boxShadow: selected ? `0 0 0 1px #fff` : undefined,
                    }}
                  >
                    <span className="pointer-events-none absolute -top-5 left-[-2px] whitespace-nowrap rounded px-1 text-[11px] font-medium text-white" style={{ background: color }}>
                      {nameFor(b.classId)}
                    </span>
                    {selected &&
                      HANDLES.map((h) => (
                        <span
                          key={h}
                          data-handle={h}
                          className="absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 border border-white"
                          style={{ left: `${HANDLE_POS[h][0]}%`, top: `${HANDLE_POS[h][1]}%`, background: color, cursor: HANDLE_CURSOR[h] }}
                        />
                      ))}
                  </div>
                );
              })}
              {(ghosts ?? []).map((g) => (
                <div
                  key={g.key}
                  data-testid={`ghost-${g.key}`}
                  className="pointer-events-none absolute"
                  style={{
                    left: `${g.x1 * 100}%`,
                    top: `${g.y1 * 100}%`,
                    width: `${(g.x2 - g.x1) * 100}%`,
                    height: `${(g.y2 - g.y1) * 100}%`,
                    border: `2px dashed ${g.color}`,
                    background: `${g.color}22`,
                  }}
                >
                  <span className="absolute -bottom-5 left-[-2px] whitespace-nowrap rounded px-1 text-[11px] font-medium text-white" style={{ background: g.color }}>
                    {g.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
