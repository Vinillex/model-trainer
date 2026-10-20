"use client";

import { useRef } from "react";

/** A box as normalized corners (0..1 of the image), which is what the pointer maths works in. */
export interface Box {
  key: string;
  classId: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const HANDLE_POS: Record<Handle, [number, number]> = {
  nw: [0, 0], n: [50, 0], ne: [100, 0], e: [100, 50], se: [100, 100], s: [50, 100], sw: [0, 100], w: [0, 50],
};
const HANDLE_CURSOR: Record<Handle, string> = {
  nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
};

const MIN_DRAW = 0.01; // smaller drags are treated as a click, not a box
const MIN_SIZE = 0.005;
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

type Drag =
  | { mode: "draw"; key: string; sx: number; sy: number }
  | { mode: "move"; key: string; sx: number; sy: number; orig: Box }
  | { mode: "resize"; key: string; handle: Handle; orig: Box };

interface Props {
  src: string;
  boxes: Box[];
  selectedKey: string | null;
  activeClassId: number | null;
  disabled?: boolean;
  colorFor: (classId: number) => string;
  nameFor: (classId: number) => string;
  onChange: (boxes: Box[]) => void;
  onSelect: (key: string | null) => void;
}

export default function BoxEditor({ src, boxes, selectedKey, activeClassId, disabled, colorFor, nameFor, onChange, onSelect }: Props) {
  const overlay = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const boxesRef = useRef(boxes);
  boxesRef.current = boxes;

  const norm = (e: React.PointerEvent) => {
    const r = overlay.current!.getBoundingClientRect();
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) };
  };
  const replace = (key: string, patch: Partial<Box>) =>
    onChange(boxesRef.current.map((b) => (b.key === key ? { ...b, ...patch } : b)));

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (disabled || e.button !== 0) return;
    const target = e.target as HTMLElement;
    const { x, y } = norm(e);
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
      onChange([...boxesRef.current, { key, classId: activeClassId, x1: x, y1: y, x2: x, y2: y }]);
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
    const tooSmall = d.mode === "draw" ? b.x2 - b.x1 < MIN_DRAW || b.y2 - b.y1 < MIN_DRAW : b.x2 - b.x1 < MIN_SIZE || b.y2 - b.y1 < MIN_SIZE;
    if (tooSmall) {
      // A click on empty space, or a degenerate resize: drop (draw) or restore (resize/move) rather than save a sliver.
      if (d.mode === "draw") {
        onChange(boxesRef.current.filter((x) => x.key !== d.key));
        onSelect(null);
      } else {
        replace(d.key, d.orig);
      }
    }
  }

  return (
    <div className="relative w-fit max-w-full select-none">
      {/* eslint-disable-next-line @next/next/no-img-element -- served by the API */}
      <img src={src} alt="" draggable={false} className="pointer-events-none block max-h-[65vh] max-w-full select-none" />
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
      </div>
    </div>
  );
}
