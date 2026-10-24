"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type PiDisplay } from "@/lib/api";

const NONE: PiDisplay = { state: "none", pi_online: false, request_id: null, run_id: null, run_name: null, message: "", camera_fps: null, results_per_s: null, detections: null, threshold: null, fps: null };
const LIVE: PiDisplay["state"][] = ["waiting", "starting", "running"];

/** The live view on the Pi's screen: its state from the backend (which has it from the Pi agent), and the two actions. */
export function usePiDisplay() {
  const [display, setDisplay] = useState<PiDisplay>(NONE);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.piDisplay().then(setDisplay).catch(() => undefined); // decoration: keep what is shown
  }, []);
  useEffect(refresh, [refresh]);

  const live = LIVE.includes(display.state);
  useEffect(() => {
    const t = setInterval(() => document.visibilityState === "visible" && refresh(), live ? 2000 : 10000);
    return () => clearInterval(t);
  }, [live, refresh]);

  async function run(call: () => Promise<PiDisplay>) {
    setError(null);
    try {
      setDisplay(await call());
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return { display, error, live, show: (runId: number) => run(() => api.showOnPi(runId)), stop: () => run(() => api.stopOnPi()) };
}

/** One line (and Stop) about what the Pi's screen is showing; nothing when nothing is. */
export function PiDisplayBar({ pi }: { pi: ReturnType<typeof usePiDisplay> }) {
  const { display: d, error, live } = pi;
  const [dismissed, setDismissed] = useState<string | null>(null);
  const name = d.run_name ?? "the model";

  if (d.state === "none" && !error) return null;
  if (!live && d.request_id && dismissed === d.request_id && !error) return null;

  let text: React.ReactNode;
  let tone = "text-neutral-500";
  if (d.state === "waiting") text = `Waiting for the Pi to pick up ${name}…`;
  else if (d.state === "starting") text = `${name}: ${d.message || "loading the model onto the camera chip…"}`;
  else if (d.state === "running")
    text = (
      <>
        Showing <b>{name}</b> on the Pi&apos;s screen · camera {d.camera_fps?.toFixed(1) ?? "?"} fps · network {d.results_per_s?.toFixed(1) ?? "?"} results/s · {d.detections ?? 0} detection
        {d.detections === 1 ? "" : "s"} now
      </>
    );
  else if (d.state === "failed" || d.state === "ended") {
    text = d.message;
    tone = "text-amber-500";
  } else if (d.state === "stopped") text = `${name}: ${d.message || "stopped"}`;

  return (
    <div className="flex flex-wrap items-center gap-3 rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid="pi-display">
      <span className="font-medium">Pi screen</span>
      {text && (
        <span className={tone} data-testid="pi-display-text">
          {text}
        </span>
      )}
      {error && <span className="text-red-600">{error}</span>}
      <span className="ml-auto flex gap-2">
        {live ? (
          <button className="btn" onClick={pi.stop} data-testid="pi-display-stop">
            Stop
          </button>
        ) : (
          d.request_id && (
            <button className="btn" onClick={() => setDismissed(d.request_id)}>
              Hide
            </button>
          )
        )}
      </span>
      {d.state === "running" && <p className="w-full text-xs text-neutral-500">Press Stop here to end it and free the camera: closing the window on the Pi does not.</p>}
    </div>
  );
}
