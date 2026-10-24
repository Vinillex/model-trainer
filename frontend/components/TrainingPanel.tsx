"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  artifactUrl,
  parseUtc,
  type ArtifactName,
  type DatasetVersion,
  type RunStage,
  type TrainingRun,
  type WorkerInfo,
} from "@/lib/api";
import { PiDisplayBar, usePiDisplay } from "@/components/PiDisplayBar";
import { RunPodDialog, RunPodPanel } from "@/components/RunPodPanel";

const ACTIVE: RunStage[] = ["queued", "training", "exporting", "awaiting_package", "packaging"];
const STAGE_LABEL: Record<RunStage, string> = {
  queued: "Queued",
  training: "Training",
  exporting: "Exporting",
  awaiting_package: "Waiting for Pi",
  packaging: "Packaging",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};
const STAGE_COLOR: Record<RunStage, string> = {
  queued: "bg-neutral-500/20",
  training: "bg-blue-500/30",
  exporting: "bg-blue-500/30",
  awaiting_package: "bg-neutral-500/20",
  packaging: "bg-blue-500/30",
  done: "bg-green-600/30",
  failed: "bg-red-600/30",
  cancelled: "bg-neutral-500/20",
};
const ARTIFACTS: { name: ArtifactName; label: string; hint: string }[] = [
  { name: "rpk", label: "network.rpk", hint: "the model for the IMX500" },
  { name: "labels", label: "labels.txt", hint: "class names, goes next to the .rpk" },
  { name: "best_pt", label: "best.pt", hint: "PyTorch weights" },
  { name: "imx_zip", label: "packerOut.zip", hint: "quantized model before packaging" },
];

const pct = (x: number | null | undefined) => (x == null ? "–" : (x * 100).toFixed(1) + "%");
const size = (n: number | null | undefined) => (n == null ? "" : n < 2 ** 20 ? ` (${Math.max(1, Math.round(n / 1024))} KB)` : ` (${(n / 2 ** 20).toFixed(1)} MB)`);

/** The steps a run goes through, for the little progress trail. */
function steps(exportImx: boolean): { stage: RunStage[]; label: string }[] {
  const trail = [
    { stage: ["queued"] as RunStage[], label: "Queued" },
    { stage: ["training"] as RunStage[], label: "Train" },
  ];
  if (exportImx) {
    trail.push({ stage: ["exporting"], label: "Export" }, { stage: ["awaiting_package", "packaging"], label: "Package (Pi)" });
  }
  trail.push({ stage: ["done"], label: "Done" });
  return trail;
}

function Trail({ run }: { run: TrainingRun }) {
  const trail = steps(run.config.export_imx);
  const current = trail.findIndex((s) => s.stage.includes(run.stage));
  return (
    <ol className="flex flex-wrap items-center gap-1 text-xs" aria-label="progress">
      {trail.map((s, i) => {
        const isDone = run.stage === "done" || (current >= 0 && i < current);
        const isNow = i === current && run.stage !== "done";
        return (
          <li key={s.label} className="flex items-center gap-1">
            {i > 0 && <span className="text-neutral-500">›</span>}
            <span
              className={`rounded px-1.5 py-0.5 ${isNow ? "bg-blue-500/30 font-medium" : isDone ? "bg-green-600/20" : "text-neutral-500"}`}
              data-state={isNow ? "now" : isDone ? "done" : "todo"}
            >
              {isDone ? "✓ " : ""}
              {s.label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/** Which workers have called in, and for the Pi a button that copies the one-time command that makes it an always-on worker. */
function WorkersBar({ workers }: { workers: WorkerInfo[] }) {
  const [copy, setCopy] = useState<{ ok: boolean; text: string } | null>(null);
  const piOnline = workers.some((w) => w.kind === "pi" && w.online);

  async function copySetup() {
    setCopy(null);
    try {
      const { command } = await api.piSetup();
      await navigator.clipboard.writeText(command); // never shown on screen: it holds the worker token
      setCopy({ ok: true, text: "Copied. Paste it once into a terminal on the Raspberry Pi and press Enter." });
    } catch (e) {
      setCopy({ ok: false, text: (e as Error).message || "Could not copy the command." });
    }
  }

  return (
    <div className="space-y-2 text-sm text-neutral-500" data-testid="workers">
      <p>
        Workers:{" "}
        {workers.length === 0 ? (
          "none have connected since the backend started."
        ) : (
          workers.map((w, i) => (
            <span key={w.name}>
              {i > 0 && ", "}
              <span className={w.online ? "text-green-500" : "text-neutral-500"}>●</span> {w.name} ({w.kind === "gpu" ? "GPU" : "Pi"}, {w.online ? "online" : "offline"})
            </span>
          ))
        )}
      </p>
      {piOnline ? (
        <p>The Raspberry Pi packages finished runs into a camera file by itself.</p>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <span>No Raspberry Pi is online. Switch it on; a finished run is packaged as soon as it connects. Never set up?</span>
          <button className="btn" onClick={copySetup} data-testid="copy-pi-setup">
            Copy Pi setup command
          </button>
          {copy && <span className={copy.ok ? "text-green-500" : "text-red-600"}>{copy.text}</span>}
        </div>
      )}
    </div>
  );
}

/** On every finished run: show its model live on the Pi's own screen (and stop it again). */
function RunOnPiButton({ run, pi }: { run: TrainingRun; pi: ReturnType<typeof usePiDisplay> }) {
  const d = pi.display;
  const showingThis = pi.live && d.run_id === run.id;
  if (!run.artifacts.rpk)
    return (
      <button className="btn" disabled title="This run has no camera file (.rpk): Export for IMX500 was off for it." data-testid={`run-${run.id}-pi`}>
        Run on Pi
      </button>
    );
  if (showingThis)
    return (
      <button className="btn" onClick={pi.stop} data-testid={`run-${run.id}-pi`}>
        Stop on Pi
      </button>
    );
  return (
    <button className="btn" onClick={() => pi.show(run.id)} disabled={!d.pi_online} title={d.pi_online ? "Show this model live on the Raspberry Pi's screen" : "No Raspberry Pi is online"} data-testid={`run-${run.id}-pi`}>
      Run on Pi
    </button>
  );
}

function RunCard({ run, datasetName, workers, onChanged, onRunPod, pi }: { run: TrainingRun; datasetName: string; workers: WorkerInfo[]; onChanged: () => void; onRunPod: (run: TrainingRun) => void; pi: ReturnType<typeof usePiDisplay> }) {
  const [confirm, setConfirm] = useState<"cancel" | "delete" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const active = ACTIVE.includes(run.stage);
  const { epoch, epochs, message } = run.progress;
  const final = run.metrics.map50 != null; // the worker reports final validation numbers when training ends
  const shown = final ? run.metrics : run.progress; // the latest epoch's numbers until then
  const online = (kind: "gpu" | "pi") => workers.some((w) => w.kind === kind && w.online);

  let note: string | null = null;
  if (run.cancel_requested && active) note = "Cancelling… the worker stops at its next check-in.";
  else if (run.stage === "queued") note = online("gpu") ? "Waiting for the GPU worker to pick it up." : "No GPU worker is online. Press Train on RunPod to rent one for this run.";
  else if (run.stage === "awaiting_package") note = online("pi") ? "Waiting for the Raspberry Pi to pick it up." : "Training is done. Switch the Raspberry Pi on: it builds the .rpk by itself as soon as it connects.";
  else if (active && message && message !== "training") note = message; // the stage badge already says "Training"
  if (run.attempts > 0 && active) note = `${note ?? ""} (a worker was lost ${run.attempts}×; this run was re-queued)`.trim();

  async function act(kind: "cancel" | "delete") {
    setError(null);
    try {
      if (kind === "cancel") await api.cancelTrainingRun(run.id);
      else await api.deleteTrainingRun(run.id);
      setConfirm(null);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const perClass = run.metrics.per_class ? Object.entries(run.metrics.per_class) : [];
  const files = ARTIFACTS.filter((a) => run.artifacts[a.name]);

  return (
    <li className="space-y-2 rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid={`run-${run.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{run.name}</span>
        <span className={`rounded px-1.5 py-0.5 text-xs ${STAGE_COLOR[run.stage]}`} data-testid={`run-${run.id}-stage`}>
          {STAGE_LABEL[run.stage]}
        </span>
        <span className="text-neutral-500">
          {run.config.model} · {datasetName} · {run.config.epochs} epochs · {run.config.imgsz}px · batch {run.config.batch} · {parseUtc(run.created_at).toLocaleString()}
          {run.worker && <> · on {run.worker}</>}
        </span>
        <span className="ml-auto flex gap-2">
          {confirm ? (
            <>
              <button className="btn bg-red-700/40" onClick={() => act(confirm)}>
                Confirm {confirm}
              </button>
              <button className="btn" onClick={() => setConfirm(null)}>
                Keep
              </button>
            </>
          ) : active ? (
            !run.cancel_requested && (
              <>
                {run.stage === "queued" && !online("gpu") && (
                  <button className="btn" onClick={() => onRunPod(run)} data-testid={`run-${run.id}-runpod`}>
                    Train on RunPod…
                  </button>
                )}
                <button className="btn" onClick={() => setConfirm("cancel")}>
                  Cancel
                </button>
              </>
            )
          ) : (
            <>
              {run.stage === "done" && <RunOnPiButton run={run} pi={pi} />}
              <button className="btn" onClick={() => setConfirm("delete")}>
                Delete
              </button>
            </>
          )}
        </span>
      </div>

      <Trail run={run} />

      {run.stage === "training" && epochs ? (
        <div className="space-y-1">
          <div className="h-1.5 w-full max-w-md overflow-hidden rounded bg-neutral-500/20" role="progressbar" aria-valuenow={epoch ?? 0} aria-valuemax={epochs}>
            <div className="h-full bg-blue-500" style={{ width: `${Math.min(100, ((epoch ?? 0) / epochs) * 100)}%` }} />
          </div>
          <p className="text-xs text-neutral-500">
            Epoch {epoch ?? 0} of {epochs}
          </p>
        </div>
      ) : null}

      {(shown.map50 != null || shown.map50_95 != null) && (
        <p className="text-xs text-neutral-500" data-testid={`run-${run.id}-metrics`}>
          {final ? "Validation" : "Latest epoch"}: mAP50 <b>{pct(shown.map50)}</b> · mAP50-95 {pct(shown.map50_95)} · precision {pct(shown.precision)} · recall {pct(shown.recall)}
        </p>
      )}
      {note && <p className="text-xs text-neutral-500">{note}</p>}
      {run.error && <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded bg-red-600/10 p-2 text-xs text-red-500">{run.error}</pre>}

      {perClass.length > 0 && (
        <table className="w-full max-w-md text-xs">
          <thead>
            <tr className="border-b border-neutral-300 text-left dark:border-neutral-700">
              <th className="py-1 pr-3">Class</th>
              <th className="px-2 text-right">AP50</th>
              <th className="px-2 text-right">AP50-95</th>
            </tr>
          </thead>
          <tbody>
            {perClass.map(([c, m]) => (
              <tr key={c} className="border-b border-neutral-200 dark:border-neutral-800">
                <td className="py-1 pr-3">{c}</td>
                <td className="px-2 text-right tabular-nums">{pct(m.ap50)}</td>
                <td className="px-2 text-right tabular-nums">{pct(m.ap50_95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((a) => (
            <a key={a.name} className="btn" href={artifactUrl(run.id, a.name)} download title={a.hint} data-testid={`run-${run.id}-${a.name}`}>
              {a.label}
              {size(run.artifacts[a.name]?.size)}
            </a>
          ))}
        </div>
      )}
      {error && <p className="text-xs text-red-600">{error}</p>}
    </li>
  );
}

export default function TrainingPanel({ projectId, refreshKey }: { projectId: number; refreshKey: number }) {
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [model, setModel] = useState<"yolo11n" | "yolov8n">("yolo11n");
  const [epochs, setEpochs] = useState(100);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(16);
  const [exportImx, setExportImx] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runpodRun, setRunpodRun] = useState<TrainingRun | null>(null); // the run the RunPod dialog is open for
  const pi = usePiDisplay();
  const [runpodTick, setRunpodTick] = useState(0); // bumped after a launch starts so the panel looks right away
  const datasetIdRef = useRef(datasetId); // lets the loader pick a default without depending on the selection
  datasetIdRef.current = datasetId;

  const loadVersions = useCallback(() => {
    api
      .listDatasets(projectId)
      .then((v) => {
        setVersions(v);
        if (datasetIdRef.current === "" && v.length > 0) setDatasetId(v[0].id);
      })
      .catch((e) => setError(e.message));
  }, [projectId]);
  useEffect(loadVersions, [loadVersions, refreshKey]);

  const refresh = useCallback(async () => {
    try {
      const [r, w] = await Promise.all([api.listTrainingRuns(projectId), api.trainingWorkers()]);
      setRuns(r);
      setWorkers(w);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [projectId]);
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll quickly while something is running; slowly otherwise so the workers' "online" lights stay roughly right.
  const anyActive = runs.some((r) => ACTIVE.includes(r.stage));
  useEffect(() => {
    const t = setInterval(refresh, anyActive ? 2000 : 15000);
    return () => clearInterval(t);
  }, [anyActive, refresh]);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    if (datasetId === "") return;
    setBusy(true);
    setError(null);
    try {
      await api.createTrainingRun(projectId, { dataset_id: datasetId, model, epochs, imgsz, batch, export_imx: exportImx });
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const names = new Map(versions.map((v) => [v.id, v.name]));
  const chosen = versions.find((v) => v.id === datasetId);
  const num = "input w-24";
  return (
    <div className="space-y-4">
      <WorkersBar workers={workers} />
      <PiDisplayBar pi={pi} />
      <RunPodPanel tick={runpodTick} onChanged={refresh} />

      {versions.length === 0 ? (
        <p className="text-sm text-neutral-500">Create a dataset version above first; a training run uses one frozen version.</p>
      ) : (
        <form onSubmit={start} className="flex flex-wrap items-end gap-3 text-sm">
          <label>
            <span className="mb-1 block text-neutral-500">Dataset version</span>
            <select className="input" value={datasetId} onChange={(e) => setDatasetId(Number(e.target.value))}>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name} ({Object.values(v.stats.images).reduce((a, b) => a + b, 0)} images)
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="mb-1 block text-neutral-500">Model</span>
            <select className="input" value={model} onChange={(e) => setModel(e.target.value as "yolo11n" | "yolov8n")}>
              <option value="yolo11n">YOLO11n</option>
              <option value="yolov8n">YOLOv8n</option>
            </select>
          </label>
          <label>
            <span className="mb-1 block text-neutral-500">Epochs</span>
            <input type="number" className={num} min={1} max={1000} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
          </label>
          <label>
            <span className="mb-1 block text-neutral-500">Image size</span>
            <select className="input" value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>
              {[320, 416, 512, 640].map((s) => (
                <option key={s} value={s}>
                  {s}px
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="mb-1 block text-neutral-500">Batch</span>
            <input type="number" className={num} min={1} max={128} value={batch} onChange={(e) => setBatch(Number(e.target.value))} />
          </label>
          <label className="flex items-center gap-1 pb-2">
            <input type="checkbox" checked={exportImx} onChange={(e) => setExportImx(e.target.checked)} /> Export for IMX500 (.rpk)
          </label>
          <button className="btn" disabled={busy || datasetId === ""}>
            {busy ? "Starting…" : "Start training"}
          </button>
        </form>
      )}
      {chosen && chosen.stats.images.val === 0 && (
        <p className="text-sm text-amber-500">This version has no validation images, so a run on it will be refused. Review more images and create a new version.</p>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {runpodRun && (
        <RunPodDialog
          run={runpodRun}
          onClose={() => setRunpodRun(null)}
          onStarted={() => {
            setRunpodRun(null);
            setRunpodTick((t) => t + 1);
            refresh();
          }}
        />
      )}

      <ul className="space-y-3">
        {runs.map((r) => (
          <RunCard key={r.id} run={r} datasetName={names.get(r.dataset_id) ?? `#${r.dataset_id}`} workers={workers} onRunPod={setRunpodRun} pi={pi} onChanged={() => { refresh(); loadVersions(); }} />
        ))}
      </ul>
    </div>
  );
}
