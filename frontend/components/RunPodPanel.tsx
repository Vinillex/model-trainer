"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, parseUtc, type RunPodCloud, type RunPodLaunch, type RunPodPlan, type RunPodStatus, type TrainingRun } from "@/lib/api";

const money = (n: number) => `$${n.toFixed(2)}`;
const gpuName = (id: string) => id.replace(/^NVIDIA /, "").replace(/^GeForce /, "");
const HIDDEN_KEY = "runpod-hidden-launch";

function readHidden(): string | null {
  try {
    return localStorage.getItem(HIDDEN_KEY);
  } catch {
    return null;
  }
}

/** How a finished launch ended, in words. Exit 0 = the GPU part of the run is over; 130 = stopped by the person (Ctrl-C path). */
function outcome(l: RunPodLaunch): { text: string; good: boolean } {
  if (l.exit_code === 0) return { text: "Finished: the GPU part of the run is over and the pod was deleted.", good: true };
  if (l.exit_code === 130) return { text: "Stopped. The pod was deleted and the tunnel closed.", good: false };
  if (l.lost_track) return { text: "The backend was restarted while this ran, so how it ended is only in the log below.", good: false };
  return { text: "Ended without finishing the run. The log below says why; nothing is left running unless it says so.", good: false };
}

/** RunPod at a glance: balance, any pod that is running, and the latest launch with its live log and a Stop button. */
export function RunPodPanel({ tick, onChanged }: { tick: number; onChanged: () => void }) {
  const [status, setStatus] = useState<RunPodStatus | null>(null);
  const [launch, setLaunch] = useState<RunPodLaunch | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<"stop" | string | null>(null); // "stop", or the id of the pod to delete
  const [busy, setBusy] = useState(false);
  const [hidden, setHidden] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => setHidden(readHidden()), []);

  const loadStatus = useCallback(() => {
    api.runpodStatus().then(setStatus).catch((e) => setError(e.message));
  }, []);
  const loadLaunch = useCallback(() => {
    api.runpodLaunch().then(setLaunch).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    loadStatus();
    loadLaunch();
  }, [tick, loadStatus, loadLaunch]);

  // The log moves every few seconds while a launch runs; the account and pods change slowly.
  const running = launch?.state === "running";
  useEffect(() => {
    const t = setInterval(() => document.visibilityState === "visible" && loadLaunch(), running ? 2000 : 15000);
    return () => clearInterval(t);
  }, [running, loadLaunch]);
  useEffect(() => {
    const t = setInterval(() => document.visibilityState === "visible" && loadStatus(), running ? 10000 : 30000);
    return () => clearInterval(t);
  }, [running, loadStatus]);

  // keep the newest line in view unless the person scrolled up to read
  useEffect(() => {
    const el = logRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [launch?.lines]);

  // When a launch ends the run, the pod and the balance have changed.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) {
      loadStatus();
      onChanged();
    }
    wasRunning.current = running;
  }, [running, loadStatus, onChanged]);

  async function act(what: "stop" | string) {
    setBusy(true);
    setError(null);
    try {
      if (what === "stop") setLaunch(await api.stopRunpodLaunch());
      else await api.deleteRunpodPod(what);
      setConfirm(null);
      loadStatus();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function hide(startedAt: string) {
    setHidden(startedAt);
    try {
      localStorage.setItem(HIDDEN_KEY, startedAt);
    } catch {
      // not remembered: it shows again after a reload
    }
  }

  if (!status) return error ? <p className="text-sm text-red-600">{error}</p> : null;
  if (!status.key_set) {
    return (
      <p className="text-sm text-neutral-500" data-testid="runpod-panel">
        RunPod (rented GPUs) is not set up: it needs <code>MT_RUNPOD_API_KEY</code> in <code>backend/.env</code>.
      </p>
    );
  }

  const strayPods = !running ? status.pods : [];
  const showLaunch = launch && launch.state !== "none" && (running || launch.started_at !== hidden);
  const result = launch && launch.state === "finished" ? outcome(launch) : null;

  return (
    <section className="space-y-3 rounded border border-neutral-300 p-3 text-sm dark:border-neutral-700" data-testid="runpod-panel">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-medium">RunPod</span>
        {status.error ? (
          <span className="text-amber-500">can&apos;t reach RunPod right now: {status.error}</span>
        ) : (
          <span className="text-neutral-500" data-testid="runpod-account">
            balance {status.balance != null ? money(status.balance) : "?"} · spending {money(status.spend_per_hr ?? 0)}/h now ·{" "}
            {status.pods.length === 0 ? "no pods running" : `${status.pods.length} pod${status.pods.length > 1 ? "s" : ""} running`}
          </span>
        )}
        {!status.cloudflared && <span className="text-amber-500">cloudflared is not installed (brew install cloudflared): a launch can&apos;t start</span>}
        <button className="btn ml-auto" onClick={loadStatus}>
          Refresh
        </button>
      </div>

      {strayPods.map((p) => (
        <div key={p.id} className="flex flex-wrap items-center gap-3 rounded bg-red-700/10 p-2" data-testid={`runpod-pod-${p.id}`}>
          <span className="text-red-500">
            A pod is running and no launch is watching it: {p.name ?? p.id} ({p.status?.toLowerCase()}), {money(p.cost_per_hr)}/h until it is deleted.
          </span>
          <span className="ml-auto flex gap-2">
            {confirm === p.id ? (
              <>
                <button className="btn bg-red-700/40" disabled={busy} onClick={() => act(p.id)}>
                  Confirm delete pod
                </button>
                <button className="btn" onClick={() => setConfirm(null)}>
                  Keep
                </button>
              </>
            ) : (
              <button className="btn" onClick={() => setConfirm(p.id)}>
                Delete pod
              </button>
            )}
          </span>
        </div>
      ))}

      {showLaunch && launch && (
        <div className="space-y-2" data-testid="runpod-launch">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-medium">
              {running ? (launch.stopping ? "Stopping…" : "Launch running") : "Last launch"} · run {launch.run_id}
            </span>
            {launch.started_at && <span className="text-neutral-500">started {parseUtc(launch.started_at).toLocaleString()}</span>}
            {launch.max_cost != null && <span className="text-neutral-500">agreed up to {money(launch.max_cost)}</span>}
            <span className="ml-auto flex gap-2">
              {running && !launch.stopping &&
                (confirm === "stop" ? (
                  <>
                    <button className="btn bg-red-700/40" disabled={busy} onClick={() => act("stop")}>
                      Confirm stop
                    </button>
                    <button className="btn" onClick={() => setConfirm(null)}>
                      Keep going
                    </button>
                  </>
                ) : (
                  <button className="btn" onClick={() => setConfirm("stop")}>
                    Stop and delete the pod
                  </button>
                ))}
              {!running && launch.started_at && (
                <button className="btn" onClick={() => hide(launch.started_at as string)}>
                  Hide
                </button>
              )}
            </span>
          </div>
          {running && <p className="text-neutral-500">Keep this Mac on while it runs. The pod is deleted when the run ends, fails or is stopped; the tunnel closes with it.</p>}
          {result && <p className={result.good ? "text-green-500" : "text-amber-500"}>{result.text}</p>}
          <pre ref={logRef} className="max-h-56 overflow-auto rounded bg-neutral-100 p-2 text-xs dark:bg-neutral-900" data-testid="runpod-log">
            {launch.lines.length ? launch.lines.join("\n") : "Starting…"}
          </pre>
        </div>
      )}
      {error && <p className="text-red-600">{error}</p>}
    </section>
  );
}

/** The price check and the explicit yes, before anything is rented. Nothing starts until the last button is pressed. */
export function RunPodDialog({ run, onClose, onStarted }: { run: TrainingRun; onClose: () => void; onStarted: () => void }) {
  const [hours, setHours] = useState(3);
  const [cloud, setCloud] = useState<RunPodCloud>("secure");
  const [plan, setPlan] = useState<RunPodPlan | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);
  const [starting, setStarting] = useState(false);
  const seq = useRef(0);

  const check = useCallback(() => {
    const mine = ++seq.current;
    setChecking(true);
    setPlan(null);
    setProblem(null);
    api
      .runpodPlan(run.id, hours, cloud)
      .then((p) => mine === seq.current && setPlan(p))
      .catch((e) => mine === seq.current && setProblem(e.message))
      .finally(() => mine === seq.current && setChecking(false));
  }, [run.id, hours, cloud]);

  // price the run on opening and whenever the limit or the cloud change (a moment after typing stops)
  useEffect(() => {
    if (!(hours > 0 && hours <= 12)) {
      seq.current++;
      setPlan(null);
      setChecking(false);
      setProblem("The time limit must be more than 0 and at most 12 hours.");
      return;
    }
    const t = setTimeout(check, 400);
    return () => clearTimeout(t);
  }, [check, hours]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !starting && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, starting]);

  async function start() {
    if (!plan) return;
    setStarting(true);
    setProblem(null);
    try {
      await api.startRunpodLaunch({ run_id: plan.run_id, max_hours: plan.max_hours, cloud: plan.cloud, max_cost: plan.worst_case });
      onStarted();
    } catch (e) {
      setProblem((e as Error).message);
      setStarting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-label="Train on RunPod" data-testid="runpod-dialog">
      <div className="absolute inset-0 bg-black/60" onClick={() => !starting && onClose()} />
      <div className="relative max-h-full w-full max-w-xl space-y-4 overflow-y-auto rounded-lg border border-neutral-300 bg-white p-5 text-sm shadow-xl dark:border-neutral-700 dark:bg-neutral-950">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Train {run.name} on RunPod</h2>
          <button className="btn" onClick={onClose} disabled={starting} aria-label="Close">
            Close (Esc)
          </button>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <label>
            <span className="mb-1 block text-neutral-500">Time limit (hours)</span>
            <input type="number" className="input w-24" min={0.5} max={12} step={0.5} value={hours} onChange={(e) => setHours(Number(e.target.value))} data-testid="runpod-hours" />
          </label>
          <label>
            <span className="mb-1 block text-neutral-500">Cloud</span>
            <select className="input" value={cloud} onChange={(e) => setCloud(e.target.value as RunPodCloud)} data-testid="runpod-cloud">
              <option value="secure">Secure (steadier)</option>
              <option value="community">Community (cheaper)</option>
            </select>
          </label>
          <button className="btn" onClick={check} disabled={checking || starting}>
            Check again
          </button>
        </div>

        {checking && <p className="text-neutral-500">Checking the price, your balance and that everything the pod needs is available…</p>}
        {problem && (
          <p className="text-amber-500" data-testid="runpod-problem">
            {problem}
          </p>
        )}

        {plan && (
          <div className="space-y-3" data-testid="runpod-plan">
            <div>
              <p className="text-neutral-500">GPU: the first of these that has stock is rented.</p>
              <ol className="mt-1 list-inside list-decimal">
                {plan.gpus.map((g) => (
                  <li key={g.id}>
                    {gpuName(g.id)} {g.price != null && <span className="text-neutral-500">{money(g.price)}/h</span>}
                  </li>
                ))}
              </ol>
            </div>
            <p>
              <span className="font-medium">The most this can cost: {money(plan.worst_case)}</span>{" "}
              <span className="text-neutral-500">
                ({plan.max_hours} h limit at the dearest GPU, {money(plan.worst_price)}/h). You pay only while the pod exists; it is deleted when the run ends, and a normal run costs
                far less than the limit. Your RunPod balance is {money(plan.balance)}.
              </span>
            </p>
            <div className="text-neutral-500">
              <p className="mb-1">What leaves your Mac:</p>
              <ul className="list-inside list-disc space-y-0.5">
                <li>
                  the {plan.images.toLocaleString()} images of this run&apos;s dataset version,{" "}
                  {plan.bucket ? "as a zip in your private dataset bucket (uploaded first if it isn't there yet; the pod gets a link valid for 6 hours)" : "through a Cloudflare tunnel to RunPod"}
                </li>
                <li>the worker token, as a setting of the pod: it only opens the worker routes of this backend, and only while the tunnel is up</li>
              </ul>
              <p className="mt-2">Keep this Mac on while it runs. You can stop it from the Training page at any time.</p>
            </div>
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2">
          <button className="btn" onClick={onClose} disabled={starting}>
            Not now
          </button>
          <button
            className="btn border-blue-600 bg-blue-600 text-white hover:bg-blue-700 dark:hover:bg-blue-700"
            disabled={!plan || checking || starting}
            onClick={start}
            data-testid="runpod-start"
          >
            {starting ? "Starting…" : plan ? `Start on RunPod (up to ${money(plan.worst_case)})` : "Start on RunPod"}
          </button>
        </div>
      </div>
    </div>
  );
}
