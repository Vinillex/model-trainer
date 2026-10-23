"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useProject } from "@/lib/projectContext";

// One entry per section; `href` is relative to the project ("" = the overview).
const ITEMS: { href: string; label: string; hint: string; icon: string }[] = [
  { href: "", label: "Overview", hint: "Counts and classes", icon: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" },
  { href: "/images", label: "Images", hint: "Browse, filter, review", icon: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4" },
  { href: "/add", label: "Add data", hint: "Upload or scrape", icon: "M12 5v14M5 12h14" },
  { href: "/ai", label: "AI labelling", hint: "Check or auto-label", icon: "M12 3l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" },
  { href: "/datasets", label: "Datasets", hint: "Versions to train on", icon: "M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7" },
  { href: "/training", label: "Training & export", hint: "Runs and the camera model", icon: "M13 3L4 14h7l-1 7 9-11h-7z" },
];

const COLLAPSE_KEY = "drawer-collapsed";
const POLL_MS = 10_000;

/** What the drawer shows beside a section: how many images need a look, whether an AI run or a training run is going. Decoration: failures are ignored. */
interface Badges {
  needsLook: number;
  aiRunning: boolean;
  training: boolean;
}
const NO_BADGES: Badges = { needsLook: 0, aiRunning: false, training: false };

function useBadges(projectId: number, refreshKey: number): Badges {
  const [badges, setBadges] = useState<Badges>(NO_BADGES);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        // the "needs a look" count is the Overview table's own number (one rule, one source)
        const [places, jobs, runs] = await Promise.all([api.folders(projectId), api.listJobs(projectId, "ai_check"), api.listTrainingRuns(projectId)]);
        if (alive) {
          setBadges({
            needsLook: places.reduce((n, p) => n + p.ai_flagged, 0),
            aiRunning: jobs.some((j) => j.status === "queued" || j.status === "running"),
            training: runs.some((r) => !["done", "failed", "cancelled"].includes(r.stage)),
          });
        }
      } catch {
        // keep what is shown
      }
    };
    load();
    const t = setInterval(() => document.visibilityState === "visible" && load(), POLL_MS); // not while the tab is hidden
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [projectId, refreshKey]);
  return badges;
}

/** The little marker beside a section: a count, or a pulsing dot for something running. */
function Badge({ href, badges, collapsed }: { href: string; badges: Badges; collapsed: boolean }) {
  const count = href === "/images" ? badges.needsLook : 0;
  const running = (href === "/ai" && badges.aiRunning) || (href === "/training" && badges.training);
  if (!count && !running) return null;
  const label = count ? `${count.toLocaleString()} images need a look` : href === "/ai" ? "an AI run is going" : "a training run is going";
  if (collapsed) return <span className={`absolute right-1 top-1 h-2 w-2 rounded-full ${count ? "bg-amber-500" : "animate-pulse bg-blue-400"}`} title={label} />;
  return count ? (
    <span className="ml-auto rounded-full bg-amber-500/20 px-2 text-xs font-medium text-amber-500" title={label} data-testid="badge-needs-look">
      {count.toLocaleString()}
    </span>
  ) : (
    <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-blue-400" title={label} data-testid={`badge-running${href}`} />
  );
}

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

function Nav({ base, collapsed, badges, onNavigate }: { base: string; collapsed: boolean; badges: Badges; onNavigate?: () => void }) {
  const path = usePathname();
  return (
    <nav className="space-y-1" aria-label="Sections">
      {ITEMS.map((item) => {
        const href = base + item.href;
        const active = item.href === "" ? path === base : path.startsWith(href);
        return (
          <Link
            key={item.href}
            href={href}
            onClick={onNavigate}
            title={collapsed ? item.label : item.hint}
            aria-current={active ? "page" : undefined}
            className={`relative flex items-center gap-3 rounded px-3 py-2 text-sm transition-colors ${
              active ? "bg-blue-600 text-white" : "text-neutral-700 hover:bg-neutral-200 dark:text-neutral-300 dark:hover:bg-neutral-800"
            } ${collapsed ? "justify-center px-2" : ""}`}
          >
            <Icon d={item.icon} />
            {!collapsed && (
              <span className="min-w-0">
                <span className="block truncate">{item.label}</span>
              </span>
            )}
            <Badge href={item.href} badges={badges} collapsed={collapsed} />
          </Link>
        );
      })}
    </nav>
  );
}

/** The project's left drawer (a slide-over on phones) and the page area beside it. */
export default function ProjectShell({ children }: { children: React.ReactNode }) {
  const { project, projectId, refreshKey } = useProject();
  const badges = useBadges(projectId, refreshKey);
  const base = `/projects/${projectId}`;
  const [collapsed, setCollapsed] = useState(false);
  const [open, setOpen] = useState(false); // phones only
  const path = usePathname();

  useEffect(() => {
    try {
      setCollapsed(localStorage.getItem(COLLAPSE_KEY) === "1");
    } catch {
      // storage blocked: start expanded
    }
  }, []);
  useEffect(() => setOpen(false), [path]);
  const toggle = () =>
    setCollapsed((c) => {
      try {
        localStorage.setItem(COLLAPSE_KEY, c ? "0" : "1");
      } catch {
        // the choice just isn't remembered
      }
      return !c;
    });

  return (
    <div className="flex gap-6">
      <aside className={`hidden shrink-0 md:block ${collapsed ? "w-14" : "w-56"}`}>
        <div className="sticky top-4 space-y-4">
          {!collapsed && (
            <div className="px-1">
              <Link href="/" className="text-xs text-neutral-500 hover:underline">
                ← Projects
              </Link>
              <p className="mt-1 truncate text-lg font-semibold" title={project.name}>
                {project.name}
              </p>
            </div>
          )}
          <Nav base={base} collapsed={collapsed} badges={badges} />
          <button onClick={toggle} className="w-full rounded px-3 py-1.5 text-left text-xs text-neutral-500 hover:bg-neutral-200 dark:hover:bg-neutral-800" aria-label={collapsed ? "Expand menu" : "Collapse menu"}>
            {collapsed ? "»" : "« Collapse"}
          </button>
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        <div className="mb-4 flex items-center gap-3 md:hidden">
          <button className="btn" onClick={() => setOpen(true)} aria-label="Open menu">
            ☰ Menu
          </button>
          <span className="truncate font-semibold">{project.name}</span>
        </div>
        {children}
      </div>

      {open && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-label="Menu">
          <div className="absolute inset-0 bg-black/50" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-64 space-y-4 bg-white p-4 shadow-xl dark:bg-neutral-950">
            <div className="flex items-center justify-between">
              <Link href="/" className="text-xs text-neutral-500 hover:underline">
                ← Projects
              </Link>
              <button className="text-neutral-500" onClick={() => setOpen(false)} aria-label="Close menu">
                ✕
              </button>
            </div>
            <p className="truncate text-lg font-semibold">{project.name}</p>
            <Nav base={base} collapsed={false} badges={badges} onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}
    </div>
  );
}

/** Title and one-line explanation at the top of each section. */
export function SectionHeader({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <header className="mb-5">
      <h1 className="text-2xl font-semibold">{title}</h1>
      {children && <p className="mt-1 text-sm text-neutral-500">{children}</p>}
    </header>
  );
}
