"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, type Project, type StatRow } from "@/lib/api";

interface ProjectState {
  projectId: number;
  project: Project;
  stats: StatRow[];
  /** Bumped after anything that changes images, so every section reloads what it shows. */
  refreshKey: number;
  /** Images changed (upload, scrape, AI, delete): reload stats and bump refreshKey. */
  imagesChanged: () => void;
  /** Classes changed: reload the project (its classes) and the stats. */
  classesChanged: () => void;
}

const ProjectContext = createContext<ProjectState | null>(null);

/** Loads the project once for all its sections (the drawer layout provides it). */
export function ProjectProvider({ projectId, children }: { projectId: number; children: React.ReactNode }) {
  const [project, setProject] = useState<Project | null>(null);
  const [stats, setStats] = useState<StatRow[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const loadProject = useCallback(() => {
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
  }, [projectId]);
  const loadStats = useCallback(() => {
    api.stats(projectId).then(setStats).catch((e) => setError(e.message));
  }, [projectId]);
  useEffect(loadProject, [loadProject]);
  useEffect(loadStats, [loadStats, refreshKey]);

  const imagesChanged = useCallback(() => setRefreshKey((k) => k + 1), []);
  const classesChanged = useCallback(() => {
    loadProject();
    loadStats();
  }, [loadProject, loadStats]);

  if (error && !project) return <p className="text-red-600">{error}</p>;
  if (!project) return <p className="text-neutral-500">Loading…</p>;
  return (
    <ProjectContext.Provider value={{ projectId, project, stats, refreshKey, imagesChanged, classesChanged }}>{children}</ProjectContext.Provider>
  );
}

export function useProject(): ProjectState {
  const p = useContext(ProjectContext);
  if (!p) throw new Error("useProject outside ProjectProvider");
  return p;
}
