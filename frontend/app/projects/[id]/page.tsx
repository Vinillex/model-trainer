"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import Gallery from "@/components/Gallery";
import StatsTable from "@/components/StatsTable";
import UploadBox from "@/components/UploadBox";
import { api, type Project, type StatRow } from "@/lib/api";

export default function ProjectPage() {
  const projectId = Number(useParams<{ id: string }>().id);
  const [project, setProject] = useState<Project | null>(null);
  const [stats, setStats] = useState<StatRow[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [newClass, setNewClass] = useState("");
  const [newNegative, setNewNegative] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProject = useCallback(() => {
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
  }, [projectId]);
  const loadStats = useCallback(() => {
    api.stats(projectId).then(setStats).catch((e) => setError(e.message));
  }, [projectId]);

  useEffect(loadProject, [loadProject]);
  useEffect(loadStats, [loadStats, refreshKey]);

  async function addClass(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api.createClass(projectId, newClass.trim(), newNegative);
      setNewClass("");
      setNewNegative(false);
      loadProject();
      loadStats();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  if (error && !project) return <p className="text-red-600">{error}</p>;
  if (!project) return <p className="text-neutral-500">Loading…</p>;

  return (
    <div className="space-y-8">
      <div>
        <Link href="/" className="text-sm text-neutral-500 hover:underline">
          ← Projects
        </Link>
        <h1 className="mt-1 text-2xl font-semibold">{project.name}</h1>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Dataset</h2>
        <StatsTable rows={stats} />
        <form onSubmit={addClass} className="flex flex-wrap items-center gap-2">
          <input
            className="input w-48"
            placeholder="new-class-name"
            pattern="^[a-z0-9][a-z0-9_\-]*$"
            title="lowercase letters, digits, - and _"
            value={newClass}
            onChange={(e) => setNewClass(e.target.value)}
            required
          />
          <label className="flex items-center gap-1 text-sm">
            <input type="checkbox" checked={newNegative} onChange={(e) => setNewNegative(e.target.checked)} /> hard negative
          </label>
          <button className="btn">Add class</button>
        </form>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Upload images</h2>
        <UploadBox projectId={projectId} classes={project.classes} onDone={() => setRefreshKey((k) => k + 1)} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Gallery</h2>
        <Gallery projectId={projectId} classes={project.classes} refreshKey={refreshKey} onChanged={() => setRefreshKey((k) => k + 1)} />
      </section>
    </div>
  );
}
