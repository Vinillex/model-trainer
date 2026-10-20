"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import Gallery from "@/components/Gallery";
import DatasetsPanel from "@/components/DatasetsPanel";
import ClassForm from "@/components/ClassForm";
import ScrapePanel from "@/components/ScrapePanel";
import StatsTable from "@/components/StatsTable";
import TrainingPanel from "@/components/TrainingPanel";
import UploadBox from "@/components/UploadBox";
import { api, type Project, type StatRow } from "@/lib/api";

export default function ProjectPage() {
  const projectId = Number(useParams<{ id: string }>().id);
  const [project, setProject] = useState<Project | null>(null);
  const [stats, setStats] = useState<StatRow[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const bumpRefresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  const loadProject = useCallback(() => {
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
  }, [projectId]);
  const loadStats = useCallback(() => {
    api.stats(projectId).then(setStats).catch((e) => setError(e.message));
  }, [projectId]);

  useEffect(loadProject, [loadProject]);
  useEffect(loadStats, [loadStats, refreshKey]);

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
        <StatsTable projectId={projectId} rows={stats} onChanged={() => { loadProject(); loadStats(); }} />
        <ClassForm projectId={projectId} onAdded={() => { loadProject(); loadStats(); }} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Upload images</h2>
        <UploadBox projectId={projectId} classes={project.classes} onDone={() => setRefreshKey((k) => k + 1)} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Scrape images</h2>
        <ScrapePanel projectId={projectId} classes={project.classes} onImagesChanged={bumpRefresh} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Dataset versions</h2>
        <DatasetsPanel projectId={projectId} refreshKey={refreshKey} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Train &amp; export</h2>
        <TrainingPanel projectId={projectId} refreshKey={refreshKey} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Gallery</h2>
        <Gallery projectId={projectId} classes={project.classes} refreshKey={refreshKey} onChanged={() => setRefreshKey((k) => k + 1)} />
      </section>
    </div>
  );
}
