"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type Project } from "@/lib/api";

export default function Home() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.listProjects().then(setProjects).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api.createProject(name.trim());
      setName("");
      load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Projects</h1>
      <form onSubmit={create} className="flex gap-2">
        <input className="input w-64" placeholder="New project name" value={name} onChange={(e) => setName(e.target.value)} required />
        <button className="btn" disabled={!name.trim()}>
          Create
        </button>
      </form>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {projects === null && !error && <p className="text-sm text-neutral-500">Loading…</p>}
      {projects?.length === 0 && <p className="text-sm text-neutral-500">No projects yet. New projects start with multirotor, fixed-wing and delta-wing classes plus hard-negative categories.</p>}
      <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {projects?.map((p) => (
          <li key={p.id}>
            <Link href={`/projects/${p.id}`} className="block rounded border border-neutral-300 p-4 hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-900">
              <div className="font-medium">{p.name}</div>
              <div className="mt-1 text-sm text-neutral-500">
                {p.classes.filter((c) => !c.is_negative).map((c) => c.name).join(", ") || "no classes"}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
