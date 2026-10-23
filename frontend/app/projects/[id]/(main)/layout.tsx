"use client";

import { useParams } from "next/navigation";
import ProjectShell from "@/components/Drawer";
import { ProjectProvider } from "@/lib/projectContext";

/** Every section of a project except the full-screen review: a left drawer and the section beside it. */
export default function ProjectLayout({ children }: { children: React.ReactNode }) {
  const projectId = Number(useParams<{ id: string }>().id);
  return (
    <ProjectProvider projectId={projectId}>
      <ProjectShell>{children}</ProjectShell>
    </ProjectProvider>
  );
}
