"use client";

import AiLabelling from "@/components/AiLabelling";
import { SectionHeader } from "@/components/Drawer";
import { useProject } from "@/lib/projectContext";

export default function AiPage() {
  const { projectId, project, imagesChanged } = useProject();
  return (
    <div>
      <SectionHeader title="AI labelling">Choose what the AI does, which models it uses and which images it works on. It runs on this Mac.</SectionHeader>
      <AiLabelling projectId={projectId} classes={project.classes} onImagesChanged={imagesChanged} />
    </div>
  );
}
