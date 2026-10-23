"use client";

import DatasetsPanel from "@/components/DatasetsPanel";
import { SectionHeader } from "@/components/Drawer";
import { useProject } from "@/lib/projectContext";

export default function DatasetsPage() {
  const { projectId, refreshKey } = useProject();
  return (
    <div>
      <SectionHeader title="Datasets">Frozen versions of your images and boxes, the data a training run uses.</SectionHeader>
      <DatasetsPanel projectId={projectId} refreshKey={refreshKey} />
    </div>
  );
}
