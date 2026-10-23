"use client";

import { SectionHeader } from "@/components/Drawer";
import TrainingPanel from "@/components/TrainingPanel";
import { useProject } from "@/lib/projectContext";

export default function TrainingPage() {
  const { projectId, refreshKey } = useProject();
  return (
    <div>
      <SectionHeader title="Training & export">Train on a dataset version and export the model for the camera.</SectionHeader>
      <TrainingPanel projectId={projectId} refreshKey={refreshKey} />
    </div>
  );
}
