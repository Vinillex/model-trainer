"use client";

import { SectionHeader } from "@/components/Drawer";
import TrashPanel from "@/components/TrashPanel";
import { useProject } from "@/lib/projectContext";

export default function TrashPage() {
  const { projectId, project, refreshKey, imagesChanged } = useProject();
  return (
    <div>
      <SectionHeader title="Trash">
        Every image you rejected, from every dataset and folder. They are kept out of the Images page, the review queue and new dataset versions. Restore puts them back as
        &ldquo;not reviewed&rdquo;; Delete forever removes the files.
      </SectionHeader>
      <TrashPanel projectId={projectId} classes={project.classes} refreshKey={refreshKey} onChanged={imagesChanged} />
    </div>
  );
}
