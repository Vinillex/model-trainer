"use client";

import { useEffect, useState } from "react";
import { SectionHeader } from "@/components/Drawer";
import Gallery from "@/components/Gallery";
import { useProject } from "@/lib/projectContext";
import { takeUploadNotice } from "@/lib/uploadNotice";

export default function ImagesPage() {
  const { projectId, project, refreshKey, imagesChanged } = useProject();
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    const text = takeUploadNotice(projectId); // set by an upload that opened a folder; reading clears it
    if (text) setNotice(text);
  }, [projectId]);
  return (
    <div>
      <SectionHeader title="Images">Filter the images, click one to review it. The review screen opens the same list you see here.</SectionHeader>
      {notice && (
        <p className="mb-3 flex items-center justify-between gap-3 rounded border border-green-600/50 bg-green-600/10 px-3 py-2 text-sm" data-testid="upload-notice">
          <span>{notice}</span>
          <button className="text-neutral-500 hover:text-current" onClick={() => setNotice(null)} aria-label="Dismiss">
            ✕
          </button>
        </p>
      )}
      <Gallery projectId={projectId} classes={project.classes} refreshKey={refreshKey} onChanged={imagesChanged} />
    </div>
  );
}
