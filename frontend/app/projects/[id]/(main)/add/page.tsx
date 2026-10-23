"use client";

import { useState } from "react";
import { SectionHeader } from "@/components/Drawer";
import ScrapePanel from "@/components/ScrapePanel";
import UploadBox from "@/components/UploadBox";
import { useProject } from "@/lib/projectContext";

const TABS = [
  { id: "upload", label: "Upload" },
  { id: "scrape", label: "Scrape" },
] as const;

export default function AddDataPage() {
  const { projectId, project, imagesChanged } = useProject();
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("upload");
  return (
    <div className="space-y-4">
      <SectionHeader title="Add data">Bring in new images. They show up in Images.</SectionHeader>
      <div className="flex gap-1 border-b border-neutral-300 dark:border-neutral-700" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px border-b-2 px-4 py-2 text-sm ${tab === t.id ? "border-blue-600 font-medium" : "border-transparent text-neutral-500 hover:text-current"}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tab === "upload" ? (
        <UploadBox projectId={projectId} classes={project.classes} onDone={imagesChanged} />
      ) : (
        <ScrapePanel projectId={projectId} classes={project.classes} onImagesChanged={imagesChanged} />
      )}
    </div>
  );
}
