"use client";

import { SectionHeader } from "@/components/Drawer";
import ClassForm from "@/components/ClassForm";
import PlacesTable from "@/components/PlacesTable";
import StatsTable from "@/components/StatsTable";
import { useProject } from "@/lib/projectContext";

export default function OverviewPage() {
  const { projectId, stats, refreshKey, classesChanged } = useProject();
  return (
    <div className="space-y-6">
      <SectionHeader title="Overview">Where your images are, how far along they are, and how many images each class has.</SectionHeader>
      <section className="space-y-2">
        <h2 className="font-medium">Where the images are</h2>
        <PlacesTable projectId={projectId} refreshKey={refreshKey} />
      </section>
      <section className="space-y-2">
        <h2 className="font-medium">Classes</h2>
        <StatsTable projectId={projectId} rows={stats} onChanged={classesChanged} />
        <ClassForm projectId={projectId} onAdded={classesChanged} />
      </section>
    </div>
  );
}
