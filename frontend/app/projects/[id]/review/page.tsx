import { Suspense } from "react";
import ReviewScreen from "@/components/ReviewScreen";

export default function ReviewPage() {
  // useSearchParams in ReviewScreen needs a Suspense boundary for production builds.
  return (
    <Suspense fallback={<p className="text-neutral-500">Loading…</p>}>
      <ReviewScreen />
    </Suspense>
  );
}
