import type { Box, Ghost } from "@/components/BoxEditor";
import type { AiBox, AiCheck } from "@/lib/api";

export type SuggestionKind = "adjust" | "suspect" | "missing";

/** One thing the AI check wants the person to look at. Nothing here is saved until they accept it and save the image. */
export interface Suggestion {
  key: string; // stable within a check: "l0" for the first label, "m1" for the second missing box
  kind: SuggestionKind;
  text: string;
  boxKey?: string; // the label's box in the editor ("a" + annotation id), when the suggestion is about a drawn label
  box?: AiBox; // what accepting does: replace that label's box with this one, or add it as a new box
  outline: AiBox; // what to draw: the suggested box, or the label itself when the AI only doubts it
  className?: string | null; // the class the detector gave a missing drone
}

export const KIND_COLOR: Record<SuggestionKind, string> = { adjust: "#f59e0b", suspect: "#ef4444", missing: "#06b6d4" };
const KIND_LABEL: Record<SuggestionKind, string> = { adjust: "AI: tighter box", suspect: "AI: doubts this box", missing: "AI: no label here" };

/** The suggestions of a check, or none when it is out of date (the boxes changed since, so they may no longer fit). */
export function buildSuggestions(check: AiCheck | null): Suggestion[] {
  if (!check || check.stale) return [];
  const out: Suggestion[] = [];
  check.result.labels.forEach((l, i) => {
    if (l.verdict !== "adjust" && l.verdict !== "suspect") return;
    out.push({
      key: `l${i}`,
      kind: l.verdict,
      text: l.reason,
      boxKey: l.annotation_id !== null ? `a${l.annotation_id}` : undefined,
      box: l.suggested ?? undefined,
      outline: l.suggested ?? l.box,
    });
  });
  check.result.missing.forEach((m, i) => out.push({ key: `m${i}`, kind: "missing", text: m.reason, box: m.box, outline: m.box, className: m.class_name }));
  return out;
}

export function ghostFor(s: Suggestion): Ghost {
  const [x1, y1, x2, y2] = s.outline;
  return { key: s.key, x1, y1, x2, y2, color: KIND_COLOR[s.kind], label: KIND_LABEL[s.kind] };
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));

/** Accept a suggestion into the editor's boxes: a drawn label's box is replaced (class kept), anything else is added in `classId`.
 *  Returns null when there is no box to accept (the AI only doubted the label). */
export function applySuggestion(boxes: Box[], s: Suggestion, classId: number, newKey: () => string): { boxes: Box[]; selected: string } | null {
  if (!s.box) return null;
  const [x1, y1, x2, y2] = s.box.map(clamp);
  if (x2 - x1 <= 0 || y2 - y1 <= 0) return null;
  if (s.boxKey && boxes.some((b) => b.key === s.boxKey)) {
    return { boxes: boxes.map((b) => (b.key === s.boxKey ? { ...b, x1, y1, x2, y2 } : b)), selected: s.boxKey };
  }
  const key = newKey();
  return { boxes: [...boxes, { key, classId, x1, y1, x2, y2 }], selected: key };
}
