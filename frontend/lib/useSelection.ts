import { useCallback, useEffect, useState } from "react";

/** A selection of the images on screen, for bulk actions. It belongs to the list it was made on (`resetKey` changing starts a new one) and only
 *  ever covers images that are still shown: a reload can drop images from the list, and a bulk action must not reach them. */
export function useSelection<T extends { id: number }>(items: T[], resetKey: unknown, max = 500) {
  const [selected, setSelected] = useState<Set<number>>(new Set());
  useEffect(() => setSelected(new Set()), [resetKey]);
  useEffect(() => {
    setSelected((prev) => {
      const onScreen = new Set(items.map((i) => i.id));
      const kept = [...prev].filter((id) => onScreen.has(id));
      return kept.length === prev.size ? prev : new Set(kept);
    });
  }, [items]);
  const toggle = useCallback(
    (id: number) =>
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else if (next.size < max) next.add(id);
        return next;
      }),
    [max],
  );
  const selectAll = useCallback(() => setSelected(new Set(items.slice(0, max).map((i) => i.id))), [items, max]);
  const clear = useCallback(() => setSelected(new Set()), []);
  return { selected, toggle, selectAll, clear, max };
}
