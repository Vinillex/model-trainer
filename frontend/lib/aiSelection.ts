/** The images picked on the Images page for "Run AI on these", handed to the AI page. Per tab; never throws. */
const key = (projectId: number) => `ai-selection:${projectId}`;

export function setAiSelection(projectId: number, ids: number[]) {
  try {
    sessionStorage.setItem(key(projectId), JSON.stringify(ids));
  } catch {
    // the AI page then simply starts without a selection
  }
}

/** Reads the selection once: asking again gives null. */
export function takeAiSelection(projectId: number): number[] | null {
  try {
    const raw = sessionStorage.getItem(key(projectId));
    sessionStorage.removeItem(key(projectId));
    const ids = raw ? JSON.parse(raw) : null;
    return Array.isArray(ids) && ids.every((i) => Number.isInteger(i)) && ids.length > 0 ? ids : null;
  } catch {
    return null;
  }
}
