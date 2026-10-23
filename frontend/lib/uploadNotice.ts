/** A one-line result of an upload, kept for the Images page that opens right after it. Per tab; never throws. */
const key = (projectId: number) => `upload-notice:${projectId}`;

export function setUploadNotice(projectId: number, text: string) {
  try {
    sessionStorage.setItem(key(projectId), text);
  } catch {
    // the notice is just not shown
  }
}

/** Reads the notice once: asking again gives null. */
export function takeUploadNotice(projectId: number): string | null {
  try {
    const text = sessionStorage.getItem(key(projectId));
    sessionStorage.removeItem(key(projectId));
    return text;
  } catch {
    return null;
  }
}
