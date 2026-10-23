/** Turning what the person picked or dropped (loose files, one or more folders) into a flat list of files with the folder each sat in. */

export interface Picked {
  file: File;
  /** Relative to where the folder was picked or dropped, e.g. "flight-1/day2/img001.jpg"; just the name for a loose file. */
  path: string;
  /** The folder that was picked or dropped (its top level, whatever the depth), null for a loose file. */
  folder: string | null;
}

export const isImageName = (name: string) => /\.(jpe?g|png|webp)$/i.test(name);

/** From a file input (with or without `webkitdirectory`): a folder pick gives each file its webkitRelativePath. */
export function fromInput(files: File[]): Picked[] {
  return files.map((file) => {
    const rel = file.webkitRelativePath;
    return rel ? { file, path: rel, folder: rel.split("/")[0] } : { file, path: file.name, folder: null };
  });
}

/** The dropped entries. Call this synchronously inside the drop handler: the browser empties the DataTransfer once the handler returns. */
export function droppedEntries(dt: DataTransfer): FileSystemEntry[] | null {
  const items = Array.from(dt.items ?? []);
  const entries = items.map((i) => i.webkitGetAsEntry?.() ?? null);
  return entries.length > 0 && entries.every((e) => e !== null) ? (entries as FileSystemEntry[]) : null; // null: this browser can't read folders, use dt.files
}

const fileOf = (e: FileSystemFileEntry) => new Promise<File>((resolve, reject) => e.file(resolve, reject));
const nextBatch = (r: FileSystemDirectoryReader) => new Promise<FileSystemEntry[]>((resolve, reject) => r.readEntries(resolve, reject));

async function walk(entry: FileSystemEntry, folder: string | null, out: Picked[]): Promise<void> {
  if (entry.isFile) {
    out.push({ file: await fileOf(entry as FileSystemFileEntry), path: entry.fullPath.replace(/^\//, ""), folder });
    return;
  }
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  // readEntries hands back at most ~100 entries a call: keep asking until it returns an empty batch.
  for (let batch = await nextBatch(reader); batch.length > 0; batch = await nextBatch(reader)) {
    await Promise.all(batch.map((child) => walk(child, folder, out)));
  }
}

/** Every file under the dropped entries. A dropped folder's files are filed under that folder's name. */
export async function fromEntries(entries: FileSystemEntry[]): Promise<Picked[]> {
  const out: Picked[] = [];
  await Promise.all(entries.map((e) => walk(e, e.isDirectory ? e.name : null, out)));
  return out.sort((a, b) => a.path.localeCompare(b.path)); // the order the files were found in is not stable
}

/** What gets uploaded: the images, grouped by folder (null = loose files), and how many other files were left out. */
export function plan(picked: Picked[]): { groups: Map<string | null, Picked[]>; skipped: number } {
  const groups = new Map<string | null, Picked[]>();
  let skipped = 0;
  for (const p of picked) {
    if (!isImageName(p.file.name)) {
      skipped++;
      continue;
    }
    groups.set(p.folder, [...(groups.get(p.folder) ?? []), p]);
  }
  return { groups, skipped };
}
