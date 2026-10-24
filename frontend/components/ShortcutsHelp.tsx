"use client";

const GROUPS: { title: string; rows: [string, string][] }[] = [
  {
    title: "Deciding",
    rows: [
      ["Enter", "Accept the image and go to the next one (a hard-negative picture: keep it as is)"],
      ["X", "Reject the image (into the trash) and go to the next one"],
      ["← / →", "Previous / next image (unsaved edits are saved first)"],
    ],
  },
  {
    title: "Boxes",
    rows: [
      ["drag on the image", "Draw a new box"],
      ["click · drag · handles", "Select a box · move it · resize it"],
      ["1 – 9", "Pick the class for new boxes (or change the selected box)"],
      ["Delete", "Remove the selected box"],
      ["Esc", "Deselect"],
    ],
  },
  {
    title: "AI",
    rows: [
      ["A", "Check this image with the AI"],
      ["S", "Snap the selected box tight around the object in it"],
      ["Alt + click", "Click on a drone: the AI draws the box around it"],
    ],
  },
  {
    title: "View",
    rows: [
      ["scroll", "Zoom at the cursor"],
      ["+ / −", "Zoom in / out"],
      ["0", "Fit the image"],
      ["middle button + drag", "Move the zoomed image"],
    ],
  },
];

/** The review screen's keys, on request (?). While it is open the screen ignores every other key (see ReviewScreen). */
export default function ShortcutsHelp({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-label="Keyboard shortcuts" data-testid="shortcuts">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative max-h-full w-full max-w-2xl overflow-y-auto rounded-lg border border-neutral-300 bg-white p-5 shadow-xl dark:border-neutral-700 dark:bg-neutral-950">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Keyboard shortcuts</h2>
          <button className="btn" onClick={onClose} aria-label="Close shortcuts">
            Close (Esc)
          </button>
        </div>
        <div className="grid gap-5 sm:grid-cols-2">
          {GROUPS.map((g) => (
            <section key={g.title}>
              <h3 className="mb-1 text-sm font-medium text-neutral-500">{g.title}</h3>
              <dl className="space-y-1 text-sm">
                {g.rows.map(([key, what]) => (
                  <div key={key} className="flex gap-3">
                    <dt className="w-40 shrink-0">
                      <kbd className="rounded border border-neutral-400 px-1.5 py-0.5 text-xs dark:border-neutral-600">{key}</kbd>
                    </dt>
                    <dd className="text-neutral-600 dark:text-neutral-400">{what}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
        <p className="mt-4 text-xs text-neutral-500">Press ? any time to open this list.</p>
      </div>
    </div>
  );
}
