import { SOURCES, type StatRow } from "@/lib/api";

export default function StatsTable({ rows }: { rows: StatRow[] }) {
  const total = (r: StatRow) => r.uploaded + r.scraped + r.generated;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-neutral-300 text-left dark:border-neutral-700">
          <th className="py-1.5 pr-4">Class</th>
          {SOURCES.map((s) => (
            <th key={s} className="px-2 text-right capitalize">
              {s}
            </th>
          ))}
          <th className="pl-2 text-right">Total</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.class_id} className="border-b border-neutral-200 dark:border-neutral-800">
            <td className="py-1.5 pr-4">
              {r.class_name}
              {r.is_negative && <span className="ml-2 rounded bg-neutral-200 px-1.5 py-0.5 text-xs dark:bg-neutral-800">negative</span>}
            </td>
            {SOURCES.map((s) => (
              <td key={s} className="px-2 text-right tabular-nums">
                {r[s]}
              </td>
            ))}
            <td className="pl-2 text-right tabular-nums font-medium">{total(r)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
