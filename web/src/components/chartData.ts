/** The data side of the charts: plain functions, tested without rendering. */

export interface ChartSeries {
  /** Unique key; also the property name in each row. */
  key: string;
  label: string;
  dates: readonly string[];
  values: readonly number[];
}

/** A training period to shade: [from, to), with `to` the first out-of-sample day. */
export interface Shade {
  from: string;
  to: string;
}

type Row = Record<string, string | number>;

/** One row per date, with each series' value under its key (dates missing from a series are left out). */
export function mergeByDate(series: readonly ChartSeries[]): Row[] {
  const rows = new Map<string, Row>();
  for (const s of series) {
    s.dates.forEach((date, i) => {
      const value = s.values[i];
      if (value === undefined) return;
      const row = rows.get(date) ?? { date };
      row[s.key] = value;
      rows.set(date, row);
    });
  }
  return [...rows.values()].sort((a, b) => String(a["date"]).localeCompare(String(b["date"])));
}
