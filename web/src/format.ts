/** Number formatting shared by the views. null (a NaN metric) shows as "n/a". */

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function signedPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "n/a";
  const text = pct(value, digits);
  return value > 0 ? `+${text}` : text;
}

export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return "n/a";
  return value.toFixed(digits);
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}

/** A field of a job's stored request, which the API types as `{ [key]: unknown }`. */
export function requestField(request: Record<string, unknown>, name: string): string {
  const value = request[name];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}
