/**
 * The backtests submitted from this browser, newest first, so the Compare
 * view has something to pick from: the API has no endpoint that lists jobs.
 *
 * Kept in localStorage, which can be unavailable (private windows, blocked
 * storage) or hold anything; every read and write tolerates that and the
 * app works without it.
 */
export interface RecentRun {
  id: string;
  label: string;
}

const KEY = "qbt.recentRuns";
export const MAX_RECENT = 20;

function isRecentRun(value: unknown): value is RecentRun {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v["id"] === "string" && typeof v["label"] === "string";
}

export function readRecentRuns(): RecentRun[] {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(isRecentRun).slice(0, MAX_RECENT) : [];
  } catch {
    return [];
  }
}

export function recordRecentRun(run: RecentRun): void {
  try {
    const runs = [run, ...readRecentRuns().filter((r) => r.id !== run.id)].slice(0, MAX_RECENT);
    window.localStorage.setItem(KEY, JSON.stringify(runs));
  } catch {
    // Storage unavailable: the run just isn't remembered.
  }
}
