/** The words and rules behind the metrics display (no components, so they can be shared and tested). */

export const FULL_PERIOD_LABEL = "full period: training months held flat and included";

export type Metrics = Record<string, number | null>;

/**
 * The interval qbt.stats.stationary_bootstrap_sharpe reports: the 5th and
 * 95th percentiles of the bootstrapped Sharpe ratios.
 */
export const SHARPE_CI_LEVEL = "90%";

/** Why a Sharpe interval is missing, or null when it is there. */
export function missingCiReason(metrics: Metrics, nBoot: number): string | null {
  if (metrics["sharpe_ci_low"] != null && metrics["sharpe_ci_high"] != null) return null;
  if (nBoot === 0) return "no interval: run with 0 bootstrap replicates";
  return "no interval: not finite (e.g. no variance in returns)";
}
