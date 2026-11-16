/** The metrics display shared by the Results and Compare views. */
import { num, pct } from "../../format";
import { FULL_PERIOD_LABEL, SHARPE_CI_LEVEL, missingCiReason, type Metrics } from "./metricsText";

export function SharpeWithCi({ metrics, nBoot }: { metrics: Metrics; nBoot: number }) {
  const reason = missingCiReason(metrics, nBoot);
  return (
    <span className="sharpe">
      <span className="headline-value">{num(metrics["sharpe_ratio"])}</span>{" "}
      <span className="ci">
        {reason ??
          `${SHARPE_CI_LEVEL} CI ${num(metrics["sharpe_ci_low"])} to ${num(metrics["sharpe_ci_high"])}`}
      </span>
    </span>
  );
}

const ROWS: { name: string; label: string; format: (v: number | null | undefined) => string }[] = [
  { name: "total_return", label: "Total return", format: (v) => pct(v) },
  { name: "cagr", label: "CAGR", format: (v) => pct(v) },
  { name: "annualized_volatility", label: "Volatility (annualized)", format: (v) => pct(v) },
  { name: "sortino_ratio", label: "Sortino ratio", format: (v) => num(v) },
  { name: "max_drawdown", label: "Max drawdown", format: (v) => pct(v) },
  { name: "exposure", label: "Exposure", format: (v) => pct(v) },
  { name: "turnover", label: "Turnover", format: (v) => num(v) },
  { name: "num_trades", label: "Trades", format: (v) => num(v, 0) },
  { name: "win_rate", label: "Win rate", format: (v) => pct(v) },
];

export interface MetricsColumn {
  key: string;
  label: string;
  metrics: Metrics;
  nBoot: number;
  fullPeriod: boolean;
}

export function MetricsTable({ columns }: { columns: readonly MetricsColumn[] }) {
  return (
    <table className="metrics">
      <caption>Metrics</caption>
      <thead>
        <tr>
          <th scope="col">Metric</th>
          {columns.map((c) => (
            <th scope="col" key={c.key}>
              {c.label}
              {c.fullPeriod && <span className="badge warn">{FULL_PERIOD_LABEL}</span>}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">Sharpe ratio</th>
          {columns.map((c) => (
            <td key={c.key}>
              <SharpeWithCi metrics={c.metrics} nBoot={c.nBoot} />
            </td>
          ))}
        </tr>
        {ROWS.map((row) => (
          <tr key={row.name}>
            <th scope="row">{row.label}</th>
            {columns.map((c) => (
              <td key={c.key}>{row.format(c.metrics[row.name])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
