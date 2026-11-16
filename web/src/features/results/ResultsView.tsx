/**
 * One backtest's results: the equity curve and drawdown from
 * GET /backtests/{id}/series, the Sharpe ratio with its bootstrap interval,
 * the other metrics, and the strategy version it ran with.
 *
 * For a learning strategy (the series has `out_of_sample_from`), the
 * training period is shaded and the metrics are labelled as full-period:
 * the service computes them over the whole window, flat training months
 * included (README, Known limitations).
 */
import { POLL_INTERVAL_MS, useBacktestJob, useBacktestSeries } from "../../api/queries";
import { isTerminal, type BacktestJob, type BacktestSeries } from "../../api/types";
import { DrawdownChart, EquityChart } from "../../components/charts";
import { Empty, Loading, ProblemView } from "../../components/status";
import { num, pct, requestField } from "../../format";
import { MetricsTable, SharpeWithCi } from "./metrics";
import { FULL_PERIOD_LABEL } from "./metricsText";

export function ResultsView({
  id,
  pollIntervalMs = POLL_INTERVAL_MS,
}: {
  id: string;
  pollIntervalMs?: number;
}) {
  // useBacktestJob (useQuery, polling): the job's status and metrics. A job
  // opened while still running is polled until it is final.
  const job = useBacktestJob(id, pollIntervalMs);
  const succeeded = job.data?.status === "succeeded";
  // useBacktestSeries (useQuery): the daily series, requested only once
  // the job has succeeded (before that the API answers 409).
  const series = useBacktestSeries(id, succeeded);

  let content;
  if (job.isPending) {
    content = <Loading label="Loading the backtest…" />;
  } else if (job.isError) {
    content = (
      <ProblemView
        context="Could not load the backtest"
        problem={job.error}
        onRetry={() => void job.refetch()}
      />
    );
  } else if (!isTerminal(job.data.status)) {
    content = (
      <Loading
        label={`The backtest is ${job.data.status}; checking every ${String(pollIntervalMs / 1000)} s.`}
      />
    );
  } else if (job.data.status === "failed") {
    content = (
      <p className="field-error" role="alert">
        The backtest failed: {job.data.last_error ?? "no error was recorded."}
      </p>
    );
  } else if (series.isPending) {
    content = <Loading label="Loading the equity curve…" />;
  } else if (series.isError) {
    content = (
      <ProblemView
        context="Could not load the equity curve"
        problem={series.error}
        onRetry={() => void series.refetch()}
      />
    );
  } else if (series.data.dates.length === 0) {
    content = <Empty title="No trading days">The backtest has no days to show.</Empty>;
  } else {
    content = <Succeeded job={job.data} series={series.data} />;
  }

  return (
    <section aria-labelledby="results-heading">
      <h2 id="results-heading">Results</h2>
      {job.data && <JobHeader job={job.data} />}
      {content}
    </section>
  );
}

function JobHeader({ job }: { job: BacktestJob }) {
  const r = job.request;
  return (
    <p className="job-header">
      <strong>{requestField(r, "strategy")}</strong>{" "}
      <span className="badge">strategy version {job.strategy_version}</span> on{" "}
      {requestField(r, "ticker")}, {requestField(r, "start")} up to {requestField(r, "end")} · cost{" "}
      {requestField(r, "cost_bps")} bps · job <code>{job.id}</code>
    </p>
  );
}

function Succeeded({ job, series }: { job: BacktestJob; series: BacktestSeries }) {
  const metrics = job.metrics ?? {};
  const first = series.dates[0];
  const shade =
    series.out_of_sample_from && first ? { from: first, to: series.out_of_sample_from } : undefined;
  const nBoot = Number(requestField(job.request, "n_boot"));

  return (
    <>
      <div className="headline">
        <div className="headline-item">
          <span className="headline-label">Sharpe ratio</span>
          <SharpeWithCi metrics={metrics} nBoot={nBoot} />
        </div>
        <div className="headline-item">
          <span className="headline-label">Total return</span>
          <span className="headline-value">{pct(metrics["total_return"])}</span>
        </div>
        <div className="headline-item">
          <span className="headline-label">Max drawdown</span>
          <span className="headline-value">{pct(metrics["max_drawdown"])}</span>
        </div>
      </div>
      {series.out_of_sample_from && (
        <p className="notice" role="note">
          <strong>{FULL_PERIOD_LABEL}.</strong> This strategy learns from history: it holds no
          position before {series.out_of_sample_from}, and every metric on this page covers the
          whole window, that flat training period included. Out-of-sample-only results are in
          results/summary.md.
        </p>
      )}
      <EquityChart
        title="Equity curve"
        series={[{ key: "equity", label: "Equity", dates: series.dates, values: series.equity }]}
        shade={shade}
      />
      <DrawdownChart dates={series.dates} drawdown={series.drawdown} shade={shade} />
      <MetricsTable
        columns={[
          {
            key: job.id,
            label: "Value",
            metrics,
            nBoot,
            fullPeriod: series.out_of_sample_from !== null,
          },
        ]}
      />
      <p className="hint">
        {series.dates.length} trading days · final equity {num(series.equity.at(-1), 3)}
      </p>
    </>
  );
}
