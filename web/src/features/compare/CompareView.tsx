/**
 * Two or more completed backtests side by side: one metrics table, and
 * their equity curves overlaid on one chart.
 *
 * The selection lives in the URL (/compare?ids=a,b), so a comparison can
 * be bookmarked or shared. Backtests are picked from the ones submitted in
 * this browser, or added by job id.
 */
import { useId, useState, type SubmitEvent } from "react";

import { POLL_INTERVAL_MS, useBacktestJobs, useBacktestSeriesList } from "../../api/queries";
import type { BacktestJob, BacktestSeries } from "../../api/types";
import { EquityChart } from "../../components/charts";
import { Link } from "../../components/Link";
import { Empty, Loading, ProblemView } from "../../components/status";
import { requestField, shortId } from "../../format";
import { compareHref, navigate } from "../../router";
import { MAX_SERIES } from "../../theme";
import { FULL_PERIOD_LABEL } from "../results/metricsText";
import { MetricsTable } from "../results/metrics";
import { readRecentRuns } from "./recentRuns";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function label(job: BacktestJob): string {
  const r = job.request;
  return `${requestField(r, "strategy")} v${String(job.strategy_version)} · ${requestField(r, "ticker")} · ${shortId(job.id)}`;
}

export function CompareView({
  ids,
  pollIntervalMs = POLL_INTERVAL_MS,
}: {
  ids: readonly string[];
  pollIntervalMs?: number;
}) {
  const selected = ids.slice(0, MAX_SERIES);
  // useBacktestJobs (useQueries): every selected job, one query each; a job
  // that is still running is polled until it is final.
  const jobs = useBacktestJobs(selected, pollIntervalMs);
  const succeededIds = selected.filter((_, i) => jobs[i]?.data?.status === "succeeded");
  // useBacktestSeriesList (useQueries): the series of each succeeded job.
  const seriesList = useBacktestSeriesList(succeededIds);

  const setIds = (next: readonly string[]) => {
    navigate(compareHref(next), { replace: true });
  };

  const ready: { job: BacktestJob; series: BacktestSeries }[] = [];
  succeededIds.forEach((id, i) => {
    const job = jobs[selected.indexOf(id)]?.data;
    const series = seriesList[i]?.data;
    if (job && series) ready.push({ job, series });
  });
  const loading =
    jobs.some((q) => q.isPending) ||
    seriesList.some((q) => q.isPending && q.fetchStatus !== "idle");

  let comparison;
  if (selected.length < 2) {
    comparison = (
      <Empty title="Pick at least two completed backtests">
        Choose them from your recent runs below, or add one by its job id.
      </Empty>
    );
  } else if (loading) {
    comparison = <Loading label="Loading the backtests…" />;
  } else if (ready.length < 2) {
    comparison = (
      <Empty title="Not enough completed backtests">
        {ready.length === 0 ? "None" : "Only one"} of the selected backtests has succeeded and
        loaded. Only succeeded backtests can be compared.
      </Empty>
    );
  } else {
    comparison = (
      <>
        <EquityChart
          title="Equity curves"
          series={ready.map(({ job, series }) => ({
            key: job.id,
            label: label(job),
            dates: series.dates,
            values: series.equity,
          }))}
        />
        {ready.some(({ series }) => series.out_of_sample_from) && (
          <p className="notice" role="note">
            Columns marked <strong>{FULL_PERIOD_LABEL}</strong> are learning strategies: they hold
            no position during their training period, and their metrics include it.
          </p>
        )}
        <MetricsTable
          columns={ready.map(({ job, series }) => ({
            key: job.id,
            label: label(job),
            metrics: job.metrics ?? {},
            nBoot: Number(requestField(job.request, "n_boot")),
            fullPeriod: series.out_of_sample_from !== null,
          }))}
        />
      </>
    );
  }

  return (
    <section aria-labelledby="compare-heading">
      <h2 id="compare-heading">Compare backtests</h2>
      {ids.length > MAX_SERIES && (
        <p className="hint">Showing the first {MAX_SERIES}: one chart can tell that many apart.</p>
      )}
      <Selection
        ids={selected}
        jobs={jobs.map((q) => ({ data: q.data, error: q.error }))}
        onRemove={(id) => {
          setIds(selected.filter((x) => x !== id));
        }}
      />
      {comparison}
      <Picker
        ids={selected}
        full={selected.length >= MAX_SERIES}
        onToggle={(id) => {
          setIds(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
        }}
      />
    </section>
  );
}

function Selection({
  ids,
  jobs,
  onRemove,
}: {
  ids: readonly string[];
  jobs: readonly {
    data: BacktestJob | undefined;
    error: ReturnType<typeof useBacktestJobs>[number]["error"];
  }[];
  onRemove: (id: string) => void;
}) {
  if (ids.length === 0) return null;
  return (
    <ul className="selection" aria-label="Selected backtests">
      {ids.map((id, i) => {
        const q = jobs[i];
        return (
          <li key={id}>
            <Link to={`/backtests/${id}`}>{q?.data ? label(q.data) : shortId(id)}</Link>{" "}
            {q?.data && q.data.status !== "succeeded" && (
              <span className="badge">{q.data.status}: not comparable</span>
            )}
            {q?.error && <ProblemView context={`Backtest ${shortId(id)}`} problem={q.error} />}
            <button
              type="button"
              className="link-button"
              onClick={() => {
                onRemove(id);
              }}
            >
              Remove
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function Picker({
  ids,
  full,
  onToggle,
}: {
  ids: readonly string[];
  full: boolean;
  onToggle: (id: string) => void;
}) {
  // useState with an initializer: localStorage is read once, when the view
  // opens, not on every render.
  const [recent] = useState(readRecentRuns);
  // useState: the job id being typed into "Add by id".
  const [typed, setTyped] = useState("");
  // useState: why the typed id was rejected, if it was.
  const [typedError, setTypedError] = useState<string | null>(null);
  // useId: ties the "Add by id" label and error to its input.
  const inputId = useId();

  const add = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const id = typed.trim().toLowerCase();
    if (!UUID.test(id)) {
      setTypedError("A job id is a UUID, like 0b6c3f5e-8d1a-4a43-9c55-2f1e7d3b9a10.");
      return;
    }
    setTypedError(null);
    setTyped("");
    if (!ids.includes(id)) onToggle(id);
  };

  return (
    <div className="picker">
      <h3>Choose backtests</h3>
      {recent.length === 0 ? (
        <p className="hint">No backtests have been run from this browser yet.</p>
      ) : (
        <fieldset>
          <legend>Recent runs from this browser</legend>
          {recent.map((run) => {
            const checked = ids.includes(run.id);
            return (
              <label key={run.id} className="check">
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={!checked && full}
                  onChange={() => {
                    onToggle(run.id);
                  }}
                />{" "}
                {run.label} <code>{shortId(run.id)}</code>
              </label>
            );
          })}
        </fieldset>
      )}
      <form onSubmit={add} className="add-id" noValidate>
        <label htmlFor={inputId}>Add by job id</label>
        <input
          id={inputId}
          value={typed}
          aria-invalid={typedError ? true : undefined}
          aria-describedby={typedError ? `${inputId}-error` : undefined}
          onChange={(e) => {
            setTyped(e.target.value);
          }}
        />
        <button type="submit" disabled={full}>
          Add
        </button>
        {typedError && (
          <p id={`${inputId}-error`} className="field-error">
            {typedError}
          </p>
        )}
      </form>
    </div>
  );
}
