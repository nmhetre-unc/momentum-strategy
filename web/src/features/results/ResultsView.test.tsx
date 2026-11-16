import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { fakeApi, json, on, problem, renderWithApi, sleep } from "../../test/api";
import { JOB_ID, job, series } from "../../test/fixtures";
import { ResultsView } from "./ResultsView";

const POLL_MS = 10;
const JOB_PATH = `/backtests/${JOB_ID}`;
const SERIES_PATH = `${JOB_PATH}/series`;

function renderResults(handlers: Parameters<typeof fakeApi>[0]) {
  const api = fakeApi(handlers);
  const view = renderWithApi(<ResultsView id={JOB_ID} pollIntervalMs={POLL_MS} />, api.fetch);
  return { ...api, ...view };
}

test("shows the charts, Sharpe with its interval, and the strategy version", async () => {
  renderResults([
    on("GET", JOB_PATH, () => json(job())),
    on("GET", SERIES_PATH, () => json(series())),
  ]);
  expect(screen.getByRole("status")).toHaveTextContent("Loading the backtest");

  expect(await screen.findByRole("figure", { name: "Equity curve" })).toBeInTheDocument();
  expect(screen.getByRole("figure", { name: "Drawdown" })).toBeInTheDocument();
  expect(screen.getByText("strategy version 1")).toBeInTheDocument();
  expect(screen.getAllByText("90% CI 0.14 to 1.19").length).toBeGreaterThan(0);
  expect(screen.getAllByText("-22.0%").length).toBeGreaterThan(0);
  // Not a learning strategy: no training period, no full-period label.
  expect(screen.queryByText(/training months held flat/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Shaded: the training period/)).not.toBeInTheDocument();
});

test("a learning strategy shades its training period and labels the metrics full-period", async () => {
  renderResults([
    on("GET", JOB_PATH, () =>
      json(job({ strategy_version: 2, request: { ...job().request, strategy: "ml_direction" } })),
    ),
    on("GET", SERIES_PATH, () =>
      json(
        series({ strategy: "ml_direction", strategy_version: 2, out_of_sample_from: "2015-01-06" }),
      ),
    ),
  ]);

  const note = await screen.findByRole("note");
  expect(note).toHaveTextContent("full period: training months held flat and included.");
  expect(note).toHaveTextContent("holds no position before 2015-01-06");
  expect(screen.getByRole("figure", { name: "Equity curve" })).toHaveTextContent(
    "Shaded: the training period, 2015-01-02 up to 2015-01-06, held flat",
  );
  expect(
    within(screen.getByRole("table")).getByText(/training months held flat/),
  ).toBeInTheDocument();
  expect(screen.getByText("strategy version 2")).toBeInTheDocument();
});

test("with n_boot 0 the missing interval is explained", async () => {
  const noCi = job({
    request: { ...job().request, n_boot: 0 },
    metrics: { ...job().metrics, sharpe_ci_low: null, sharpe_ci_high: null },
  });
  renderResults([
    on("GET", JOB_PATH, () => json(noCi)),
    on("GET", SERIES_PATH, () => json(series())),
  ]);
  expect(
    (await screen.findAllByText("no interval: run with 0 bootstrap replicates")).length,
  ).toBeGreaterThan(0);
  expect(screen.queryByText(/90% CI/)).not.toBeInTheDocument();
});

test("a running job is polled until it succeeds, and the series is fetched only then", async () => {
  let n = 0;
  const { count } = renderResults([
    on("GET", JOB_PATH, () => json(job({ status: n++ < 2 ? "running" : "succeeded" }))),
    on("GET", SERIES_PATH, () => json(series())),
  ]);
  expect(await screen.findByText(/The backtest is running/)).toBeInTheDocument();
  expect(count("GET", SERIES_PATH)).toBe(0);

  expect(await screen.findByRole("figure", { name: "Equity curve" })).toBeInTheDocument();
  expect(count("GET", SERIES_PATH)).toBe(1);
  const polls = count("GET", JOB_PATH);
  await sleep(POLL_MS * 8);
  expect(count("GET", JOB_PATH)).toBe(polls);
});

test("a failed job shows its last error and no charts", async () => {
  const { count } = renderResults([on("GET", JOB_PATH, () => json(job({ status: "failed" })))]);
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "The backtest failed: invalid request: no bars",
  );
  expect(screen.queryByRole("figure")).not.toBeInTheDocument();
  expect(count("GET", SERIES_PATH)).toBe(0);
});

test("an unknown job is the API's 404 problem", async () => {
  renderResults([on("GET", JOB_PATH, () => problem(404, "job-not-found"))]);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Could not load the backtest");
  expect(alert).toHaveTextContent("(404)");
  expect(alert).toHaveTextContent("/problems/job-not-found");
});

test("a problem loading the series is shown, and can be retried", async () => {
  let fail = true;
  renderResults([
    on("GET", JOB_PATH, () => json(job())),
    on("GET", SERIES_PATH, () => (fail ? problem(503, "database-unavailable") : json(series()))),
  ]);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Could not load the equity curve");

  fail = false;
  await userEvent.setup().click(within(alert).getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("figure", { name: "Equity curve" })).toBeInTheDocument();
});

test("a series with no days is an empty state", async () => {
  renderResults([
    on("GET", JOB_PATH, () => json(job())),
    on("GET", SERIES_PATH, () =>
      json(series({ dates: [], equity: [], drawdown: [], position: [] })),
    ),
  ]);
  expect(await screen.findByText("No trading days")).toBeInTheDocument();
});
