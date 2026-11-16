import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../../App";
import { fakeApi, json, on, problem, renderWithApi } from "../../test/api";
import { JOB_ID, JOB_ID_2, job, series } from "../../test/fixtures";
import { CompareView } from "./CompareView";
import { recordRecentRun } from "./recentRuns";

const ML_JOB = job({
  id: JOB_ID_2,
  strategy_version: 2,
  request: { ...job().request, strategy: "ml_direction", ticker: "QQQ" },
  metrics: { ...job().metrics, sharpe_ratio: 0.31, total_return: 0.12 },
});

const both = [
  on("GET", `/backtests/${JOB_ID}`, () => json(job())),
  on("GET", `/backtests/${JOB_ID_2}`, () => json(ML_JOB)),
  on("GET", `/backtests/${JOB_ID}/series`, () => json(series())),
  on("GET", `/backtests/${JOB_ID_2}/series`, () =>
    json(series({ job_id: JOB_ID_2, out_of_sample_from: "2015-01-06" })),
  ),
];

function renderCompare(ids: string[], handlers: Parameters<typeof fakeApi>[0]) {
  const api = fakeApi(handlers);
  const view = renderWithApi(<CompareView ids={ids} pollIntervalMs={10} />, api.fetch);
  return { ...api, ...view };
}

test("two succeeded backtests side by side: one table, one overlaid chart", async () => {
  renderCompare([JOB_ID, JOB_ID_2], both);
  expect(screen.getByRole("status")).toHaveTextContent("Loading the backtests");

  expect(await screen.findByRole("figure", { name: "Equity curves" })).toBeInTheDocument();
  const table = screen.getByRole("table");
  const headers = within(table)
    .getAllByRole("columnheader")
    .map((h) => h.textContent);
  expect(headers[1]).toContain("sma_crossover v1 · SPY · 0b6c3f5e");
  expect(headers[2]).toContain("ml_direction v2 · QQQ · 1c7d4f6e");
  // The learning strategy's column carries the full-period label.
  expect(headers[2]).toContain("full period: training months held flat and included");
  expect(headers[1]).not.toContain("full period");

  const sharpe = within(table).getByRole("row", { name: /Sharpe ratio/ });
  expect(sharpe).toHaveTextContent("0.65");
  expect(sharpe).toHaveTextContent("0.31");
  expect(within(table).getByRole("row", { name: /Total return/ })).toHaveTextContent("42.0%12.0%");
});

test("fewer than two selected is an empty state, and nothing is fetched", async () => {
  const { calls } = renderCompare([JOB_ID], both);
  expect(await screen.findByText("Pick at least two completed backtests")).toBeInTheDocument();
  expect(calls.filter((c) => c.path.endsWith("/series"))).toHaveLength(0);
});

test("a backtest that hasn't succeeded is marked and left out", async () => {
  renderCompare(
    [JOB_ID, JOB_ID_2],
    [
      on("GET", `/backtests/${JOB_ID}`, () => json(job())),
      on("GET", `/backtests/${JOB_ID_2}`, () => json(job({ id: JOB_ID_2, status: "failed" }))),
      on("GET", `/backtests/${JOB_ID}/series`, () => json(series())),
    ],
  );
  expect(await screen.findByText("Not enough completed backtests")).toBeInTheDocument();
  expect(screen.getByText("failed: not comparable")).toBeInTheDocument();
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});

test("an unknown id shows the API's problem next to it", async () => {
  renderCompare(
    [JOB_ID, JOB_ID_2],
    [
      on("GET", `/backtests/${JOB_ID}`, () => json(job())),
      on("GET", `/backtests/${JOB_ID_2}`, () => problem(404, "job-not-found")),
      on("GET", `/backtests/${JOB_ID}/series`, () => json(series())),
    ],
  );
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Backtest 1c7d4f6e: job not found (404)");
});

test("recent runs and typed ids change the selection, which lives in the URL", async () => {
  recordRecentRun({ id: JOB_ID, label: "sma_crossover · SPY" });
  window.history.replaceState(null, "", "/compare");
  const user = userEvent.setup();
  renderWithApi(<App />, fakeApi(both).fetch);

  // The Compare page is loaded lazily (App.tsx): find* waits for it.
  await user.click(await screen.findByRole("checkbox", { name: /sma_crossover · SPY/ }));
  expect(window.location.search).toBe(`?ids=${JOB_ID}`);
  expect(screen.getByText("Pick at least two completed backtests")).toBeInTheDocument();

  const input = screen.getByLabelText("Add by job id");
  await user.type(input, "not-a-uuid");
  await user.click(screen.getByRole("button", { name: "Add" }));
  expect(screen.getByText(/A job id is a UUID/)).toBeInTheDocument();

  await user.clear(input);
  await user.type(input, JOB_ID_2.toUpperCase());
  await user.click(screen.getByRole("button", { name: "Add" }));
  expect(window.location.search).toBe(`?ids=${JOB_ID},${JOB_ID_2}`);
  expect(await screen.findByRole("figure", { name: "Equity curves" })).toBeInTheDocument();
});
