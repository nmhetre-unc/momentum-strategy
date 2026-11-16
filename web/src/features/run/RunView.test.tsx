import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { fakeApi, json, on, problem, renderWithApi, sleep, type Call } from "../../test/api";
import { JOB_ID, STRATEGIES, job } from "../../test/fixtures";
import { readRecentRuns } from "../compare/recentRuns";
import { RunView } from "./RunView";

const POLL_MS = 10;
const JOB_PATH = `/backtests/${JOB_ID}`;
const accepted = (status = 202) => json({ id: JOB_ID, status: "queued", href: JOB_PATH }, status);

const strategies = on("GET", "/strategies", () => json(STRATEGIES));

/** GET /backtests/{id} answers each status in turn, then repeats the last. */
function jobSequence(...statuses: ("queued" | "running" | "succeeded" | "failed")[]) {
  let n = 0;
  return on("GET", JOB_PATH, () => {
    const status = statuses[Math.min(n++, statuses.length - 1)] ?? "queued";
    return json(job({ status }));
  });
}

const posts = (calls: Call[]) => calls.filter((c) => c.method === "POST");
const keyOf = (call: Call | undefined) => call?.headers.get("Idempotency-Key");

function renderRun(handlers: Parameters<typeof fakeApi>[0]) {
  const api = fakeApi(handlers);
  const view = renderWithApi(<RunView pollIntervalMs={POLL_MS} retryDelayMs={0} />, api.fetch);
  return { ...api, ...view, user: userEvent.setup() };
}

async function replace(user: ReturnType<typeof userEvent.setup>, label: string, text: string) {
  const input = screen.getByLabelText(label);
  await user.clear(input);
  if (text) await user.type(input, text);
}

test("shows a loading state, then a form generated from the strategy's JSON Schema", async () => {
  renderRun([strategies]);
  expect(screen.getByRole("status")).toHaveTextContent("Loading strategies");

  expect(await screen.findByLabelText("Short SMA window")).toHaveValue(50);
  expect(screen.getByLabelText("Long SMA window")).toHaveValue(200);
  expect(screen.getByText("version 1")).toBeInTheDocument();
  expect(screen.getByText(/5 to 100\./)).toBeInTheDocument();
});

test("switching strategy shows that strategy's parameters and version", async () => {
  const { user } = renderRun([strategies]);
  await user.selectOptions(await screen.findByLabelText("Strategy"), "ml_direction");
  expect(screen.getByLabelText("Train fraction")).toHaveValue(0.7);
  expect(screen.queryByLabelText("Short SMA window")).not.toBeInTheDocument();
  expect(screen.getByText("version 2")).toBeInTheDocument();
});

test("validates on the client like the API, and sends nothing while invalid", async () => {
  const { user, calls } = renderRun([strategies]);
  await screen.findByLabelText("Short SMA window");

  await replace(user, "Short SMA window", "90");
  await replace(user, "Long SMA window", "60");
  await replace(user, "Ticker", "bad ticker!");
  await replace(user, "End", "2014-01-01");
  await replace(user, "Bootstrap replicates", "10.5");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));

  // Both in range, but the API also requires short_window < long_window.
  expect(screen.getByText("Must be greater than short_window (90).")).toBeInTheDocument();
  expect(screen.getByText(/letters, digits and/)).toBeInTheDocument();
  expect(screen.getByText(/End must be after start/)).toBeInTheDocument();
  expect(screen.getByText("Enter a whole number.")).toBeInTheDocument();
  expect(screen.getByLabelText("Ticker")).toHaveAttribute("aria-invalid", "true");
  expect(posts(calls)).toHaveLength(0);

  await replace(user, "Short SMA window", "250");
  expect(screen.getByText("Must be at most 100.")).toBeInTheDocument();

  // A float parameter: bounds from its schema.
  await user.selectOptions(screen.getByLabelText("Strategy"), "ml_direction");
  await replace(user, "Train fraction", "0.95");
  expect(screen.getByText("Must be at most 0.9.")).toBeInTheDocument();
});

test("submits the validated request, polls until the job succeeds, then stops", async () => {
  const { user, calls, count } = renderRun([
    strategies,
    on("POST", "/backtests", () => accepted()),
    jobSequence("queued", "running", "succeeded"),
  ]);
  await screen.findByLabelText("Short SMA window");
  await replace(user, "Ticker", " qqq ");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));

  const link = await screen.findByRole("link", { name: "See the results" });
  expect(link).toHaveAttribute("href", JOB_PATH);
  expect(posts(calls)[0]?.body).toEqual({
    strategy: "sma_crossover",
    params: { short_window: 50, long_window: 200 },
    ticker: "QQQ",
    start: "2015-01-01",
    end: "2025-01-01",
    cost_bps: 5,
    n_boot: 2000,
    seed: 0,
  });
  expect(keyOf(posts(calls)[0])).toMatch(/^"[0-9a-f]{32}"$/);
  expect(count("GET", JOB_PATH)).toBe(3);

  // Terminal: no more polls, however long we wait.
  await sleep(POLL_MS * 8);
  expect(count("GET", JOB_PATH)).toBe(3);
  expect(
    within(screen.getByRole("region", { name: "Job progress" })).getByText("succeeded"),
  ).toBeInTheDocument();
  expect(readRecentRuns()[0]?.id).toBe(JOB_ID);
});

test("a failed job stops polling and shows why", async () => {
  const { user, count } = renderRun([
    strategies,
    on("POST", "/backtests", () => accepted()),
    jobSequence("running", "failed"),
  ]);
  await screen.findByLabelText("Short SMA window");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));

  expect(
    await screen.findByText(/The backtest failed: invalid request: no bars/),
  ).toBeInTheDocument();
  const polls = count("GET", JOB_PATH);
  await sleep(POLL_MS * 8);
  expect(count("GET", JOB_PATH)).toBe(polls);
});

test("polling stops when the view unmounts", async () => {
  const { user, count, unmount } = renderRun([
    strategies,
    on("POST", "/backtests", () => accepted()),
    jobSequence("running"),
  ]);
  await screen.findByLabelText("Short SMA window");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));
  await waitFor(() => {
    expect(count("GET", JOB_PATH)).toBeGreaterThanOrEqual(3);
  });

  unmount();
  const polls = count("GET", JOB_PATH);
  await sleep(POLL_MS * 8);
  expect(count("GET", JOB_PATH)).toBe(polls);
});

test("a transient failure is retried automatically with the same Idempotency-Key", async () => {
  let n = 0;
  const { user, calls } = renderRun([
    strategies,
    on("POST", "/backtests", () => (n++ === 0 ? problem(503, "database-unavailable") : accepted())),
    jobSequence("succeeded"),
  ]);
  await screen.findByLabelText("Short SMA window");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));

  await screen.findByRole("link", { name: "See the results" });
  const [first, second] = posts(calls);
  expect(posts(calls)).toHaveLength(2);
  expect(keyOf(second)).toBe(keyOf(first));
  expect(second?.body).toEqual(first?.body);
});

test("'Try again' resends the same key; a new submission gets a new one", async () => {
  let down = true;
  const { user, calls } = renderRun([
    strategies,
    on("POST", "/backtests", () => {
      if (down) throw new TypeError("Failed to fetch");
      return accepted();
    }),
    jobSequence("succeeded"),
  ]);
  await screen.findByLabelText("Short SMA window");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));

  // The first attempt and both automatic retries fail: 3 posts, one key.
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("No response from the API");
  expect(posts(calls)).toHaveLength(3);
  const key = keyOf(posts(calls)[0]);
  expect(new Set(posts(calls).map(keyOf))).toEqual(new Set([key]));

  down = false;
  await user.click(within(alert).getByRole("button", { name: "Try again" }));
  await screen.findByRole("link", { name: "See the results" });
  expect(posts(calls)).toHaveLength(4);
  expect(keyOf(posts(calls)[3])).toBe(key);

  // Submitting again is a new submission, even with the same values.
  await user.click(screen.getByRole("button", { name: "Run backtest" }));
  await waitFor(() => {
    expect(posts(calls)).toHaveLength(5);
  });
  expect(keyOf(posts(calls)[4])).not.toBe(key);
});

test("the API's validation errors land on the form's fields, with no retry offered", async () => {
  const { user } = renderRun([
    strategies,
    on("POST", "/backtests", () =>
      problem(422, "validation-error", {
        errors: [
          {
            loc: ["body", "sma_crossover", "ticker"],
            msg: "Value error, ticker may contain only letters",
            type: "value_error",
          },
        ],
      }),
    ),
  ]);
  await screen.findByLabelText("Short SMA window");
  await user.click(screen.getByRole("button", { name: "Run backtest" }));

  expect(await screen.findByText("ticker may contain only letters")).toHaveAttribute(
    "class",
    "field-error",
  );
  expect(screen.getByRole("alert")).toHaveTextContent("The backtest was not submitted");
  expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
});

test("a problem loading the strategies is shown, with a retry", async () => {
  let fail = true;
  const { user } = renderRun([
    on("GET", "/strategies", () =>
      fail ? problem(503, "database-unavailable") : json(STRATEGIES),
    ),
  ]);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Could not load the strategies");
  expect(alert).toHaveTextContent("detail of database-unavailable");

  fail = false;
  await user.click(within(alert).getByRole("button", { name: "Try again" }));
  expect(await screen.findByLabelText("Short SMA window")).toBeInTheDocument();
});

test("no strategies is an empty state", async () => {
  renderRun([on("GET", "/strategies", () => json({ strategies: [] }))]);
  expect(await screen.findByText("No strategies")).toBeInTheDocument();
});
