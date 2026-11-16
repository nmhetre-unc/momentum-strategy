/**
 * The dashboard end to end: the production build served by nginx, against
 * the real API, worker and database (e2e/stack.mjs ingests SPY and trains a
 * model first).
 *
 * Every test also fails on any Content-Security-Policy violation the
 * browser reports, any uncaught page error and any console error, so the
 * CSP nginx sends is one the app actually passes.
 */
import { expect, test as base, type APIRequestContext, type Page } from "@playwright/test";

const test = base.extend<{ guard: undefined }>({
  guard: [
    async ({ page }, use) => {
      const problems: string[] = [];
      page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));
      page.on("console", (message) => {
        if (message.type() === "error") problems.push(`console error: ${message.text()}`);
      });
      await page.addInitScript(() => {
        document.addEventListener("securitypolicyviolation", (event) => {
          console.error(`CSP violation: ${event.violatedDirective} ${event.blockedURI}`);
        });
      });
      await use(undefined);
      expect(problems, "CSP violations, page errors or console errors").toEqual([]);
    },
    { auto: true },
  ],
});

/** The equity curve's SVG path, with data in it: the chart really rendered. */
async function expectEquityCurve(page: Page) {
  const curve = page
    .getByRole("figure", { name: "Equity curve" })
    .locator("path.recharts-line-curve");
  await expect(curve).toBeVisible();
  await expect(curve).toHaveAttribute("d", /^M[\d.]+,[\d.]+(L[\d.]+,[\d.]+){100,}/);
  const drawdown = page
    .getByRole("figure", { name: "Drawdown" })
    .locator("path.recharts-area-curve");
  await expect(drawdown).toHaveAttribute("d", /^M/);
}

async function succeededBacktest(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/backtests", {
    data: {
      strategy: "momentum",
      ticker: "SPY",
      start: "2018-01-01",
      end: "2024-01-01",
      n_boot: 200,
      params: { lookback: 20, threshold: 0 },
    },
  });
  expect(response.status()).toBe(202);
  const { id } = (await response.json()) as { id: string };
  await expect
    .poll(
      async () =>
        ((await (await request.get(`/api/backtests/${id}`)).json()) as { status: string }).status,
      { timeout: 60_000 },
    )
    .toBe("succeeded");
  return id;
}

test("pick a strategy, run it, and see the chart", async ({ page }) => {
  const response = await page.goto("/");
  expect(response?.headers()["content-security-policy"]).toContain("script-src 'self'");
  expect(response?.headers()["x-content-type-options"]).toBe("nosniff");
  expect(response?.headers()["referrer-policy"]).toBe("no-referrer");

  await page.getByLabel("Strategy").selectOption("mean_reversion");
  await expect(page.getByText(/version \d+/).first()).toBeVisible();
  await page.getByLabel("RSI period").fill("10");
  await page.getByRole("button", { name: "Run backtest" }).click();

  const job = page.getByRole("region", { name: "Job progress" });
  await expect(job.getByText("succeeded")).toBeVisible({ timeout: 60_000 });
  await job.getByRole("link", { name: "See the results" }).click();

  await expect(page).toHaveURL(/\/backtests\/[0-9a-f-]{36}$/);
  await expectEquityCurve(page);
  await expect(page.getByText(/strategy version \d+/)).toBeVisible();
  await expect(page.getByText(/90% CI/).first()).toBeVisible();
});

test("the Model page makes no POST until Predict is pressed", async ({ page }) => {
  const predicts: string[] = [];
  page.on("request", (request) => {
    if (request.method() !== "GET") predicts.push(`${request.method()} ${request.url()}`);
  });

  await page.goto("/model");
  const verdict = page.getByRole("region", { name: "Against its baseline" });
  await expect(verdict.getByText(/beat/i).first()).toBeVisible();
  await expect(verdict).toContainText("beats:");
  await page.waitForLoadState("networkidle");
  expect(predicts).toEqual([]);

  await page.getByRole("button", { name: "Predict" }).click();
  const prediction = page.getByRole("region", { name: "Prediction" });
  await expect(prediction.getByRole("list", { name: "Probabilities" })).toBeVisible();
  await page.waitForLoadState("networkidle");
  expect(predicts).toHaveLength(1);
  expect(predicts[0]).toMatch(/^POST .*\/api\/predict$/);
});

test("a results URL opened directly is served by the SPA fallback", async ({ page, request }) => {
  const id = await succeededBacktest(request);

  // A fresh page load of a deep path: nginx must answer with index.html.
  const response = await page.goto(`/backtests/${id}`);
  expect(response?.status()).toBe(200);
  expect(response?.headers()["content-type"]).toContain("text/html");
  expect(response?.headers()["cache-control"]).toBe("no-cache");
  await expectEquityCurve(page);

  // A compare URL too, with the same job twice listed once.
  await page.goto(`/compare?ids=${id},${id}`);
  await expect(page.getByText("Pick at least two completed backtests")).toBeVisible();
});

test("nginx: hashed assets are long-lived, a missing one is a 404", async ({ request }) => {
  const html = await (await request.get("/")).text();
  const asset = /\/assets\/[^"]+\.js/.exec(html)?.[0];
  expect(asset).toBeDefined();
  const response = await request.get(asset ?? "");
  expect(response.headers()["cache-control"]).toBe("public, max-age=31536000, immutable");
  expect((await request.get("/assets/missing-0000.js")).status()).toBe(404);
  const api = await request.get("/api/backtests/not-a-uuid");
  expect(api.status()).toBe(422);
  expect(api.headers()["content-type"]).toBe("application/problem+json");
});
