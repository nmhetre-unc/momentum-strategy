#!/usr/bin/env node
/**
 * Screenshots of each view for the README, taken from the real stack.
 * Run by e2e/screenshots.mjs inside the Playwright image, on the stack's
 * network:
 *
 *   node scripts/screenshots.mjs --base http://web:8080 --out /docs
 *
 * Submits three backtests through the API (the Run page's defaults for
 * sma_crossover, plus momentum and ml_direction on the same range), then
 * captures: the Run page after a run succeeded, an ml_direction results
 * page (its training period shaded), Compare with all three, and the Model
 * page after Predict. 1280 px wide, light theme, PNG.
 */
import { chromium } from "@playwright/test";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 ? fallback : process.argv[i + 1];
}

const base = arg("base", "http://web:8080");
const out = arg("out", "/docs");
const RANGE = { ticker: "SPY", start: "2015-01-01", end: "2025-01-01", n_boot: 2000 };

async function submit(request, body) {
  const response = await request.post(`${base}/api/backtests`, { data: { ...RANGE, ...body } });
  if (response.status() !== 202) throw new Error(`POST: ${String(response.status())}`);
  const { id } = await response.json();
  for (let i = 0; i < 240; i++) {
    const job = await (await request.get(`${base}/api/backtests/${id}`)).json();
    if (job.status === "succeeded") return id;
    if (job.status === "failed") throw new Error(job.last_error);
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("backtest did not finish");
}

const browser = await chromium.launch();
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
    colorScheme: "light",
  });
  const page = await context.newPage();
  const shot = async (name, fullPage = true) => {
    await page.screenshot({ path: `${out}/${name}.png`, fullPage, animations: "disabled" });
    console.error(`wrote ${name}.png`);
  };

  // Run: the form, then a job that has succeeded.
  await page.goto(`${base}/`);
  await page.getByLabel("Strategy").selectOption("sma_crossover");
  await page.getByRole("button", { name: "Run backtest" }).click();
  await page.getByRole("link", { name: "See the results" }).waitFor({ timeout: 60_000 });
  await shot("dashboard-run");

  const sma = new URL(
    await page.getByRole("link", { name: "See the results" }).getAttribute("href"),
    base,
  ).pathname
    .split("/")
    .pop();
  const momentum = await submit(context.request, {
    strategy: "momentum",
    params: { lookback: 20, threshold: 0 },
  });
  const ml = await submit(context.request, { strategy: "ml_direction", params: {} });

  // Results: a learning strategy, so the shading and the full-period label show.
  await page.goto(`${base}/backtests/${ml}`);
  await page.locator("path.recharts-line-curve").waitFor();
  await shot("dashboard-results");

  await page.goto(`${base}/compare?ids=${sma},${momentum},${ml}`);
  await page.locator("path.recharts-line-curve").nth(2).waitFor();
  await shot("dashboard-compare");

  await page.goto(`${base}/model`);
  await page.getByRole("button", { name: "Predict" }).click();
  await page.getByRole("list", { name: "Probabilities" }).waitFor();
  await shot("dashboard-model");
} finally {
  await browser.close();
}
