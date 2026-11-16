#!/usr/bin/env node
/**
 * Browser measurements of the production build, served by nginx. Run by
 * bench/web_bench.py inside the Playwright image (which has Chromium and
 * its system libraries), on the Compose project's network:
 *
 *   node scripts/measure-web.mjs --base http://web:8080 --job <id> \
 *        [--lighthouse-runs 3] [--chart-runs 10]
 *
 * Prints one JSON object:
 * - lighthouse: Lighthouse's performance category for the results page and
 *   the Run page, each run in a fresh Chrome (cold cache), Lighthouse's
 *   default settings (mobile emulation, simulated slow 4G and 4x CPU
 *   slowdown), so scores are comparable with other Lighthouse reports.
 * - first_chart: on the results page, milliseconds from the start of
 *   navigation until the equity curve's SVG path exists with data in it,
 *   recorded in the page by a MutationObserver, each run in a fresh browser
 *   context (cold cache), with no throttling.
 */
/* global document, window, MutationObserver -- in functions that run in the page */
import { chromium } from "@playwright/test";
import * as chromeLauncher from "chrome-launcher";
import lighthouse from "lighthouse";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 ? fallback : process.argv[i + 1];
}

const base = arg("base", "http://web:8080");
const job = arg("job");
if (!job) throw new Error("--job <id of a succeeded backtest> is required");
const lighthouseRuns = Number(arg("lighthouse-runs", "3"));
const chartRuns = Number(arg("chart-runs", "10"));

const pages = { results: `${base}/backtests/${job}`, run: `${base}/` };
const AUDITS = [
  "first-contentful-paint",
  "largest-contentful-paint",
  "total-blocking-time",
  "speed-index",
  "cumulative-layout-shift",
];

async function lighthouseRun(url) {
  const chrome = await chromeLauncher.launch({
    chromePath: chromium.executablePath(),
    // --no-sandbox: the container has no user namespaces for Chrome's
    // sandbox; the page measured is our own.
    chromeFlags: ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage"],
  });
  try {
    const result = await lighthouse(url, {
      port: chrome.port,
      onlyCategories: ["performance"],
      output: "json",
      logLevel: "error",
    });
    const lhr = result.lhr;
    if (lhr.runtimeError) throw new Error(`${url}: ${lhr.runtimeError.message}`);
    return {
      score: Math.round(lhr.categories.performance.score * 100),
      audits_ms: Object.fromEntries(AUDITS.map((a) => [a, lhr.audits[a]?.numericValue ?? null])),
      lighthouse_version: lhr.lighthouseVersion,
      user_agent: lhr.environment.hostUserAgent,
      warnings: lhr.runWarnings,
    };
  } finally {
    await chrome.kill();
  }
}

// Installed before any page script: notes performance.now() (milliseconds
// since navigation start) when the first equity-curve path with data appears.
const FIRST_CHART = () => {
  const found = () => {
    const path = document.querySelector("path.recharts-line-curve");
    return path !== null && (path.getAttribute("d") ?? "").length > 10;
  };
  new MutationObserver((_, observer) => {
    if (found()) {
      window.__firstChartMs = performance.now();
      observer.disconnect();
    }
  }).observe(document, { subtree: true, childList: true, attributes: true });
};

async function firstChart(browser) {
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.addInitScript(FIRST_CHART);
    await page.goto(pages.results);
    await page.waitForFunction(() => window.__firstChartMs !== undefined, null, {
      timeout: 30_000,
    });
    return await page.evaluate(() => window.__firstChartMs);
  } finally {
    await context.close();
  }
}

const out = { pages, lighthouse: {}, first_chart_ms: [] };
for (const [name, url] of Object.entries(pages)) {
  out.lighthouse[name] = [];
  for (let i = 0; i < lighthouseRuns; i++) {
    out.lighthouse[name].push(await lighthouseRun(url));
    console.error(`lighthouse ${name} ${String(i + 1)}/${String(lighthouseRuns)}`);
  }
}
const browser = await chromium.launch();
try {
  out.browser_version = browser.version();
  await firstChart(browser); // warm-up: not recorded (first launch, DNS)
  for (let i = 0; i < chartRuns; i++) out.first_chart_ms.push(await firstChart(browser));
} finally {
  await browser.close();
}
console.log(JSON.stringify(out));
