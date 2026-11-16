/**
 * API responses for tests, typed with the generated schema so a fixture
 * that no longer matches the API is a compile error. The parameter schemas
 * are copied from GET /strategies.
 */
import type {
  BacktestJob,
  BacktestSeries,
  ModelInfo,
  Prediction,
  StrategyInfo,
} from "../api/types";

export const SMA: StrategyInfo = {
  name: "sma_crossover",
  family: "Trend-following",
  description: "Goes long when the short moving average is above the long one, flat otherwise.",
  params_schema: {
    additionalProperties: false,
    properties: {
      short_window: {
        type: "integer",
        minimum: 5,
        maximum: 100,
        default: 50,
        title: "Short SMA window",
        description: "Days in the fast average.",
      },
      long_window: {
        type: "integer",
        minimum: 50,
        maximum: 300,
        default: 200,
        title: "Long SMA window",
        description: "Days in the slow average.",
      },
    },
    title: "SmaCrossoverParams",
    type: "object",
  },
  defaults: { short_window: 50, long_window: 200 },
  version: 1,
  schema_hash: "a".repeat(64),
};

export const ML: StrategyInfo = {
  name: "ml_direction",
  family: "Machine learning",
  description: "Trains a classifier on lagged features to predict tomorrow's direction.",
  params_schema: {
    additionalProperties: false,
    properties: {
      train_frac: {
        type: "number",
        minimum: 0.5,
        maximum: 0.9,
        default: 0.7,
        title: "Train fraction",
        description: "Share of history used for fitting.",
      },
    },
    title: "MlDirectionParams",
    type: "object",
  },
  defaults: { train_frac: 0.7 },
  version: 2,
  schema_hash: "b".repeat(64),
};

export const STRATEGIES = { strategies: [SMA, ML] };

export const JOB_ID = "0b6c3f5e-8d1a-4a43-9c55-2f1e7d3b9a10";
export const JOB_ID_2 = "1c7d4f6e-9e2b-4b54-8d66-3f2e8e4cab21";

const METRICS = {
  total_return: 0.42,
  cagr: 0.075,
  annualized_volatility: 0.12,
  sharpe_ratio: 0.65,
  sharpe_ci_low: 0.14,
  sharpe_ci_high: 1.19,
  sortino_ratio: 0.9,
  max_drawdown: -0.22,
  exposure: 0.7,
  turnover: 0.01,
  num_trades: 25,
  win_rate: 0.55,
};

export function job(overrides: Partial<BacktestJob> = {}): BacktestJob {
  const status = overrides.status ?? "succeeded";
  const done = status === "succeeded";
  return {
    id: JOB_ID,
    status,
    attempts: status === "queued" ? 0 : 1,
    max_attempts: 3,
    created_at: "2026-11-15T10:00:00Z",
    run_after: "2026-11-15T10:00:00Z",
    started_at: status === "queued" ? null : "2026-11-15T10:00:01Z",
    heartbeat_at: null,
    finished_at: done || status === "failed" ? "2026-11-15T10:00:02Z" : null,
    last_error: status === "failed" ? "invalid request: no bars" : null,
    request: {
      strategy: "sma_crossover",
      params: { short_window: 50, long_window: 200 },
      ticker: "SPY",
      start: "2015-01-01",
      end: "2025-01-01",
      cost_bps: 5,
      n_boot: 2000,
      seed: 0,
    },
    price_series_id: 1,
    strategy_version: 1,
    metrics: done ? METRICS : null,
    result: done
      ? {
          attempt: 1,
          qbt_version: "0.1.0",
          numpy_version: "2.3.0",
          bootstrap_impl: "cpp",
          duration_ms: 71,
          n_days: 5,
          created_at: "2026-11-15T10:00:02Z",
        }
      : null,
    ...overrides,
  };
}

export function series(overrides: Partial<BacktestSeries> = {}): BacktestSeries {
  return {
    job_id: JOB_ID,
    strategy: "sma_crossover",
    strategy_version: 1,
    dates: ["2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07", "2015-01-08"],
    equity: [1, 1.01, 0.99, 1.02, 1.03],
    drawdown: [0, 0, -0.0198, 0, 0],
    position: [0, 1, 1, 1, 1],
    out_of_sample_from: null,
    ...overrides,
  };
}

const BASELINE = {
  name: "always_up",
  description: "Predicts the most common class in training every month.",
  accuracy: 0.613,
  difference: -0.032,
  difference_ci: [-0.11, 0.05],
  mcnemar_p: 0.48,
  beats: false,
};

const OOS = {
  accuracy: 0.581,
  accuracy_ci: [0.49, 0.67],
  n_months: 93,
  first_test_decision: "2017-03-31",
  last_test_decision: "2024-11-29",
  auc: 0.52,
  log_loss: 0.69,
  brier: 0.49,
};

export function model(overrides: Partial<ModelInfo> = {}): ModelInfo {
  return {
    version: 3,
    href: "/models/3",
    model_type: "logistic",
    target: "direction",
    predicts: "the S&P 500's direction over the next month",
    classes: [
      { label: 0, name: "down" },
      { label: 1, name: "up" },
    ],
    feature_set: "monthly_v1",
    features: ["ret_1m", "ret_12m"],
    price_series_id: 1,
    training_window: {
      first_decision: "2008-12-31",
      last_decision: "2017-02-28",
      last_label_end: "2017-03-31",
      n_months: 99,
    },
    seed: 0,
    hyperparameters: { C: 1 },
    out_of_sample: OOS,
    baseline: BASELINE,
    summary: "Out of sample, 58.1% accuracy; the always_up baseline gets 61.3%.",
    serving: { as_of: "2024-12-31", features_sha256: "c".repeat(64) },
    artifact: { format: "skops", sha256: "d".repeat(64), bytes: 1234 },
    library_versions: { "scikit-learn": "1.7.0" },
    created_at: "2026-11-01T12:00:00Z",
    ...overrides,
  };
}

export function prediction(overrides: Partial<Prediction> = {}): Prediction {
  return {
    prediction_id: 11,
    created_at: "2026-11-15T10:00:00Z",
    model_version: 3,
    model_href: "/models/3",
    target: "direction",
    predicts: "the S&P 500's direction over the next month",
    as_of: "2024-12-31",
    prediction: 1,
    prediction_name: "up",
    probabilities: [
      { label: 0, name: "down", probability: 0.38 },
      { label: 1, name: "up", probability: 0.62 },
    ],
    features_sha256: "c".repeat(64),
    out_of_sample: OOS,
    baseline: BASELINE,
    summary: "Out of sample, 58.1% accuracy; the always_up baseline gets 61.3%.",
    ...overrides,
  };
}
