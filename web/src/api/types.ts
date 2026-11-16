/** Short names for the API types the views use, all from the generated schema. */
import type { components, paths } from "./schema";

type Schemas = components["schemas"];

export type StrategyInfo = Schemas["StrategyInfo"];
export type BacktestJob = Schemas["BacktestJob"];
export type JobStatus = BacktestJob["status"];
export type BacktestAccepted = Schemas["BacktestAccepted"];
export type BacktestSeries = Schemas["BacktestSeries"];
export type ModelInfo = Schemas["ModelInfo"];
export type Prediction = Schemas["Prediction"];
export type BaselineComparison = Schemas["BaselineComparison"];

/** The POST /backtests body: a union with one member per strategy. */
export type BacktestRequest =
  paths["/backtests"]["post"]["requestBody"]["content"]["application/json"];

/** `succeeded` and `failed` are final: a job in either never changes again. */
export function isTerminal(status: JobStatus): boolean {
  return status === "succeeded" || status === "failed";
}
