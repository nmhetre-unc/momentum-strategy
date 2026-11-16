/**
 * The Run form without React: parameter fields generated from a strategy's
 * JSON Schema (GET /strategies), and validation that accepts exactly what
 * POST /backtests accepts.
 *
 * Where the rules come from:
 * - Each parameter's type (integer or number), minimum, maximum, default,
 *   title and description: the strategy's `params_schema`, read at runtime,
 *   so a new parameter or a changed bound needs no change here.
 * - Rules the JSON Schema can't express, mirrored by hand from
 *   src/qbt_service/models.py: ORDERED_PARAMS (short window below long
 *   window, oversold below overbought), the ticker pattern, start before
 *   end, and the cost_bps / n_boot / seed bounds of the request itself.
 *   If the two ever disagree, the API's 422 still wins:
 *   fieldErrorsFromProblem() puts its errors on the same fields.
 */
import type { ValidationErrorItem } from "../../api/problem";
import type { BacktestRequest, StrategyInfo } from "../../api/types";

export interface ParamField {
  name: string;
  kind: "integer" | "number";
  title: string;
  description: string;
  minimum: number | undefined;
  maximum: number | undefined;
  default: number | undefined;
}

/** Mirrors ORDERED_PARAMS in src/qbt_service/models.py: params[lower] < params[upper]. */
export const ORDERED_PARAMS: Readonly<Record<string, readonly (readonly [string, string])[]>> = {
  sma_crossover: [["short_window", "long_window"]],
  mean_reversion: [["oversold", "overbought"]],
};

/** The request-level bounds in src/qbt_service/models.py (MAX_COST_BPS, MAX_N_BOOT). */
export const LIMITS = {
  cost_bps: { min: 0, max: 100 },
  n_boot: { min: 0, max: 5000 },
  // The API allows up to 2**63 - 1; a JavaScript number is exact only up
  // to 2**53 - 1, so the form stops there rather than send a rounded seed.
  seed: { min: 0, max: Number.MAX_SAFE_INTEGER },
} as const;

/** TICKER_PATTERN in src/qbt_service/models.py, after trimming and upper-casing. */
const TICKER_PATTERN = /^[A-Z0-9.\-^=]{1,15}$/;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

/** A real calendar date written YYYY-MM-DD: 2023-02-29 has the right shape but isn't one. */
function isIsoDate(text: string): boolean {
  if (!ISO_DATE.test(text)) return false;
  const [y, m, d] = text.split("-").map(Number) as [number, number, number];
  // setUTCFullYear, not Date.UTC, which reads years 0-99 as 1900-1999.
  const date = new Date(0);
  date.setUTCFullYear(y, m - 1, d);
  return date.getUTCFullYear() === y && date.getUTCMonth() === m - 1 && date.getUTCDate() === d;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * The form fields for a params JSON Schema, in the schema's order, and the
 * names of any properties the form can't render (anything but a plain
 * integer or number).
 */
export function paramFields(schema: Record<string, unknown>): {
  fields: ParamField[];
  unsupported: string[];
} {
  const properties = isRecord(schema["properties"]) ? schema["properties"] : {};
  const fields: ParamField[] = [];
  const unsupported: string[] = [];
  for (const [name, raw] of Object.entries(properties)) {
    const type = isRecord(raw) ? raw["type"] : undefined;
    if (!isRecord(raw) || (type !== "integer" && type !== "number")) {
      unsupported.push(name);
      continue;
    }
    fields.push({
      name,
      kind: type,
      title: typeof raw["title"] === "string" ? raw["title"] : name,
      description: typeof raw["description"] === "string" ? raw["description"] : "",
      minimum: numberOrUndefined(raw["minimum"]),
      maximum: numberOrUndefined(raw["maximum"]),
      default: numberOrUndefined(raw["default"]),
    });
  }
  return { fields, unsupported };
}

/** What the inputs hold: always strings, exactly as typed. */
export interface FormValues {
  ticker: string;
  start: string;
  end: string;
  cost_bps: string;
  n_boot: string;
  seed: string;
  params: Record<string, string>;
}

export const DEFAULT_VALUES: Omit<FormValues, "params"> = {
  ticker: "SPY",
  start: "2015-01-01",
  end: "2025-01-01",
  cost_bps: "5",
  n_boot: "2000",
  seed: "0",
};

/** A strategy's parameter defaults, as input strings. */
export function defaultParams(strategy: StrategyInfo): Record<string, string> {
  const { fields } = paramFields(strategy.params_schema);
  const values: Record<string, string> = {};
  for (const field of fields) {
    const value = numberOrUndefined(strategy.defaults[field.name]) ?? field.default;
    values[field.name] = value === undefined ? "" : String(value);
  }
  return values;
}

/** Field errors by field: "ticker", "start", ..., "params.<name>", or "form". */
export type FieldErrors = Record<string, string>;

function parseNumber(
  text: string,
  {
    kind,
    min,
    max,
  }: { kind: "integer" | "number"; min?: number | undefined; max?: number | undefined },
): number | string {
  const trimmed = text.trim();
  if (trimmed === "") return "Enter a value.";
  const value = Number(trimmed);
  if (!Number.isFinite(value)) return "Enter a number.";
  if (kind === "integer" && !Number.isInteger(value)) return "Enter a whole number.";
  if (min !== undefined && value < min) return `Must be at least ${String(min)}.`;
  if (max !== undefined && value > max) return `Must be at most ${String(max)}.`;
  return value;
}

/**
 * The POST /backtests body for these values, or the errors that stop it.
 * Every rule the API applies is checked here, so a valid form is a request
 * the API accepts (unless no price snapshot covers the range, which only
 * the API can know).
 */
export function validate(
  strategy: StrategyInfo,
  values: FormValues,
): { body: BacktestRequest; errors?: undefined } | { body?: undefined; errors: FieldErrors } {
  const errors: FieldErrors = {};

  const ticker = values.ticker.trim().toUpperCase();
  if (!ticker) errors["ticker"] = "Enter a ticker.";
  else if (!TICKER_PATTERN.test(ticker))
    errors["ticker"] = "Use letters, digits and . - ^ = only, at most 15 characters.";

  for (const name of ["start", "end"] as const) {
    if (!isIsoDate(values[name])) errors[name] = "Enter a valid date.";
  }
  if (!errors["start"] && !errors["end"] && values.start >= values.end) {
    errors["end"] = "End must be after start (the end date itself is not included).";
  }

  const request: Record<string, number> = {};
  for (const name of ["cost_bps", "n_boot", "seed"] as const) {
    const parsed = parseNumber(values[name], {
      kind: name === "cost_bps" ? "number" : "integer",
      ...LIMITS[name],
    });
    if (typeof parsed === "string") errors[name] = parsed;
    else request[name] = parsed;
  }

  const params: Record<string, number> = {};
  for (const field of paramFields(strategy.params_schema).fields) {
    const parsed = parseNumber(values.params[field.name] ?? "", {
      kind: field.kind,
      min: field.minimum,
      max: field.maximum,
    });
    if (typeof parsed === "string") errors[`params.${field.name}`] = parsed;
    else params[field.name] = parsed;
  }
  for (const [lower, upper] of ORDERED_PARAMS[strategy.name] ?? []) {
    const a = params[lower];
    const b = params[upper];
    if (a !== undefined && b !== undefined && a >= b) {
      errors[`params.${upper}`] = `Must be greater than ${lower} (${String(a)}).`;
    }
  }

  if (Object.keys(errors).length > 0) return { errors };
  // The body is assembled from a schema read at runtime, so TypeScript
  // can't see which member of the request union it is; the checks above
  // are what make it one.
  const body = {
    strategy: strategy.name,
    params,
    ticker,
    start: values.start,
    end: values.end,
    cost_bps: request["cost_bps"],
    n_boot: request["n_boot"],
    seed: request["seed"],
  } as BacktestRequest;
  return { body };
}

const FORM_FIELDS = new Set(["ticker", "start", "end", "cost_bps", "n_boot", "seed"]);

/**
 * The API's 422 `errors`, on the form's fields. A loc is
 * ["body", <strategy>, <field>] or ["body", <strategy>, "params", <name>];
 * a rule across params (["body", <strategy>, "params"]) or anything else
 * goes under "form".
 */
export function fieldErrorsFromProblem(errors: readonly ValidationErrorItem[]): FieldErrors {
  const result: FieldErrors = {};
  for (const error of errors) {
    const [where, , field, param] = error.loc.map(String);
    const message = error.msg.replace(/^Value error, /, "");
    let key = "form";
    if (where === "body" && field === "params" && param) key = `params.${param}`;
    else if (where === "body" && field && FORM_FIELDS.has(field)) key = field;
    const earlier = result[key];
    result[key] = earlier ? `${earlier} ${message}` : message;
  }
  return result;
}
