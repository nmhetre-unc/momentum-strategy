/**
 * The client-side validation against the request contract shared with the
 * API: contracts/backtest_requests.json at the repo root lists POST
 * /backtests bodies and whether the API accepts them (the API is checked
 * against the same file by tests/service/test_request_contract.py).
 *
 * Each case's body becomes what the form would hold (every value as the
 * string a user would type), with the strategy's params schema from
 * openapi.json, which the Python test checks is the schema GET /strategies
 * serves.
 */
import contract from "../../../../contracts/backtest_requests.json";
import openapi from "../../../openapi.json";
import type { StrategyInfo } from "../../api/types";
import { validate, type FormValues } from "./form";

interface Case {
  name: string;
  valid: boolean;
  body: Record<string, unknown> & { strategy: string; params: Record<string, unknown> };
}

const schemas = openapi.components.schemas as Record<string, Record<string, unknown>>;

function strategyInfo(name: string): StrategyInfo {
  const schemaName = name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");
  const params_schema = schemas[`${schemaName}Params`];
  if (!params_schema) throw new Error(`openapi.json has no ${schemaName}Params`);
  return {
    name,
    family: "",
    description: "",
    params_schema,
    defaults: {},
    version: 1,
    schema_hash: "",
  };
}

const FIELDS = ["ticker", "start", "end", "cost_bps", "n_boot", "seed"] as const;

/** The form's values for a body: every field and parameter, as typed text. */
function formValues(strategy: StrategyInfo, body: Case["body"]): FormValues {
  const text = (value: unknown, what: string): string => {
    if (typeof value === "string") return value;
    if (typeof value === "number") return String(value);
    throw new Error(`case must give ${what} as a string or number`);
  };
  const properties = strategy.params_schema["properties"] as Record<string, unknown>;
  const params = Object.fromEntries(
    Object.keys(properties).map((name) => [name, text(body.params[name], `params.${name}`)]),
  );
  const values = Object.fromEntries(FIELDS.map((f) => [f, text(body[f], f)]));
  return { ...(values as Omit<FormValues, "params">), params };
}

const cases = (contract as { cases: Case[] }).cases;

test("the contract has cases", () => {
  expect(cases.length).toBeGreaterThan(50);
});

test.each(cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
  const strategy = strategyInfo(c.body.strategy);
  const result = validate(strategy, formValues(strategy, c.body));
  expect(result.errors === undefined, JSON.stringify(result.errors)).toBe(c.valid);
});
