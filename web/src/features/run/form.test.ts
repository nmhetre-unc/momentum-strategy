import { ML, SMA } from "../../test/fixtures";
import {
  DEFAULT_VALUES,
  defaultParams,
  fieldErrorsFromProblem,
  paramFields,
  validate,
} from "./form";

test("fields come from the schema; unsupported properties are reported, not dropped silently", () => {
  const { fields, unsupported } = paramFields({
    properties: {
      window: { type: "integer", minimum: 1, maximum: 9, default: 3, title: "Window" },
      weights: { type: "array" },
    },
  });
  expect(fields).toEqual([
    {
      name: "window",
      kind: "integer",
      title: "Window",
      description: "",
      minimum: 1,
      maximum: 9,
      default: 3,
    },
  ]);
  expect(unsupported).toEqual(["weights"]);
});

test("defaults become input strings", () => {
  expect(defaultParams(SMA)).toEqual({ short_window: "50", long_window: "200" });
  expect(defaultParams(ML)).toEqual({ train_frac: "0.7" });
});

test("a valid form is the request body, ticker normalized", () => {
  const result = validate(ML, {
    ...DEFAULT_VALUES,
    ticker: " brk-b ",
    params: { train_frac: "0.6" },
  });
  expect(result.errors).toBeUndefined();
  expect(result.body).toEqual({
    strategy: "ml_direction",
    params: { train_frac: 0.6 },
    ticker: "BRK-B",
    start: "2015-01-01",
    end: "2025-01-01",
    cost_bps: 5,
    n_boot: 2000,
    seed: 0,
  });
});

test("the request-level bounds match the API's", () => {
  const values = {
    ...DEFAULT_VALUES,
    cost_bps: "100.5",
    n_boot: "5001",
    seed: "-1",
    params: defaultParams(SMA),
  };
  expect(validate(SMA, values).errors).toEqual({
    cost_bps: "Must be at most 100.",
    n_boot: "Must be at most 5000.",
    seed: "Must be at least 0.",
  });
  expect(
    validate(SMA, { ...values, cost_bps: "0", n_boot: "0", seed: "0" }).errors,
  ).toBeUndefined();
});

test("the API's 422 locations map onto the form's fields", () => {
  expect(
    fieldErrorsFromProblem([
      { loc: ["body", "sma_crossover", "params", "long_window"], msg: "too big", type: "x" },
      { loc: ["body", "sma_crossover", "params"], msg: "Value error, short < long", type: "x" },
      { loc: ["body", "sma_crossover", "end"], msg: "bad", type: "x" },
      { loc: ["header", "Idempotency-Key"], msg: "odd", type: "x" },
    ]),
  ).toEqual({
    "params.long_window": "too big",
    end: "bad",
    form: "short < long odd",
  });
});
