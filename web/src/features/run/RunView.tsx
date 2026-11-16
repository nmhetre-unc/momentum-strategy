/**
 * Run a backtest: pick a strategy, fill in its generated parameter form,
 * submit, and watch the job until it succeeds or fails.
 *
 * web/README.md ("How the React code works") walks through this file.
 */
import { useId, useState, type SubmitEvent } from "react";

import {
  POLL_INTERVAL_MS,
  isTransient,
  newIdempotencyKey,
  useBacktestJob,
  useStrategies,
  useSubmitBacktest,
  type Submission,
} from "../../api/queries";
import { isTerminal, type BacktestJob, type StrategyInfo } from "../../api/types";
import { Link } from "../../components/Link";
import { Empty, Loading, ProblemView } from "../../components/status";
import { recordRecentRun } from "../compare/recentRuns";
import {
  DEFAULT_VALUES,
  LIMITS,
  defaultParams,
  fieldErrorsFromProblem,
  paramFields,
  validate,
  type FieldErrors,
  type FormValues,
  type ParamField,
} from "./form";

interface Timing {
  /** How often to poll the job; tests shorten it. */
  pollIntervalMs?: number;
  /** The delay before the first automatic retry of a failed submit. */
  retryDelayMs?: number;
}

export function RunView({ pollIntervalMs = POLL_INTERVAL_MS, retryDelayMs = 1000 }: Timing) {
  // useStrategies (useQuery): the strategies and their parameter schemas
  // come from the API, cached and shared with any other view that needs them.
  const strategies = useStrategies();

  let content;
  if (strategies.isPending) {
    content = <Loading label="Loading strategies…" />;
  } else if (strategies.isError) {
    content = (
      <ProblemView
        context="Could not load the strategies"
        problem={strategies.error}
        onRetry={() => void strategies.refetch()}
      />
    );
  } else if (strategies.data.strategies.length === 0) {
    content = <Empty title="No strategies">The API offers no strategies to run.</Empty>;
  } else {
    content = (
      <RunForm
        strategies={strategies.data.strategies}
        pollIntervalMs={pollIntervalMs}
        retryDelayMs={retryDelayMs}
      />
    );
  }

  return (
    <section aria-labelledby="run-heading">
      <h2 id="run-heading">Run a backtest</h2>
      {content}
    </section>
  );
}

function RunForm({
  strategies,
  pollIntervalMs,
  retryDelayMs,
}: {
  strategies: readonly StrategyInfo[];
  pollIntervalMs: number;
  retryDelayMs: number;
}) {
  // useState: which strategy is picked. Everything else about the strategy
  // (its schema, version, defaults) is looked up from it on each render.
  const [selected, setSelected] = useState(strategies[0]?.name ?? "");
  const strategy = strategies.find((s) => s.name === selected) ?? strategies[0];

  // useState: the inputs shared by all strategies, as typed (strings).
  const [common, setCommon] = useState(DEFAULT_VALUES);
  // useState: each strategy's parameter inputs, kept per strategy so that
  // switching away and back doesn't lose what was typed. A strategy not
  // yet edited shows its defaults.
  const [paramsByStrategy, setParamsByStrategy] = useState<Record<string, Record<string, string>>>(
    {},
  );
  // useState: show errors only once the user has tried to submit, not
  // while they are still filling the form in for the first time.
  const [showErrors, setShowErrors] = useState(false);
  // useState: the submission being sent, kept so "Try again" can resend it
  // with the same Idempotency-Key.
  const [submission, setSubmission] = useState<Submission | null>(null);

  // useSubmitBacktest (useMutation): sends POST /backtests and tracks it
  // (pending, error, the accepted job), retrying transient failures.
  const submit = useSubmitBacktest(retryDelayMs);
  // useBacktestJob (useQuery): polls the accepted job until it is final.
  // It waits (disabled) while there is no job id yet.
  const job = useBacktestJob(submit.data?.id, pollIntervalMs);

  if (!strategy) return null;
  const params = paramsByStrategy[strategy.name] ?? defaultParams(strategy);
  const values: FormValues = { ...common, params };
  const checked = validate(strategy, values);
  const { fields, unsupported } = paramFields(strategy.params_schema);

  const serverErrors: FieldErrors =
    submit.error?.type === "/problems/validation-error"
      ? fieldErrorsFromProblem(submit.error.errors)
      : {};
  const errors: FieldErrors = showErrors ? { ...serverErrors, ...checked.errors } : {};

  // Editing after a failed submit makes the next submit a new submission.
  const edited = () => {
    if (submit.isError) submit.reset();
  };
  const setField = (name: keyof typeof DEFAULT_VALUES, value: string) => {
    edited();
    setCommon((current) => ({ ...current, [name]: value }));
  };
  const setParam = (name: string, value: string) => {
    edited();
    setParamsByStrategy((current) => ({
      ...current,
      [strategy.name]: { ...params, [name]: value },
    }));
  };

  const send = (next: Submission) => {
    submit.mutate(next, {
      onSuccess: (accepted) => {
        recordRecentRun({
          id: accepted.id,
          label: `${next.body.strategy} · ${next.body.ticker} ${next.body.start} to ${next.body.end}`,
        });
      },
    });
  };

  const onSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    setShowErrors(true);
    if (checked.errors) return;
    // A new submission: a new key. Only "Try again" reuses one.
    const next = { key: newIdempotencyKey(), body: checked.body };
    setSubmission(next);
    send(next);
  };

  return (
    <>
      <form className="run-form" onSubmit={onSubmit} noValidate aria-label="Backtest">
        <StrategyPicker
          strategies={strategies}
          strategy={strategy}
          onChange={(name) => {
            edited();
            setSelected(name);
          }}
        />

        <fieldset>
          <legend>Parameters</legend>
          {fields.length === 0 && <p className="hint">This strategy has no parameters.</p>}
          {fields.map((field) => (
            <NumberField
              key={`${strategy.name}.${field.name}`}
              field={field}
              value={params[field.name] ?? ""}
              error={errors[`params.${field.name}`]}
              onChange={(value) => {
                setParam(field.name, value);
              }}
            />
          ))}
          {unsupported.length > 0 && (
            <p className="hint">
              This form can&apos;t edit {unsupported.join(", ")}; the API&apos;s defaults are used.
            </p>
          )}
        </fieldset>

        <fieldset>
          <legend>Data and costs</legend>
          <TextField
            label="Ticker"
            value={common.ticker}
            error={errors["ticker"]}
            onChange={(v) => {
              setField("ticker", v);
            }}
          />
          <TextField
            label="Start"
            type="date"
            hint="First day included."
            value={common.start}
            error={errors["start"]}
            onChange={(v) => {
              setField("start", v);
            }}
          />
          <TextField
            label="End"
            type="date"
            hint="Not included."
            value={common.end}
            error={errors["end"]}
            onChange={(v) => {
              setField("end", v);
            }}
          />
          {(["cost_bps", "n_boot", "seed"] as const).map((name) => (
            <NumberField
              key={name}
              field={REQUEST_FIELDS[name]}
              value={common[name]}
              error={errors[name]}
              onChange={(v) => {
                setField(name, v);
              }}
            />
          ))}
        </fieldset>

        {errors["form"] && (
          <p className="field-error" role="alert">
            {errors["form"]}
          </p>
        )}
        <button type="submit" disabled={submit.isPending}>
          {submit.isPending ? "Submitting…" : "Run backtest"}
        </button>
      </form>

      <SubmitState
        submit={submit}
        onRetry={
          submission
            ? () => {
                send(submission);
              }
            : undefined
        }
      />
      {submit.data && (
        <JobProgress
          id={submit.data.id}
          job={job.data}
          error={job.error}
          onRetry={() => void job.refetch()}
          pollIntervalMs={pollIntervalMs}
        />
      )}
    </>
  );
}

const REQUEST_FIELDS: Record<"cost_bps" | "n_boot" | "seed", ParamField> = {
  cost_bps: {
    name: "cost_bps",
    kind: "number",
    title: "Cost (bps)",
    description: "One-way cost per unit of position change, in basis points.",
    minimum: LIMITS.cost_bps.min,
    maximum: LIMITS.cost_bps.max,
    default: undefined,
  },
  n_boot: {
    name: "n_boot",
    kind: "integer",
    title: "Bootstrap replicates",
    description: "For the Sharpe ratio's interval; 0 skips it.",
    minimum: LIMITS.n_boot.min,
    maximum: LIMITS.n_boot.max,
    default: undefined,
  },
  seed: {
    name: "seed",
    kind: "integer",
    title: "Seed",
    description: "Bootstrap seed.",
    minimum: LIMITS.seed.min,
    maximum: undefined,
    default: undefined,
  },
};

function StrategyPicker({
  strategies,
  strategy,
  onChange,
}: {
  strategies: readonly StrategyInfo[];
  strategy: StrategyInfo;
  onChange: (name: string) => void;
}) {
  // useId: a unique id to tie the <label> and the description to the
  // <select>, unique even if this component appears twice on a page.
  const id = useId();
  return (
    <div className="field strategy-picker">
      <label htmlFor={id}>Strategy</label>
      <select
        id={id}
        value={strategy.name}
        aria-describedby={`${id}-about`}
        onChange={(e) => {
          onChange(e.target.value);
        }}
      >
        {strategies.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name} (v{s.version})
          </option>
        ))}
      </select>
      <p id={`${id}-about`} className="hint">
        <span className="badge">version {strategy.version}</span>{" "}
        <span className="badge">{strategy.family}</span> {strategy.description}
      </p>
    </div>
  );
}

function NumberField({
  field,
  value,
  error,
  onChange,
}: {
  field: ParamField;
  value: string;
  error: string | undefined;
  onChange: (value: string) => void;
}) {
  const range =
    field.minimum !== undefined && field.maximum !== undefined
      ? ` ${String(field.minimum)} to ${String(field.maximum)}.`
      : field.minimum !== undefined
        ? ` At least ${String(field.minimum)}.`
        : "";
  return (
    <TextField
      label={field.title}
      type="number"
      step={field.kind === "integer" ? "1" : "any"}
      hint={`${field.description}${range}`}
      value={value}
      error={error}
      onChange={onChange}
    />
  );
}

function TextField({
  label,
  value,
  onChange,
  error,
  hint,
  type = "text",
  step,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  error: string | undefined;
  hint?: string;
  type?: string;
  step?: string;
}) {
  // useId: as in StrategyPicker, ties the label, hint and error to the input.
  const id = useId();
  const described = [hint ? `${id}-hint` : "", error ? `${id}-error` : ""].filter(Boolean);
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type={type}
        step={step}
        value={value}
        aria-invalid={error ? true : undefined}
        aria-describedby={described.length ? described.join(" ") : undefined}
        onChange={(e) => {
          onChange(e.target.value);
        }}
      />
      {hint && (
        <p id={`${id}-hint`} className="hint">
          {hint}
        </p>
      )}
      {error && (
        <p id={`${id}-error`} className="field-error">
          {error}
        </p>
      )}
    </div>
  );
}

function SubmitState({
  submit,
  onRetry,
}: {
  submit: ReturnType<typeof useSubmitBacktest>;
  onRetry: (() => void) | undefined;
}) {
  if (submit.isPending) {
    return (
      <Loading
        label={
          submit.failureCount > 0
            ? `Submitting… (retry ${String(submit.failureCount)}, same request)`
            : "Submitting…"
        }
      />
    );
  }
  if (!submit.isError) return null;
  // A 4xx (validation, no price data) would fail again: fix the form
  // instead. A transient failure can be retried with the same key.
  return (
    <ProblemView
      context="The backtest was not submitted"
      problem={submit.error}
      onRetry={isTransient(submit.error) ? onRetry : undefined}
    />
  );
}

function JobProgress({
  id,
  job,
  error,
  onRetry,
  pollIntervalMs,
}: {
  id: string;
  job: BacktestJob | undefined;
  error: ReturnType<typeof useBacktestJob>["error"];
  onRetry: () => void;
  pollIntervalMs: number;
}) {
  return (
    <section className="job" aria-label="Job progress" aria-live="polite">
      <h3>
        Job <code>{id}</code>
      </h3>
      {error && <ProblemView context="Lost track of the job" problem={error} onRetry={onRetry} />}
      {!error && !job && <Loading label="Waiting for the job…" />}
      {job && (
        <>
          <p>
            Status: <strong className={`job-status ${job.status}`}>{job.status}</strong> · attempt{" "}
            {job.attempts} of {job.max_attempts} · strategy version {job.strategy_version}
          </p>
          {!isTerminal(job.status) && (
            <p className="hint">
              Checking every {String(pollIntervalMs / 1000)} s until it succeeds or fails.
              {job.last_error && ` The last attempt failed and will be retried: ${job.last_error}`}
            </p>
          )}
          {job.status === "succeeded" && (
            <p>
              <Link to={`/backtests/${id}`}>See the results</Link>
            </p>
          )}
          {job.status === "failed" && (
            <p className="field-error" role="alert">
              The backtest failed: {job.last_error ?? "no error was recorded."}
            </p>
          )}
        </>
      )}
    </section>
  );
}
