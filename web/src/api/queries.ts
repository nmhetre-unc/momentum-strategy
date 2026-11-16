/**
 * Every API read and write the views make, as TanStack Query hooks.
 *
 * A query hook gives a component `{ data, error, isPending, ... }` and keeps
 * the cache: two components asking for the same job share one request, and
 * a view that comes back into sight shows the cached copy at once. Keys say
 * what a piece of data is, so it can be shared and refetched: ["backtests",
 * id] is one job, ["backtests", id, "series"] its series.
 */
import { useMutation, useQueries, useQuery } from "@tanstack/react-query";

import { ApiProblem } from "./problem";
import { unwrap } from "./client";
import { useApi } from "./context";
import { isTerminal, type BacktestJob, type BacktestRequest } from "./types";

export const POLL_INTERVAL_MS = 1000;

export const queryKeys = {
  strategies: ["strategies"] as const,
  job: (id: string) => ["backtests", id] as const,
  series: (id: string) => ["backtests", id, "series"] as const,
  models: ["models"] as const,
};

export function useStrategies() {
  const api = useApi();
  // useQuery: the strategy list is server state; the cache shares it between
  // views and refetches it when it is older than staleTime.
  return useQuery({
    queryKey: queryKeys.strategies,
    queryFn: ({ signal }) => unwrap(api.GET("/strategies", { signal })),
  });
}

/**
 * The next poll delay for a job, or false to stop: stop once the job is
 * final, and stop on an error (the view shows it, with a button to retry).
 */
export function nextPoll(
  job: BacktestJob | undefined,
  failed: boolean,
  intervalMs: number,
): number | false {
  if (failed) return false;
  if (job && isTerminal(job.status)) return false;
  return intervalMs;
}

export function jobQueryOptions(api: ReturnType<typeof useApi>, id: string, intervalMs: number) {
  return {
    queryKey: queryKeys.job(id),
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      unwrap(api.GET("/backtests/{job_id}", { params: { path: { job_id: id } }, signal })),
    // Polling is per component using the query: when the last one unmounts,
    // the interval stops with it, so leaving the page stops polling.
    refetchInterval: (query: { state: { data: BacktestJob | undefined; status: string } }) =>
      nextPoll(query.state.data, query.state.status === "error", intervalMs),
  };
}

/** One job, polled every `intervalMs` until it succeeds or fails. */
export function useBacktestJob(id: string | undefined, intervalMs = POLL_INTERVAL_MS) {
  const api = useApi();
  // useQuery with refetchInterval: TanStack Query owns the timer, starts it
  // when the component mounts, and clears it on unmount or when nextPoll()
  // returns false. No hand-written setInterval to forget to clear.
  return useQuery({
    ...jobQueryOptions(api, id ?? "", intervalMs),
    enabled: id !== undefined,
  });
}

/** Several jobs at once (the Compare view), each polled like useBacktestJob. */
export function useBacktestJobs(ids: readonly string[], intervalMs = POLL_INTERVAL_MS) {
  const api = useApi();
  // useQueries: one query per id, a number that changes with the selection,
  // which a fixed number of useQuery calls can't do (hooks can't be called
  // in a loop whose length changes).
  return useQueries({ queries: ids.map((id) => jobQueryOptions(api, id, intervalMs)) });
}

function seriesQueryOptions(api: ReturnType<typeof useApi>, id: string) {
  return {
    queryKey: queryKeys.series(id),
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      unwrap(api.GET("/backtests/{job_id}/series", { params: { path: { job_id: id } }, signal })),
    // A succeeded job's series never changes, so it never needs refetching.
    staleTime: Infinity,
  };
}

/** A succeeded job's daily series. `enabled` is false until the job has succeeded. */
export function useBacktestSeries(id: string, enabled: boolean) {
  const api = useApi();
  // useQuery with `enabled`: the request waits until the job has succeeded;
  // before that the API would answer 409.
  return useQuery({ ...seriesQueryOptions(api, id), enabled });
}

/** The series of several succeeded jobs. */
export function useBacktestSeriesList(ids: readonly string[]) {
  const api = useApi();
  // useQueries: as for useBacktestJobs, one query per selected job.
  return useQueries({ queries: ids.map((id) => seriesQueryOptions(api, id)) });
}

export function useModels() {
  const api = useApi();
  // useQuery: the registered models are server state, like the strategies.
  return useQuery({
    queryKey: queryKeys.models,
    queryFn: ({ signal }) => unwrap(api.GET("/models", { signal })),
  });
}

/**
 * POST /predict for one model version, sent only when the user asks.
 *
 * The API logs every prediction it serves, so a prediction is a write:
 * loading the Model page must not create one. Each call to `mutate` is one
 * POST and one logged prediction; nothing is retried (a 4xx or the 500
 * integrity error would fail again, and a retry after a lost response
 * would log a second prediction).
 */
export function usePredict() {
  const api = useApi();
  // useMutation: runs only when the user presses "Predict", and tracks that
  // request's pending, error and result state.
  return useMutation({
    mutationFn: (version: number) => unwrap(api.POST("/predict", { body: { version } })),
  });
}

/** One submission: a body and the Idempotency-Key that goes with it. */
export interface Submission {
  key: string;
  body: BacktestRequest;
}

/**
 * A new Idempotency-Key: 128 random bits, hex. crypto.getRandomValues
 * rather than crypto.randomUUID, which browsers only provide on HTTPS or
 * localhost.
 */
export function newIdempotencyKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** No response, or a 5xx: the same request may well succeed if sent again. */
export function isTransient(error: ApiProblem): boolean {
  return error.status === 0 || error.status >= 500;
}

export const SUBMIT_RETRIES = 2;

/**
 * POST /backtests. A transient failure is retried automatically, up to
 * SUBMIT_RETRIES times, with the same body and key: the API then returns
 * the job the first attempt created, if it did, instead of creating a
 * second. That is what the Idempotency-Key is for.
 */
export function useSubmitBacktest(retryDelayMs = 1000) {
  const api = useApi();
  // useMutation: a write the user triggers, with its own pending and error
  // state. Unlike a query it never runs by itself.
  return useMutation({
    mutationFn: ({ key, body }: Submission) =>
      unwrap(
        api.POST("/backtests", {
          body,
          // Quoted: the header is a Structured Field String (the API also accepts it bare).
          params: { header: { "Idempotency-Key": `"${key}"` } },
        }),
      ),
    retry: (failures, error) => failures < SUBMIT_RETRIES && isTransient(error),
    retryDelay: (attempt) => retryDelayMs * (attempt + 1),
  });
}
