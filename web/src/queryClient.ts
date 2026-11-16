import { QueryClient } from "@tanstack/react-query";

import { ApiProblem } from "./api/problem";

// Every query and mutation error is an ApiProblem (src/api/client.ts,
// unwrap), so components get `error.type`, `error.detail` and so on,
// typed, without checking `instanceof` first.
declare module "@tanstack/react-query" {
  interface Register {
    defaultError: ApiProblem;
  }
}

/** Retry a failed query at most twice, and never a 4xx: the same request would fail again. */
export function shouldRetry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiProblem && error.isClientError) return false;
  return failureCount < 2;
}

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: shouldRetry, staleTime: 30_000 },
      mutations: { retry: false },
    },
  });
}
