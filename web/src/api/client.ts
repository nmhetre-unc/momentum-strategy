/**
 * The typed API client.
 *
 * `schema.d.ts` is generated from the API's OpenAPI schema (npm run
 * api:generate); never edit it by hand. openapi-fetch uses those types, so
 * a path, parameter or body the API doesn't accept is a compile error, and
 * every response is typed from the schema.
 *
 * openapi-fetch returns `{ data, error, response }` instead of throwing.
 * `unwrap` is where that becomes "data, or throw an ApiProblem": the single
 * place API errors are handled (see ./problem.ts). Queries and mutations
 * call it, so TanStack Query sees a thrown ApiProblem on any failure.
 */
import createClient from "openapi-fetch";

import { problemFromNetworkError, problemFromResponse } from "./problem";
import type { paths } from "./schema";

/** The browser calls /api/...; Vite's dev server forwards it to the API (vite.config.ts). */
export const API_BASE_URL = "/api";

export function createApiClient(options: { baseUrl?: string; fetch?: typeof fetch } = {}) {
  return createClient<paths>({
    baseUrl: options.baseUrl ?? API_BASE_URL,
    ...(options.fetch ? { fetch: options.fetch } : {}),
  });
}

export type ApiClient = ReturnType<typeof createApiClient>;

export const api: ApiClient = createApiClient();

type FetchResult<T> =
  | { data: T; error?: never; response: Response }
  | { data?: never; error: unknown; response: Response };

/**
 * The response data, or a thrown ApiProblem.
 *
 *     const strategies = await unwrap(api.GET("/strategies"));
 */
export async function unwrap<T>(request: Promise<FetchResult<T>>): Promise<T> {
  let result: FetchResult<T>;
  try {
    result = await request;
  } catch (cause) {
    throw problemFromNetworkError(cause);
  }
  if (!result.response.ok) {
    throw problemFromResponse(result.response, result.error);
  }
  return result.data as T;
}
