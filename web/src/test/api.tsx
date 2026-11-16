/**
 * Test helpers: a fake API at the fetch layer, and a render() that wires a
 * component to it.
 *
 * The component under test uses the real typed client, the real unwrap()
 * and the real TanStack Query; only `fetch` is replaced. A test describes
 * the API as a list of handlers, and every request made is recorded in
 * `calls` so the test can check what was sent (headers, bodies, counts).
 */
import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";

import { createApiClient } from "../api/client";
import { ApiContext } from "../api/context";
import { createQueryClient } from "../queryClient";

export const BASE = "http://api.test";

export interface Call {
  method: string;
  path: string;
  headers: Headers;
  body: unknown;
}

/** Answers a request, or returns undefined to let the next handler try. */
export type Handler = (call: Call) => Response | Promise<Response> | undefined;

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

export const problem = (status: number, slug: string, extra: Record<string, unknown> = {}) =>
  new Response(
    JSON.stringify({
      type: `/problems/${slug}`,
      title: slug.replace(/-/g, " "),
      status,
      detail: `detail of ${slug}`,
      ...extra,
    }),
    { status, headers: { "Content-Type": "application/problem+json" } },
  );

/** A handler for one method and path (exact, or a RegExp on the path). */
export function on(
  method: string,
  path: string | RegExp,
  respond: (call: Call) => Response | Promise<Response>,
): Handler {
  return (call) => {
    const matches = typeof path === "string" ? call.path === path : path.test(call.path);
    return call.method === method && matches ? respond(call) : undefined;
  };
}

/** A fetch that answers with `handlers`, in order; an unhandled request fails the test. */
export function fakeApi(handlers: Handler[]) {
  const calls: Call[] = [];
  const fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = new Request(input, init);
    const text = await request.text();
    const call: Call = {
      method: request.method,
      path: new URL(request.url).pathname,
      headers: request.headers,
      body: text ? (JSON.parse(text) as unknown) : undefined,
    };
    calls.push(call);
    for (const handler of handlers) {
      const response = await handler(call);
      if (response) return response;
    }
    throw new Error(`fakeApi: no handler for ${call.method} ${call.path}`);
  };
  const count = (method: string, path: string | RegExp) =>
    calls.filter(
      (c) =>
        c.method === method && (typeof path === "string" ? c.path === path : path.test(c.path)),
    ).length;
  return { fetch, calls, count };
}

/** Renders `ui` against `fetch`, with a fresh cache and no query retries. */
export function renderWithApi(ui: ReactElement, fetch: typeof globalThis.fetch) {
  const client = createQueryClient();
  client.setDefaultOptions({
    ...client.getDefaultOptions(),
    queries: { ...client.getDefaultOptions().queries, retry: false },
  });
  const api = createApiClient({ baseUrl: BASE, fetch });
  return render(
    <ApiContext value={api}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </ApiContext>,
  );
}

export const sleep = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
