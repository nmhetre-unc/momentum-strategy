import { createApiClient, unwrap } from "./client";
import { ApiProblem, NETWORK_ERROR } from "./problem";

function fakeFetch(respond: (request: Request) => Response | Promise<Response>) {
  const calls: Request[] = [];
  const fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    calls.push(request);
    return Promise.resolve(respond(request));
  };
  return { fetch, calls };
}

const json = (body: unknown, status = 200, contentType = "application/json") =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": contentType } });

test("returns typed data and calls the API under /api", async () => {
  const { fetch, calls } = fakeFetch(() => json({ status: "ok" }));
  const api = createApiClient({ baseUrl: "http://localhost/api", fetch });
  const health = await unwrap(api.GET("/health"));
  expect(health.status).toBe("ok");
  expect(calls[0]?.url).toBe("http://localhost/api/health");
});

test("throws the API's problem details as an ApiProblem", async () => {
  const problem = {
    type: "/problems/job-not-found",
    title: "Job not found",
    status: 404,
    detail: "There is no backtest job 0b6c3f5e-8d1a-4a43-9c55-2f1e7d3b9a10.",
  };
  const { fetch } = fakeFetch(() => json(problem, 404, "application/problem+json"));
  const api = createApiClient({ baseUrl: "http://localhost/api", fetch });
  const call = unwrap(
    api.GET("/backtests/{job_id}", {
      params: { path: { job_id: "0b6c3f5e-8d1a-4a43-9c55-2f1e7d3b9a10" } },
    }),
  );
  await expect(call).rejects.toBeInstanceOf(ApiProblem);
  await expect(call).rejects.toMatchObject({ status: 404, type: "/problems/job-not-found" });
});

test("a failed fetch becomes a network-error ApiProblem", async () => {
  const fetch = (() => Promise.reject(new TypeError("Failed to fetch"))) as typeof globalThis.fetch;
  const api = createApiClient({ baseUrl: "http://localhost/api", fetch });
  await expect(unwrap(api.GET("/health"))).rejects.toMatchObject({
    status: 0,
    type: NETWORK_ERROR,
  });
});
