/**
 * Every failed API call becomes one `ApiProblem` here, whatever went wrong.
 *
 * The API answers errors with RFC 9457 problem details
 * (`application/problem+json`): `type` names the kind of problem and is
 * what code should compare, `title` summarises it, `detail` explains this
 * occurrence, and 422 validation problems add `errors`. Responses that are
 * not problem details (a proxy's 502 page, an empty 500) and requests that
 * never got a response (API down, network gone) are turned into the same
 * shape, so components only ever deal with `ApiProblem`.
 */
import type { components } from "./schema";

export type Problem = components["schemas"]["Problem"];
export type ValidationProblem = components["schemas"]["ValidationProblem"];
export type ValidationErrorItem = components["schemas"]["ValidationErrorItem"];

/** RFC 9457: "about:blank" means the status code says all there is to say. */
export const ABOUT_BLANK = "about:blank";
/** Client-side only: the request got no HTTP response at all. */
export const NETWORK_ERROR = "network-error";

export class ApiProblem extends Error {
  override readonly name = "ApiProblem";
  /** The HTTP status, or 0 when there was no response. */
  readonly status: number;
  /** The problem type, e.g. "/problems/job-not-found"; compare this, not `title`. */
  readonly type: string;
  readonly title: string;
  readonly detail: string;
  /** Per-field errors, on /problems/validation-error only. */
  readonly errors: readonly ValidationErrorItem[];

  constructor(problem: {
    status: number;
    type: string;
    title: string;
    detail: string;
    errors?: readonly ValidationErrorItem[] | null | undefined;
  }) {
    super(problem.detail || problem.title);
    this.status = problem.status;
    this.type = problem.type;
    this.title = problem.title;
    this.detail = problem.detail;
    this.errors = problem.errors ?? [];
  }

  /** 4xx: the request itself was wrong, so sending it again won't help. */
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }
}

function isProblemBody(body: unknown): body is ValidationProblem {
  if (typeof body !== "object" || body === null) return false;
  const b = body as Record<string, unknown>;
  return (
    typeof b["type"] === "string" &&
    typeof b["title"] === "string" &&
    typeof b["status"] === "number" &&
    typeof b["detail"] === "string"
  );
}

/**
 * The problem for an error response. `body` is what the client parsed:
 * the JSON value when the body was JSON, the raw text otherwise, or
 * undefined when it was empty.
 */
export function problemFromResponse(response: Response, body: unknown): ApiProblem {
  if (isProblemBody(body)) {
    return new ApiProblem({ ...body, status: response.status });
  }
  const text = typeof body === "string" ? body.trim().slice(0, 500) : "";
  return new ApiProblem({
    status: response.status,
    type: ABOUT_BLANK,
    title: response.statusText || `HTTP ${String(response.status)}`,
    detail: text || `The API answered ${String(response.status)} without problem details.`,
  });
}

/** The problem for a request that got no response (fetch rejected). */
export function problemFromNetworkError(cause: unknown): ApiProblem {
  const reason = cause instanceof Error ? cause.message : String(cause);
  return new ApiProblem({
    status: 0,
    type: NETWORK_ERROR,
    title: "No response from the API",
    detail: `The request failed before the API answered: ${reason}`,
  });
}
