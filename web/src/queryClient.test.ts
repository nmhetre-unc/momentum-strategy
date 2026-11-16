import { ApiProblem } from "./api/problem";
import { shouldRetry } from "./queryClient";

const problem = (status: number) => new ApiProblem({ status, type: "t", title: "t", detail: "d" });

test("never retries a 4xx", () => {
  expect(shouldRetry(0, problem(404))).toBe(false);
  expect(shouldRetry(0, problem(422))).toBe(false);
});

test("retries server errors and network errors twice", () => {
  expect(shouldRetry(0, problem(503))).toBe(true);
  expect(shouldRetry(1, problem(0))).toBe(true);
  expect(shouldRetry(2, problem(503))).toBe(false);
});
