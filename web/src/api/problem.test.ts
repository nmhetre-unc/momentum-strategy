import {
  ABOUT_BLANK,
  ApiProblem,
  NETWORK_ERROR,
  problemFromNetworkError,
  problemFromResponse,
} from "./problem";

const response = (status: number, statusText = "") => new Response(null, { status, statusText });

test("keeps the API's problem details", () => {
  const p = problemFromResponse(response(404), {
    type: "/problems/job-not-found",
    title: "Job not found",
    status: 404,
    detail: "There is no backtest job 0b6c3f5e.",
  });
  expect(p).toBeInstanceOf(ApiProblem);
  expect(p).toMatchObject({ status: 404, type: "/problems/job-not-found", title: "Job not found" });
  expect(p.message).toBe("There is no backtest job 0b6c3f5e.");
  expect(p.isClientError).toBe(true);
  expect(p.errors).toEqual([]);
});

test("keeps validation errors", () => {
  const errors = [
    { loc: ["body", "start"], msg: "Input should be a valid date", type: "date_parsing" },
  ];
  const p = problemFromResponse(response(422), {
    type: "/problems/validation-error",
    title: "Invalid request",
    status: 422,
    detail: "1 invalid value",
    errors,
  });
  expect(p.errors).toEqual(errors);
});

test("a response without problem details becomes about:blank", () => {
  const p = problemFromResponse(response(502, "Bad Gateway"), "<html>upstream down</html>");
  expect(p).toMatchObject({ status: 502, type: ABOUT_BLANK, title: "Bad Gateway" });
  expect(p.detail).toBe("<html>upstream down</html>");
  expect(p.isClientError).toBe(false);
});

test("an empty error body still says what happened", () => {
  const p = problemFromResponse(response(500), undefined);
  expect(p).toMatchObject({ status: 500, type: ABOUT_BLANK, title: "HTTP 500" });
  expect(p.detail).toContain("500");
});

test("JSON that isn't a problem is not trusted as one", () => {
  const p = problemFromResponse(response(400), { message: "nope" });
  expect(p.type).toBe(ABOUT_BLANK);
});

test("no response at all is a network-error problem with status 0", () => {
  const p = problemFromNetworkError(new TypeError("Failed to fetch"));
  expect(p).toMatchObject({ status: 0, type: NETWORK_ERROR });
  expect(p.detail).toContain("Failed to fetch");
});
