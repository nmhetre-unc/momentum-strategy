import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "./App";
import { matchRoute, parseIds } from "./router";
import { fakeApi, json, on, renderWithApi } from "./test/api";
import { STRATEGIES, model, prediction } from "./test/fixtures";

const api = () =>
  fakeApi([
    on("GET", "/strategies", () => json(STRATEGIES)),
    on("GET", "/models", () => json({ models: [model()] })),
    on("POST", "/predict", () => json(prediction())),
  ]);

test("starts on the Run page and navigates without reloading", async () => {
  const user = userEvent.setup();
  renderWithApi(<App />, api().fetch);
  expect(screen.getByRole("heading", { name: "qbt dashboard" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Run a backtest" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Run" })).toHaveAttribute("aria-current", "page");

  await user.click(screen.getByRole("link", { name: "Model" }));
  expect(window.location.pathname).toBe("/model");
  expect(await screen.findByRole("heading", { name: "Model" })).toBeInTheDocument();

  window.history.back();
  expect(await screen.findByRole("heading", { name: "Run a backtest" })).toBeInTheDocument();
});

test("an unknown path is a not-found page", () => {
  window.history.replaceState(null, "", "/nowhere");
  renderWithApi(<App />, api().fetch);
  expect(screen.getByText("Page not found")).toBeInTheDocument();
});

test("routes", () => {
  expect(matchRoute("/")).toEqual({ name: "run" });
  expect(matchRoute("/backtests/abc")).toEqual({ name: "results", id: "abc" });
  expect(matchRoute("/compare?ids=a,b,a")).toEqual({ name: "compare", ids: ["a", "b"] });
  expect(matchRoute("/model/")).toEqual({ name: "model" });
  expect(matchRoute("/backtests/a/b")).toEqual({ name: "not-found", path: "/backtests/a/b" });
  expect(parseIds("?ids= a ,,b")).toEqual(["a", "b"]);
});
