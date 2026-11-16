import { screen, within } from "@testing-library/react";

import { fakeApi, json, on, problem, renderWithApi } from "../../test/api";
import { model, prediction } from "../../test/fixtures";
import { ModelView } from "./ModelView";

function renderModel(handlers: Parameters<typeof fakeApi>[0]) {
  const api = fakeApi(handlers);
  const view = renderWithApi(<ModelView />, api.fetch);
  return { ...api, ...view };
}

test("the prediction and the baseline it fails to beat, equally prominent", async () => {
  const { calls } = renderModel([
    on("GET", "/models", () => json({ models: [model({ version: 2 }), model()] })),
    on("POST", "/predict", () => json(prediction())),
  ]);
  expect(screen.getByRole("status")).toHaveTextContent("Loading the models");

  const predictionPanel = await screen.findByRole("region", { name: "Latest prediction" });
  const baselinePanel = screen.getByRole("region", { name: "Against its baseline" });
  const predicted = await within(predictionPanel).findByText("up");
  const verdict = within(baselinePanel).getByText(/Does not beat always_up/);
  const baselineAccuracy = within(baselinePanel).getByText("61.3%");

  // The same headline style for the prediction, the verdict and the
  // accuracy the model fails to beat.
  for (const element of [predicted, verdict, baselineAccuracy]) {
    expect(element).toHaveClass("headline-big");
  }
  expect(within(baselinePanel).getByText("58.1%")).toHaveClass("headline-big");
  expect(baselinePanel).toHaveTextContent("beats: false");
  expect(baselinePanel).toHaveClass("does-not-beat");
  expect(screen.getByText("model version 3")).toBeInTheDocument();

  // The latest version (the last listed), asked for explicitly; one prediction.
  const predicts = calls.filter((c) => c.method === "POST");
  expect(predicts).toHaveLength(1);
  expect(predicts[0]?.body).toEqual({ version: 3 });
});

test("a model that beats its baseline says so", async () => {
  const beating = { ...model().baseline, beats: true, difference: 0.05 };
  renderModel([
    on("GET", "/models", () => json({ models: [model({ baseline: beating })] })),
    on("POST", "/predict", () => json(prediction({ baseline: beating }))),
  ]);
  expect(await screen.findByText(/Beats always_up/)).toHaveClass("headline-big");
  expect(screen.getByText("true")).toBeInTheDocument();
});

test("no registered model is an empty state, and nothing is predicted", async () => {
  const { calls } = renderModel([on("GET", "/models", () => json({ models: [] }))]);
  expect(await screen.findByText("No model is registered yet")).toBeInTheDocument();
  expect(calls.filter((c) => c.method === "POST")).toHaveLength(0);
});

test("a failed prediction is shown next to the model, which still shows its baseline", async () => {
  renderModel([
    on("GET", "/models", () => json({ models: [model()] })),
    on("POST", "/predict", () => problem(500, "model-integrity-error")),
  ]);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Could not get a prediction");
  expect(alert).toHaveTextContent("(500)");
  expect(screen.getByText(/Does not beat always_up/)).toBeInTheDocument();
});

test("a problem loading the models is shown", async () => {
  renderModel([on("GET", "/models", () => problem(503, "database-unavailable"))]);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the models");
});
