import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { fakeApi, json, on, problem, renderWithApi, sleep } from "../../test/api";
import { model, prediction } from "../../test/fixtures";
import { ModelView } from "./ModelView";

function renderModel(handlers: Parameters<typeof fakeApi>[0]) {
  const api = fakeApi(handlers);
  const view = renderWithApi(<ModelView />, api.fetch);
  return { ...api, ...view, user: userEvent.setup() };
}

const predicts = (calls: { method: string }[]) => calls.filter((c) => c.method === "POST");

test("opening the page writes nothing: registry info and baseline verdict from GET /models", async () => {
  const { calls } = renderModel([
    on("GET", "/models", () => json({ models: [model({ version: 2 }), model()] })),
    on("POST", "/predict", () => json(prediction())),
  ]);
  expect(screen.getByRole("status")).toHaveTextContent("Loading the models");

  const baselinePanel = await screen.findByRole("region", { name: "Against its baseline" });
  expect(within(baselinePanel).getByText(/Does not beat always_up/)).toHaveClass("headline-big");
  expect(within(baselinePanel).getByText("61.3%")).toHaveClass("headline-big");
  expect(within(baselinePanel).getByText("58.1%")).toHaveClass("headline-big");
  expect(baselinePanel).toHaveTextContent("beats: false");
  expect(baselinePanel).toHaveClass("does-not-beat");
  expect(screen.getByText("model version 3")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Predict" })).toBeInTheDocument();

  await sleep(20);
  expect(predicts(calls)).toHaveLength(0);
});

test("each press of Predict is one POST for the latest version", async () => {
  const { calls, user } = renderModel([
    on("GET", "/models", () => json({ models: [model({ version: 2 }), model()] })),
    on("POST", "/predict", () => json(prediction())),
  ]);
  await user.click(await screen.findByRole("button", { name: "Predict" }));

  const panel = screen.getByRole("region", { name: "Prediction" });
  // The prediction and the verdict share the headline style: neither outranks the other.
  expect(await within(panel).findByText("up")).toHaveClass("headline-big");
  expect(predicts(calls)).toHaveLength(1);
  expect(predicts(calls)[0]).toMatchObject({ body: { version: 3 } });

  await user.click(screen.getByRole("button", { name: "Predict again" }));
  await waitFor(() => {
    expect(predicts(calls)).toHaveLength(2);
  });
  await sleep(20);
  expect(predicts(calls)).toHaveLength(2);
});

test("a model that beats its baseline says so", async () => {
  const beating = { ...model().baseline, beats: true, difference: 0.05 };
  renderModel([on("GET", "/models", () => json({ models: [model({ baseline: beating })] }))]);
  expect(await screen.findByText(/Beats always_up/)).toHaveClass("headline-big");
  expect(screen.getByText("true")).toBeInTheDocument();
});

test("no registered model is an empty state with no Predict button", async () => {
  const { calls } = renderModel([on("GET", "/models", () => json({ models: [] }))]);
  expect(await screen.findByText("No model is registered yet")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Predict" })).not.toBeInTheDocument();
  expect(predicts(calls)).toHaveLength(0);
});

test("a failed prediction is shown, once, next to the baseline verdict", async () => {
  const { calls, user } = renderModel([
    on("GET", "/models", () => json({ models: [model()] })),
    on("POST", "/predict", () => problem(500, "model-integrity-error")),
  ]);
  await user.click(await screen.findByRole("button", { name: "Predict" }));
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Could not get a prediction");
  expect(alert).toHaveTextContent("(500)");
  expect(screen.getByText(/Does not beat always_up/)).toBeInTheDocument();
  await sleep(20);
  expect(predicts(calls)).toHaveLength(1);
});

test("a problem loading the models is shown", async () => {
  renderModel([on("GET", "/models", () => problem(503, "database-unavailable"))]);
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the models");
});
