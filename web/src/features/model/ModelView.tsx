/**
 * The served model: its latest prediction, and, just as prominently,
 * whether it beats the baseline it was measured against. A model that
 * doesn't beat its baseline has no demonstrated skill, so its prediction
 * must never be read without that next to it.
 */
import type { ReactNode } from "react";

import { useModels, usePrediction } from "../../api/queries";
import type { ModelInfo, Prediction } from "../../api/types";
import { Empty, Loading, ProblemView } from "../../components/status";
import { num, pct, signedPct } from "../../format";

export function ModelView() {
  // useModels (useQuery): the registered models; the last is the latest.
  const models = useModels();
  const latest = models.data?.models.at(-1);
  // usePrediction (useQuery, sent once): POST /predict for that version.
  // It waits until the latest version is known, so the prediction and the
  // model shown beside it are always the same version.
  const prediction = usePrediction(latest?.version);

  let content;
  if (models.isPending) {
    content = <Loading label="Loading the models…" />;
  } else if (models.isError) {
    content = (
      <ProblemView
        context="Could not load the models"
        problem={models.error}
        onRetry={() => void models.refetch()}
      />
    );
  } else if (!latest) {
    content = (
      <Empty title="No model is registered yet">
        <p>
          Train one, then reload this page: <code>python -m qbt_service.train SPY</code> (see the
          README).
        </p>
      </Empty>
    );
  } else {
    content = (
      <ModelCard
        model={latest}
        prediction={prediction.data}
        predictionState={
          prediction.isError ? (
            <ProblemView
              context="Could not get a prediction"
              problem={prediction.error}
              onRetry={() => void prediction.refetch()}
            />
          ) : prediction.isPending ? (
            <Loading label="Predicting…" />
          ) : null
        }
      />
    );
  }

  return (
    <section aria-labelledby="model-heading">
      <h2 id="model-heading">Model</h2>
      {content}
    </section>
  );
}

function ModelCard({
  model,
  prediction,
  predictionState,
}: {
  model: ModelInfo;
  prediction: Prediction | undefined;
  predictionState: ReactNode;
}) {
  const { baseline, out_of_sample: oos } = model;
  return (
    <article className="model-card" aria-label={`Model version ${String(model.version)}`}>
      <p className="job-header">
        <span className="badge">model version {model.version}</span> {model.model_type} · target{" "}
        {model.target} · trained on {model.training_window.n_months} months up to{" "}
        {model.training_window.last_decision}
      </p>

      <div className="model-headlines">
        <section className="headline-panel" aria-labelledby="prediction-heading">
          <h3 id="prediction-heading">Latest prediction</h3>
          {predictionState}
          {prediction && (
            <>
              <p className="headline-big">{prediction.prediction_name}</p>
              <p>
                For {prediction.predicts}, as of {prediction.as_of} (model version{" "}
                {prediction.model_version}).
              </p>
              <ul className="probabilities" aria-label="Probabilities">
                {prediction.probabilities.map((p) => (
                  <li key={p.label}>
                    {p.name}: {pct(p.probability)}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        <section
          className={`headline-panel ${baseline.beats ? "beats" : "does-not-beat"}`}
          aria-labelledby="baseline-heading"
        >
          <h3 id="baseline-heading">Against its baseline</h3>
          <p className="headline-big">
            <span aria-hidden="true">{baseline.beats ? "✓ " : "✕ "}</span>
            {baseline.beats ? "Beats" : "Does not beat"} {baseline.name}
          </p>
          <p className="headline-pair">
            <span>
              <span className="headline-big">{pct(baseline.accuracy)}</span>
              <span className="headline-label">baseline accuracy</span>
            </span>
            <span>
              <span className="headline-big">{pct(oos.accuracy)}</span>
              <span className="headline-label">model accuracy</span>
            </span>
          </p>
          <p>
            beats: <code>{String(baseline.beats)}</code>. Difference{" "}
            {signedPct(baseline.difference)} (95% CI {signedPct(baseline.difference_ci[0])} to{" "}
            {signedPct(baseline.difference_ci[1])}), McNemar p = {num(baseline.mcnemar_p, 3)}, over{" "}
            {oos.n_months} out-of-sample months ({oos.first_test_decision} to{" "}
            {oos.last_test_decision}).
          </p>
          <p className="hint">{baseline.description}</p>
        </section>
      </div>

      <p>{model.summary}</p>
    </article>
  );
}
