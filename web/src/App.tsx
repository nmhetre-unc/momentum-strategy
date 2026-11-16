/** The application shell: the header and navigation, and the page for the current URL. */
import { lazy, Suspense } from "react";

import { Link } from "./components/Link";
import { Empty, Loading } from "./components/status";
import { ModelView } from "./features/model/ModelView";
import { RunView } from "./features/run/RunView";
import { matchRoute, useLocation, type Route } from "./router";

// The two pages with charts are loaded only when first opened: they bring
// Recharts, most of the app's code, which the Run and Model pages don't
// need. Vite puts each lazy import in its own file (a "chunk").
const ResultsView = lazy(() =>
  import("./features/results/ResultsView").then((m) => ({ default: m.ResultsView })),
);
const CompareView = lazy(() =>
  import("./features/compare/CompareView").then((m) => ({ default: m.CompareView })),
);

function Page({ route }: { route: Route }) {
  switch (route.name) {
    case "run":
      return <RunView />;
    case "results":
      // key: a different job id is a different page; its state starts fresh.
      return <ResultsView key={route.id} id={route.id} />;
    case "compare":
      return <CompareView ids={route.ids} />;
    case "model":
      return <ModelView />;
    case "not-found":
      return (
        <Empty title="Page not found">
          <p>
            Nothing lives at <code>{route.path}</code>. <Link to="/">Run a backtest</Link>.
          </p>
        </Empty>
      );
  }
}

export function App() {
  // useLocation (useSyncExternalStore): re-renders the app when the URL
  // changes, so the right page is shown.
  const route = matchRoute(useLocation());
  const current = (name: Route["name"]) => (route.name === name ? "page" : undefined);
  return (
    <>
      <header className="app-header">
        <h1>qbt dashboard</h1>
        <nav aria-label="Main">
          <Link to="/" aria-current={current("run")}>
            Run
          </Link>
          <Link to="/compare" aria-current={current("compare")}>
            Compare
          </Link>
          <Link to="/model" aria-current={current("model")}>
            Model
          </Link>
        </nav>
      </header>
      <main>
        {/* Suspense: what to show while a lazy page's code is downloading. */}
        <Suspense fallback={<Loading label="Loading the page…" />}>
          <Page route={route} />
        </Suspense>
      </main>
    </>
  );
}
