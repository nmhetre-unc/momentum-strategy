/** The application shell: the header and navigation, and the page for the current URL. */
import { Link } from "./components/Link";
import { Empty } from "./components/status";
import { CompareView } from "./features/compare/CompareView";
import { ModelView } from "./features/model/ModelView";
import { ResultsView } from "./features/results/ResultsView";
import { RunView } from "./features/run/RunView";
import { matchRoute, useLocation, type Route } from "./router";

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
        <Page route={route} />
      </main>
    </>
  );
}
