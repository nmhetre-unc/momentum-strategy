// Adds DOM matchers to Vitest's expect: toBeInTheDocument(), toHaveTextContent() ...
import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver, which Recharts' ResponsiveContainer uses to
// measure its box. A no-op is enough: jsdom has no layout, so charts render
// their frame and the tests check what surrounds them (captions, labels).
class NoopResizeObserver {
  observe(): void {
    // nothing to observe without layout
  }
  unobserve(): void {
    // nothing to stop
  }
  disconnect(): void {
    // nothing to stop
  }
}
if (!("ResizeObserver" in globalThis)) globalThis.ResizeObserver = NoopResizeObserver;

afterEach(() => {
  window.localStorage.clear();
  window.history.replaceState(null, "", "/");
});
