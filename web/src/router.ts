/**
 * Client-side routing in about 60 lines, without a library.
 *
 * The app has four pages and no nested layouts, data loaders or route
 * guards, which is what a router library (React Router, TanStack Router)
 * mostly provides. So the URL is read with the browser's own History API:
 *
 * - navigate(to) changes the address bar without reloading the page
 *   (history.pushState) and tells subscribers it changed.
 * - useLocation() re-renders a component when the address changes, by
 *   navigate() or by the back and forward buttons (the popstate event).
 * - matchRoute() turns a path into one of the routes below.
 *
 * The server has to answer every one of these paths with index.html (an
 * "SPA fallback"); Vite's dev server already does.
 */
import { useSyncExternalStore } from "react";

export type Route =
  | { name: "run" }
  | { name: "results"; id: string }
  | { name: "compare"; ids: string[] }
  | { name: "model" }
  | { name: "not-found"; path: string };

const NAVIGATE = "qbt:navigate";

function subscribe(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  window.addEventListener(NAVIGATE, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(NAVIGATE, onChange);
  };
}

function snapshot(): string {
  return window.location.pathname + window.location.search;
}

/** The current path and query string, e.g. "/compare?ids=a,b". */
export function useLocation(): string {
  // useSyncExternalStore: the address bar is state that lives outside
  // React. This hook subscribes to its changes and reads it consistently
  // during a render, which useState + useEffect would only approximate.
  return useSyncExternalStore(subscribe, snapshot);
}

export function navigate(to: string, { replace = false } = {}): void {
  if (replace) window.history.replaceState(null, "", to);
  else window.history.pushState(null, "", to);
  window.dispatchEvent(new Event(NAVIGATE));
}

/** The ids in a compare URL's `ids` parameter, comma-separated, without duplicates. */
export function parseIds(search: string): string[] {
  const raw = new URLSearchParams(search).get("ids") ?? "";
  return [
    ...new Set(
      raw
        .split(",")
        .map((id) => id.trim())
        .filter(Boolean),
    ),
  ];
}

export function compareHref(ids: readonly string[]): string {
  return ids.length ? `/compare?ids=${ids.map(encodeURIComponent).join(",")}` : "/compare";
}

export function matchRoute(location: string): Route {
  const url = new URL(location, "http://localhost");
  const path = url.pathname.replace(/\/+$/, "") || "/";
  if (path === "/") return { name: "run" };
  if (path === "/compare") return { name: "compare", ids: parseIds(url.search) };
  if (path === "/model") return { name: "model" };
  const results = /^\/backtests\/([^/]+)$/.exec(path);
  if (results?.[1]) return { name: "results", id: decodeURIComponent(results[1]) };
  return { name: "not-found", path };
}
