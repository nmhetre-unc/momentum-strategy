/**
 * Chart colors. Recharts writes colors into SVG attributes, where CSS
 * variables don't apply, so the light and dark values are picked here.
 *
 * The categorical slots are a validated colorblind-safe order (adjacent
 * series stay distinguishable under the common color-vision deficiencies);
 * series take them in this fixed order and never cycle, which is why the
 * Compare view allows at most SERIES_COLORS.light.length backtests.
 */
import { useSyncExternalStore } from "react";

export const SERIES_COLORS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
} as const;

const CHROME = {
  light: { grid: "#e1e0d9", axis: "#898781", shade: "#898781", text: "#52514e" },
  dark: { grid: "#2c2c2a", axis: "#898781", shade: "#c3c2b7", text: "#c3c2b7" },
} as const;

export const MAX_SERIES = SERIES_COLORS.light.length;

const QUERY = "(prefers-color-scheme: dark)";

function subscribe(onChange: () => void): () => void {
  // jsdom (tests) has no matchMedia.
  if (typeof window.matchMedia !== "function") return () => undefined;
  const media = window.matchMedia(QUERY);
  media.addEventListener("change", onChange);
  return () => {
    media.removeEventListener("change", onChange);
  };
}

function isDark(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia(QUERY).matches;
}

export function useChartTheme() {
  // useSyncExternalStore: the OS light/dark setting lives outside React and
  // can change while the page is open; this re-renders the charts when it does.
  const dark = useSyncExternalStore(subscribe, isDark);
  const mode = dark ? "dark" : "light";
  return { series: SERIES_COLORS[mode], ...CHROME[mode] };
}
