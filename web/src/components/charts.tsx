/**
 * The equity and drawdown charts (Recharts). Each chart has one y-axis and
 * comes with an HTML caption that says what it shows, including the shaded
 * training period, so the meaning never depends on the SVG alone.
 */
import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { num, pct } from "../format";
import { mergeByDate, type ChartSeries, type Shade } from "./chartData";
import { useChartTheme } from "../theme";

const HEIGHT = 260;

function TrainingShade({ shade, color }: { shade: Shade | undefined; color: string }) {
  if (!shade) return null;
  return (
    <ReferenceArea
      x1={shade.from}
      x2={shade.to}
      fill={color}
      fillOpacity={0.15}
      stroke="none"
      ifOverflow="hidden"
      label={{ value: "training (flat)", position: "insideTopLeft", fill: color, fontSize: 12 }}
    />
  );
}

export function EquityChart({
  title,
  series,
  shade,
}: {
  title: string;
  series: readonly ChartSeries[];
  shade?: Shade | undefined;
}) {
  // useChartTheme: picks light or dark colors and follows the OS setting.
  const theme = useChartTheme();
  // useMemo: merging thousands of daily points is the most expensive work
  // here; it is redone only when the series change, not on every render.
  const rows = useMemo(() => mergeByDate(series), [series]);
  const many = series.length > 1;

  return (
    <figure className="chart" aria-label={title}>
      <figcaption>
        <span className="chart-title">{title}</span>
        <span className="chart-note">
          Growth of 1 after costs.
          {shade &&
            ` Shaded: the training period, ${shade.from} up to ${shade.to}, held flat (no position).`}
        </span>
      </figcaption>
      <ResponsiveContainer width="100%" height={HEIGHT}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis dataKey="date" stroke={theme.axis} minTickGap={48} tick={{ fontSize: 12 }} />
          <YAxis
            stroke={theme.axis}
            width={48}
            tick={{ fontSize: 12 }}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => num(v)}
          />
          <TrainingShade shade={shade} color={theme.shade} />
          <Tooltip formatter={(v) => (typeof v === "number" ? num(v, 3) : String(v))} />
          {many && <Legend />}
          {series.map((s, i) => (
            <Line
              key={s.key}
              dataKey={s.key}
              name={s.label}
              stroke={theme.series[i] ?? theme.series[0]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}

export function DrawdownChart({
  dates,
  drawdown,
  shade,
}: {
  dates: readonly string[];
  drawdown: readonly number[];
  shade?: Shade | undefined;
}) {
  // useChartTheme: as in EquityChart.
  const theme = useChartTheme();
  // useMemo: as in EquityChart, the rows are rebuilt only when the data changes.
  const rows = useMemo(
    () => dates.map((date, i) => ({ date, drawdown: drawdown[i] ?? 0 })),
    [dates, drawdown],
  );
  const color = theme.series[0];

  return (
    <figure className="chart" aria-label="Drawdown">
      <figcaption>
        <span className="chart-title">Drawdown</span>
        <span className="chart-note">
          Fall from the running peak of the equity curve; its lowest point is the max drawdown.
        </span>
      </figcaption>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis dataKey="date" stroke={theme.axis} minTickGap={48} tick={{ fontSize: 12 }} />
          <YAxis
            stroke={theme.axis}
            width={48}
            tick={{ fontSize: 12 }}
            tickFormatter={(v: number) => pct(v, 0)}
          />
          <TrainingShade shade={shade} color={theme.shade} />
          <Tooltip formatter={(v) => (typeof v === "number" ? pct(v) : String(v))} />
          <Area
            dataKey="drawdown"
            name="Drawdown"
            stroke={color}
            fill={color}
            fillOpacity={0.25}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </figure>
  );
}
