/**
 * The three states every piece of server data can be in besides "here it
 * is": still loading, failed, or loaded but empty. One component each, so
 * every view shows them the same way.
 */
import type { ReactNode } from "react";

import type { ApiProblem } from "../api/problem";

export function Loading({ label }: { label: string }) {
  return (
    <p className="status loading" role="status">
      {label}
    </p>
  );
}

/** An ApiProblem: what went wrong, the API's detail, any per-field errors. */
export function ProblemView({
  problem,
  onRetry,
  context,
}: {
  problem: ApiProblem;
  onRetry?: (() => void) | undefined;
  context?: string | undefined;
}) {
  return (
    <div className="status problem" role="alert">
      <p className="problem-title">
        {context ? `${context}: ` : ""}
        {problem.title}
        {problem.status ? ` (${String(problem.status)})` : ""}
      </p>
      <p>{problem.detail}</p>
      {problem.errors.length > 0 && (
        <ul>
          {problem.errors.map((e) => (
            <li key={`${e.loc.join(".")}:${e.msg}`}>
              <code>{e.loc.join(".")}</code>: {e.msg}
            </li>
          ))}
        </ul>
      )}
      <p className="problem-type">
        <code>{problem.type}</code>
      </p>
      {onRetry && (
        <button type="button" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="status empty">
      <p className="empty-title">{title}</p>
      {children}
    </div>
  );
}
