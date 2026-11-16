import type { AnchorHTMLAttributes } from "react";

import { navigate } from "../router";

type Props = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> & { to: string };

/**
 * A real <a href>, so it can be opened in a new tab, copied and read by
 * screen readers as a link. A plain left click is handled in the page
 * (navigate) instead of loading the whole app again.
 */
export function Link({ to, onClick, ...rest }: Props) {
  return (
    <a
      {...rest}
      href={to}
      onClick={(event) => {
        onClick?.(event);
        const modified = event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
        if (event.defaultPrevented || event.button !== 0 || modified) return;
        event.preventDefault();
        navigate(to);
      }}
    />
  );
}
