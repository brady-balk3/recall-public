// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import type { ReactNode } from "react";

type EmptyVariant = "sessions" | "source" | "review" | "library" | "insights" | "editor";

type Props = {
  variant: EmptyVariant;
  icon: ReactNode;
  title: string;
  body?: string;
  action?: ReactNode;
  compact?: boolean;
};

/**
 * The page kit's empty state: a glass card with the icon, a display title, a
 * line of what to do, and the action. `compact` for one inside a panel.
 * `variant` is kept for callers; it no longer changes the look.
 */
export default function StudioEmptyState({ variant, icon, title, body, action, compact = false }: Props) {
  return (
    <div className={`empty glass studio-empty is-${variant}${compact ? " is-compact" : ""}`}>
      <span className="studio-empty-icon" aria-hidden="true">{icon}</span>
      <h3 className="disp">{title}</h3>
      {body ? <p>{body}</p> : null}
      {action ? <div className="w-cta studio-empty-action">{action}</div> : null}
    </div>
  );
}
