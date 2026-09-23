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
 * Product-specific empty state: an honest, quiet preview of Recall's reaction
 * curve flowing into a review frame. The illustration is structural rather
 * than decorative, so first-use screens teach what will occupy the surface.
 */
export default function StudioEmptyState({ variant, icon, title, body, action, compact = false }: Props) {
  return (
    <div className={`studio-empty-content is-${variant}${compact ? " is-compact" : ""}`}>
      <div className="studio-empty-visual" aria-hidden="true">
        <svg viewBox="0 0 184 92" preserveAspectRatio="none">
          <path className="studio-empty-grid" d="M10 68H174M10 47H174M10 26H174M42 12V78M82 12V78M122 12V78M162 12V78" />
          <path className="studio-empty-area" d="M10 67C25 65 31 58 43 61C56 64 61 48 74 51C88 55 91 23 105 31C116 37 122 48 134 39C147 29 151 43 174 19V78H10Z" />
          <path className="studio-empty-curve" d="M10 67C25 65 31 58 43 61C56 64 61 48 74 51C88 55 91 23 105 31C116 37 122 48 134 39C147 29 151 43 174 19" />
          <circle className="studio-empty-peak" cx="105" cy="31" r="4" />
          <path className="studio-empty-window" d="M96 15H116V74H96Z" />
        </svg>
        <span className="studio-empty-glyph">{icon}</span>
        <span className="studio-empty-readout">R(t)</span>
      </div>
      <strong>{title}</strong>
      {body ? <span className="studio-empty-copy">{body}</span> : null}
      {action ? <div className="studio-empty-action">{action}</div> : null}
    </div>
  );
}
