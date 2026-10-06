// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import React from "react";
import ReactDOM from "react-dom/client";
import "./theme/index.css";
import "./studio.css";
import "./studio-components.css";
import "./theme/legacy.css";
import "./shell/shell.css";
import "./ui/ui.css";
import "./home/home.css";
import "./library/library.css";
import "./scans/scans.css";
import "./scans/newscan.css";
import "./scans/livescan.css";
import "./review/review.css";
import "./review/deck.css";
import "./review/theater.css";
import "./live/live.css";
import "./memory/memory.css";
import "./insights/insights.css";
import "./export/compilations.css";
import "./export/export.css";
import "./settings/settings.css";
import "./settings/settings-kit.css";
import "./onboarding/onboarding.css";
import "./tools/tools.css";
import "./pets/pets.css";
import "./moments/moments.css";
import StudioApp from "./StudioApp";
import { applyAccent, readStoredAccent } from "./theme/accent";

// Apply the persisted theme and accent before first paint so the window never
// flashes the wrong ground or color. The fallback must match the store's
// default ("dark") or first launch flashes light before React mounts.
try {
  const root = document.documentElement;
  root.dataset.theme = localStorage.getItem("recall-theme") || "dark";
  applyAccent(readStoredAccent(), root);
} catch {}

// Dev only: `?fixtures` fills the library with synthetic sessions for UI work.
// Statically false in production builds, so the fixtures are never bundled.
if (import.meta.env.DEV && new URLSearchParams(window.location.search).has("fixtures")) {
  void import("./dev/fixtures").then(({ seedFixtures, exposeDevTriggers }) => { seedFixtures(); void exposeDevTriggers(); });
}

// Dev only: `?opencut-stream=<url>` mounts just the editor on a streamed video,
// for measuring it (dev/opencut-stream.tsx).
const streamTest = import.meta.env.DEV ? new URLSearchParams(window.location.search).get("opencut-stream") : null;
const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);
if (streamTest) {
  void import("./dev/opencut-stream").then(({ OpenCutStreamHarness }) => root.render(<OpenCutStreamHarness url={streamTest} />));
} else {
  root.render(
    <React.StrictMode>
      <StudioApp />
    </React.StrictMode>,
  );
}
