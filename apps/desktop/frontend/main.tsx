// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import React from "react";
import ReactDOM from "react-dom/client";
import "./studio.css";
import "./studio-components.css";
import StudioApp from "./StudioApp";

// Apply persisted theme before first paint to avoid a flash of the wrong
// theme. Fallback must match the store's default ("dark") or first launch
// flashes light before React mounts.
try {
  const root = document.documentElement;
  root.dataset.theme = localStorage.getItem("recall-theme") || "dark";
} catch {}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <StudioApp />
  </React.StrictMode>,
);
