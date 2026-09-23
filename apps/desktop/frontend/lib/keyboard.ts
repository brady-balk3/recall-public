// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
const INTERACTIVE_KEYBOARD_TARGET = [
  "button",
  "a[href]",
  "input",
  "select",
  "textarea",
  "summary",
  "[contenteditable]:not([contenteditable='false'])",
  "[role='button']",
  "[role='checkbox']",
  "[role='combobox']",
  "[role='gridcell']",
  "[role='link']",
  "[role='listbox']",
  "[role='menuitem']",
  "[role='option']",
  "[role='radio']",
  "[role='scrollbar']",
  "[role='searchbox']",
  "[role='slider']",
  "[role='spinbutton']",
  "[role='switch']",
  "[role='tab']",
  "[role='textbox']",
  "[role='treeitem']",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

/**
 * Global editor shortcuts must yield to the control that owns keyboard focus.
 * `closest` also covers icon/SVG descendants inside a semantic control.
 */
export function isInteractiveKeyboardTarget(target: EventTarget | null): boolean {
  return target instanceof Element && target.closest(INTERACTIVE_KEYBOARD_TARGET) !== null;
}
