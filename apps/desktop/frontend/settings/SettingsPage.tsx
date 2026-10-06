// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Settings, as one page: a sticky section list on the left, every section on
 * the right. Changes save as you make them.
 */
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { CaptionEditor } from "../components/CaptionEditor";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import { HardwareSettingsPanel } from "../components/HardwareSetup";
import RememberHotkeySetting from "../components/RememberHotkeySetting";
import { ChoiceCards, ExportLayoutChoices, StudioSwitch } from "../components/StudioControls";
import { Check } from "../lib/icons";
import { DEFAULT_CAPTION_STYLE, useJobStore, type CaptionStyle, type Settings } from "../lib/store";
import { AvatarPicker } from "./AvatarPicker";
import type { View } from "../shell/views";
import { CAPTION_COLORS, captionColorHex, type AccentChoice, type CaptionColor } from "../theme/accent";
import { AccentWheel } from "./AccentWheel";
import { CrewSection } from "../pets/CrewSection";
import { LearningPanel, PERFORMANCE_CHOICES, PROCESSING_CHOICES, SettingRow, Shortcut, StoragePanel } from "./panels";

export type SettingsSection = "you" | "scanning" | "clips" | "look" | "crew" | "storage" | "shortcuts" | "pc" | "about";

const SECTIONS: { id: SettingsSection; label: string }[] = [
  { id: "you", label: "You & learning" },
  { id: "scanning", label: "Scanning" },
  { id: "clips", label: "Clips & captions" },
  { id: "look", label: "Look" },
  { id: "crew", label: "Your crew" },
  { id: "storage", label: "Storage" },
  { id: "shortcuts", label: "Shortcuts" },
  { id: "pc", label: "This PC" },
  { id: "about", label: "About" },
];

/** Older deep links used modal tab names. */
export function sectionForTab(tab: string): SettingsSection {
  return ({ general: "you", project: "scanning", export: "clips", appearance: "look", shortcuts: "shortcuts" } as Record<string, SettingsSection>)[tab]
    ?? (SECTIONS.some((s) => s.id === tab) ? (tab as SettingsSection) : "you");
}

const persistLocal = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch {} };

export function SettingsPage({ section, sectionNonce, onNavigate }: { section: SettingsSection; sectionNonce: number; onNavigate: (view: View) => void }) {
  const settings = useJobStore((state) => state.settings);
  const keepers = useJobStore((state) => state.sessions.reduce((sum, s) => sum + s.savedClipIds.length, 0));
  const setSettings = useJobStore((state) => state.setSettings);
  const [saved, setSaved] = useState(false);
  const [active, setActive] = useState<SettingsSection>(section);
  const savedTimer = useRef<number | null>(null);
  const root = useRef<HTMLDivElement>(null);

  const markSaved = () => {
    setSaved(true);
    if (savedTimer.current) window.clearTimeout(savedTimer.current);
    savedTimer.current = window.setTimeout(() => setSaved(false), 1600);
  };
  const commit = (patch: Partial<Settings>) => { setSettings(patch); markSaved(); };
  const caption = settings.captionStyle ?? DEFAULT_CAPTION_STYLE;
  const patchCaption = (patch: Partial<CaptionStyle>) => {
    const next = { ...caption, ...patch };
    setSettings({ captionStyle: next });
    persistLocal("recall-caption-style", JSON.stringify(next));
    markSaved();
  };
  const setCaptionColor = (captionColor: CaptionColor) => {
    const next = { ...caption, highlightColor: captionColorHex(captionColor, settings.accent) };
    setSettings({ captionColor, captionStyle: next });
    persistLocal("recall-caption-style", JSON.stringify(next));
    markSaved();
  };
  const setAccent = (accent: AccentChoice) => {
    // "Match accent" captions follow the accent into the renderer too.
    if (settings.captionColor === "accent") {
      const next = { ...caption, highlightColor: captionColorHex("accent", accent) };
      setSettings({ accent, captionStyle: next });
      persistLocal("recall-caption-style", JSON.stringify(next));
    } else {
      setSettings({ accent });
    }
    markSaved();
  };

  const go = (id: SettingsSection, smooth = true) => {
    setActive(id);
    root.current?.querySelector(`#settings-${id}`)?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
  };

  // A deep link (Ctrl K "Keyboard shortcuts", Home's accent step) lands on its section.
  useEffect(() => {
    if (section === "you") {
      setActive("you");
      root.current?.closest(".view-panel")?.scrollTo({ top: 0 });
    } else {
      go(section, false);
    }
  }, [section, sectionNonce]);

  // Keep the section list in step with scrolling.
  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        const top = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (top) setActive(top.target.id.replace("settings-", "") as SettingsSection);
      },
      { rootMargin: "-10% 0px -70% 0px" },
    );
    el.querySelectorAll("section[id^='settings-']").forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="page settings-page" ref={root}>
      <div className="ph">
        <div>
          <h1 className="disp">Settings</h1>
          <div className="sub">Everything saves as you go.</div>
        </div>
        <div className="actions">
          <span className={`set-saved ${saved ? "on" : ""}`} role="status" aria-live="polite">{saved && <><Check aria-hidden="true" />Saved</>}</span>
        </div>
      </div>

      <div className="set-grid">
        <nav className="set-nav glass" aria-label="Settings sections">
          {SECTIONS.map((s) => (
            <button key={s.id} type="button" className={active === s.id ? "on" : ""} aria-current={active === s.id ? "true" : undefined} onClick={() => go(s.id)}>
              {s.label}
            </button>
          ))}
        </nav>

        <div className="set-body">
          {/* Two independent stacks, side by side when there is room. Not CSS
              columns: fragmenting nine glass cards made every layout ~13x
              slower (22 ms vs 1.7 ms), which is what made Settings lag. */}
          <div className="set-col">
          <section className="set-sec glass" id="settings-you" aria-labelledby="settings-you-h">
            <h2 className="disp" id="settings-you-h">You & learning</h2>
            <p>Your name and photo, and what Recall has picked up about your taste.</p>
            <div className="set-name">
              <AvatarPicker name={settings.displayName || "Creator"} size={64} />
              <label className="set-name-field">
                <span>Your name</span>
                <input
                  className="field"
                  value={settings.displayName}
                  placeholder="Creator"
                  maxLength={32}
                  onChange={(event) => { commit({ displayName: event.target.value }); persistLocal("recall-display-name", event.target.value); }}
                />
                <small className="t3">Shows in the greeting and on your avatar. Your name and photo never leave this PC.</small>
              </label>
            </div>
            <LearningPanel />
            <SettingRow title="Keep this PC awake while working" copy="Stops the PC from sleeping during scans and exports. The display can still turn off, and Recall lets go the moment the work is done.">
              <StudioSwitch checked={settings.keepAwakeWhileWorking} onChange={(keepAwakeWhileWorking) => commit({ keepAwakeWhileWorking })} label="Keep this PC awake while scans and exports are active" />
            </SettingRow>
            <SettingRow title="Tell me when things finish" copy="A desktop notification when a scan, an overnight batch, an export, or a reel finishes or needs you.">
              <StudioSwitch checked={settings.desktopNotifications} onChange={(desktopNotifications) => commit({ desktopNotifications })} label="Show completion and failure notifications" />
            </SettingRow>
          </section>

          <section className="set-sec glass" id="settings-scanning" aria-labelledby="settings-scanning-h">
            <h2 className="disp" id="settings-scanning-h">Scanning</h2>
            <p>Recall tunes sensitivity, clip count and clip length to each stream. You choose how deep it looks and how much of this PC it uses. Speech is English-first for now; other languages still get voice, face, chat and game signals.</p>
            <SettingRow block title="Scan speed" copy="Smart scan looks closely where it matters. Best quality reads the whole VOD and takes longer.">
              <ChoiceCards value={settings.processingMode} options={PROCESSING_CHOICES} onChange={(processingMode) => commit({ processingMode })} label="Scan speed" />
            </SettingRow>
            <SettingRow block title="While scanning" copy="Keep PC responsive is for scanning while you stream or play. Full speed is for when you step away.">
              <ChoiceCards value={settings.performanceProfile} options={PERFORMANCE_CHOICES} onChange={(performanceProfile) => commit({ performanceProfile })} label="While scanning" />
            </SettingRow>
          </section>

          <section className="set-sec glass" id="settings-clips" aria-labelledby="settings-clips-h">
            <h2 className="disp" id="settings-clips-h">Clips & captions</h2>
            <p>How your clips come out: the framing and the burned-in captions.</p>
            <SettingRow block title="Facecam layout" copy="Automatic keeps one look per stream: stacked for games that need context, picture-in-picture for action. Or lock one.">
              <ExportLayoutChoices value={settings.exportLayout} onChange={(exportLayout) => { commit({ exportLayout }); persistLocal("recall-export-layout", exportLayout); }} />
            </SettingRow>
            <CaptionEditor caption={caption} onChange={patchCaption} highlightControl={<CaptionHighlightPicker value={settings.captionColor} accent={settings.accent} onChange={setCaptionColor} />} />
          </section>

          <section className="set-sec glass" id="settings-look" aria-labelledby="settings-look-h">
            <h2 className="disp" id="settings-look-h">Look</h2>
            <p>Recall is black and white. Pick one accent and it carries through everything that matters. Keeps, live and warnings keep their own colors.</p>
            <SettingRow title="Theme" copy="Dark is made for dim rooms and footage. Light keeps the same layout for bright ones.">
              <div className="seg" role="radiogroup" aria-label="Theme">
                {(["dark", "light"] as const).map((theme) => (
                  <button key={theme} type="button" role="radio" aria-checked={settings.theme === theme} onClick={() => commit({ theme })}>{theme === "dark" ? "Dark" : "Light"}</button>
                ))}
              </div>
            </SettingRow>
            <AccentWheel value={settings.accent} onChange={setAccent} />
          </section>

          </div>
          <div className="set-col">
          <section className="set-sec glass" id="settings-crew" aria-labelledby="settings-crew-h">
            <h2 className="disp" id="settings-crew-h">Your crew</h2>
            <p>Little friends who live at the bottom of the sidebar. They cheer your keepers, lose it on big moments, and wander out while a scan runs. They never get hungry or sad.</p>
            <CrewSection keepers={keepers} />
          </section>

          <section className="set-sec glass" id="settings-storage" aria-labelledby="settings-storage-h">
            <h2 className="disp" id="settings-storage-h">Storage</h2>
            <p>What Recall keeps on this PC, and what it can clear.</p>
            <StoragePanel commitSettings={commit} onManageClips={() => onNavigate("clips")} />
          </section>

          <section className="set-sec glass" id="settings-shortcuts" aria-labelledby="settings-shortcuts-h">
            <h2 className="disp" id="settings-shortcuts-h">Shortcuts</h2>
            <p>Review at the speed of your gut.</p>
            <RememberHotkeySetting />
            <div className="shortcuts">
            <Shortcut label="Keep" value="Space" />
            <Shortcut label="Maybe" value="M" />
            <Shortcut label="Pass" value="X or Backspace" />
            <Shortcut label="Next / previous moment" value="→ / ←" />
            <Shortcut label="Search and commands" value="Ctrl K" />
            <Shortcut label="Back to the deck" value="Esc" />
            </div>
          </section>

          <section className="set-sec glass" id="settings-pc" aria-labelledby="settings-pc-h">
            <h2 className="disp" id="settings-pc-h">This PC</h2>
            <p>What Recall found on this machine and how it runs scans here.</p>
            <HardwareSettingsPanel active apiEndpoint={settings.apiEndpoint} />
            <DiagnosticsPanel />
          </section>

          <section className="set-sec glass" id="settings-about" aria-labelledby="settings-about-h">
            <h2 className="disp" id="settings-about-h">About</h2>
            <p>Recall finds the moments worth posting in your streams. It runs entirely on this PC, and nothing uploads.</p>
            <p className="t3">Free and open source under the GNU AGPL 3.0.</p>
          </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function CaptionHighlightPicker({ value, accent, onChange }: { value: CaptionColor; accent: AccentChoice; onChange: (next: CaptionColor) => void }) {
  return (
    <div className="form-group cap-pick">
      <span className="form-label">Highlight color</span>
      <div className="cap-colors" role="radiogroup" aria-label="Caption highlight color">
        {CAPTION_COLORS.map((color) => (
          <button
            key={color.id}
            type="button"
            role="radio"
            aria-checked={value === color.id}
            aria-label={color.label}
            title={color.label}
            className={`cap-dot ${value === color.id ? "on" : ""}`}
            style={{ "--dot": captionColorHex(color.id, accent) } as CSSProperties}
            onClick={() => onChange(color.id)}
          >
            {color.id === "accent" && <span aria-hidden="true">A</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
