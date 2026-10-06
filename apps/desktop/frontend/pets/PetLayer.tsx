// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * The crew, alive. Pets live in the den at the bottom of the sidebar and
 * wander out along the window's bottom edge only while a scan runs (and never
 * over Review, Theater or the Cutting Room). They react to real events:
 * a keeper (cheer), a 90+ moment (spin + confetti), a finished scan (jump),
 * a milestone (unlock). Idle long enough and they nap. Only one talks at a time.
 *
 * Movement runs in one requestAnimationFrame loop that writes transforms
 * directly; React only renders the sprites, bubbles and particles. Reduced
 * motion keeps them home and still.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { PET_LINES, PET_UNLOCK_2, heartSvg, petName, petPick, petSvg } from "./petArt";
import { onPetEvent, useCrew, type PetReaction } from "./petStore";

const NAP_AFTER_MS = 4 * 60_000;
const REACTIONS: Record<PetReaction, [mood: string, ms: number, fx: "heart" | "confetti", count: number]> = {
  pet: ["cheer", 1100, "heart", 4],
  keep: ["cheer", 1300, "heart", 5],
  wild: ["wild", 1700, "confetti", 12],
  scan: ["jump", 1600, "confetti", 8],
  unlock: ["wild", 1900, "confetti", 14],
};
const CONFETTI = ["#FFC83D", "#4C8DFF", "#FF5A5F", "#7FE3A8", "#FFFFFF"];
const HEARTS = ["#FF6B8B", "#FF8FA3", "#FFB3C4"];

interface Motion {
  x: number | null;
  y: number | null;
  tx: number | null;
  dir: number;
  wait: number;
  mode?: "den" | "roam";
  mood: string;
  moodUntil: number;
  hop: number;
  shown?: string;
}

interface Particle { id: number; kind: "heart" | "confetti"; dx: number; delay: number; color: string }

export const prefersReducedMotion = () =>
  typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

let particleId = 0;

/** `hidden`: a full-screen view (the theater) covers the sidebar the den lives in. */
export function PetLayer({ keepers, scanning, focus, hidden = false }: { keepers: number; scanning: boolean; focus: boolean; hidden?: boolean }) {
  const { crew, show, roam } = useCrew();
  const active = useMemo(() => (show ? crew.slice(0, keepers >= PET_UNLOCK_2 ? 2 : 1) : []), [crew, show, keepers]);
  const signature = active.map((c) => `${c.k}|${c.look}|${c.acc}`).join(",");
  const sprites = useMemo(() => active.map((c) => petSvg(c.k, c.acc, c.look)), [signature]);

  const layer = useRef<HTMLDivElement>(null);
  const els = useRef<(HTMLDivElement | null)[]>([]);
  const motion = useRef<Motion[]>([]);
  const env = useRef({ scanning, focus, roam, sleeping: false, idleAt: performance.now() });
  env.current.scanning = scanning;
  env.current.focus = focus;
  env.current.roam = roam;

  const [bubbles, setBubbles] = useState<Record<number, { text: string; key: number } | undefined>>({});
  const [particles, setParticles] = useState<Record<number, Particle[]>>({});
  const [sleeping, setSleeping] = useState(false);

  const say = (slot: number, text: string, ms = 2200) => {
    if (!text) return;
    const key = Date.now() + slot;
    setBubbles((b) => ({ ...b, [slot]: { text, key } }));
    window.setTimeout(() => setBubbles((b) => (b[slot]?.key === key ? { ...b, [slot]: undefined } : b)), ms);
  };
  const burst = (slot: number, kind: "heart" | "confetti", n: number) => {
    if (prefersReducedMotion()) return;
    const fresh: Particle[] = Array.from({ length: n }, (_, i) => ({
      id: ++particleId,
      kind,
      dx: (Math.random() - 0.5) * 60,
      delay: i * 90,
      color: kind === "heart" ? HEARTS[i % 3] : CONFETTI[i % 5],
    }));
    setParticles((p) => ({ ...p, [slot]: [...(p[slot] ?? []), ...fresh] }));
    const ids = new Set(fresh.map((f) => f.id));
    window.setTimeout(() => setParticles((p) => ({ ...p, [slot]: (p[slot] ?? []).filter((f) => !ids.has(f.id)) })), 1500 + n * 90);
  };
  const wake = () => {
    env.current.idleAt = performance.now();
    if (!env.current.sleeping) return false;
    env.current.sleeping = false;
    setSleeping(false);
    say(0, petPick(PET_LINES.wake));
    return true;
  };
  const react = (slot: number, what: PetReaction, quiet = false) => {
    if (wake()) return;
    const m = motion.current[slot];
    if (!m) return;
    const [mood, ms, fx, count] = REACTIONS[what];
    m.mood = mood;
    m.moodUntil = performance.now() + ms;
    m.tx = null;
    m.shown = undefined;
    burst(slot, fx, count);
    if (what !== "unlock" && !quiet) say(slot, petPick(PET_LINES[what === "pet" ? "pet" : what]));
  };

  // Reactions from the app: together they react, only the first one talks.
  useEffect(() => onPetEvent(({ what, text }) => {
    motion.current.forEach((_, slot) => window.setTimeout(() => {
      react(slot, what, slot > 0 || !!text);
      if (text && slot === 0) say(0, text, 3200);
    }, slot * 180));
  }));

  // Any input keeps them awake.
  useEffect(() => {
    const onInput = () => { wake(); };
    const opts = { passive: true, capture: true } as const;
    ["pointerdown", "keydown", "wheel"].forEach((e) => document.addEventListener(e, onInput, opts));
    return () => ["pointerdown", "keydown", "wheel"].forEach((e) => document.removeEventListener(e, onInput, opts));
  });

  // One motion record per active slot.
  useEffect(() => {
    motion.current = active.map((_, i) => motion.current[i] ?? { x: null, y: null, tx: null, dir: i ? -1 : 1, wait: 800 + i * 1200, mood: "idle", moodUntil: 0, hop: 0 });
  }, [signature]);

  useEffect(() => {
    if (!active.length) return;
    let raf = 0;
    let last = performance.now();
    const reduced = prefersReducedMotion();

    const bounds = () => {
      const e = env.current;
      if (e.roam && e.scanning && !e.focus && !reduced) {
        const sb = document.querySelector(".sidebar");
        const l = sb ? sb.getBoundingClientRect().right + 24 : 24;
        return { mode: "roam" as const, l, r: window.innerWidth - 90, y: window.innerHeight - 4, size: 58 };
      }
      const den = document.querySelector("[data-den]");
      if (den) {
        const r = den.getBoundingClientRect();
        if (r.width) return { mode: "den" as const, l: r.left + 6, r: r.right - 6, y: r.bottom - 8, size: 46 };
      }
      return null;
    };

    const tick = (t: number) => {
      const dt = Math.min(60, t - last);
      last = t;
      const B = bounds();
      layer.current?.classList.toggle("is-hidden", !B);
      const pets = motion.current;
      if (B) pets.forEach((p, i) => {
        const el = els.current[i];
        if (!el) return;
        const w = B.size;
        const maxX = B.r - w;
        if (p.x == null || p.y == null) { p.x = B.l + (B.r - B.l - w) * (i ? 0.72 : 0.22); p.y = B.y; }
        if (p.mode !== B.mode) {
          // Hop between the den and the floor.
          if (p.mode && !reduced) { p.hop = 1; if (B.mode === "roam" && i === 0) say(0, petPick(PET_LINES.roam), 1600); }
          p.mode = B.mode;
          p.tx = null;
          p.wait = 200 + i * 500;
        }
        p.x = Math.min(Math.max(p.x, B.l), Math.max(B.l, maxX));
        const busy = p.moodUntil > t;
        if (!busy && !env.current.sleeping && !reduced) {
          if (p.tx == null) {
            p.wait -= dt;
            if (p.wait <= 0) {
              const lead = pets[0];
              const other = pets[i ? 0 : 1];
              const gap = w + 8;
              const free = (x: number) => !other || other.x == null || Math.abs(x - (other.tx ?? other.x)) >= gap;
              if (i && lead?.x != null && Math.random() < 0.45) {
                // The second pet sometimes tags along after the first.
                const side = lead.x - gap >= B.l ? (lead.x + gap <= maxX && Math.random() < 0.5 ? 1 : -1) : 1;
                p.tx = Math.min(Math.max(lead.x + side * gap, B.l), maxX);
              } else {
                let x = B.l;
                let n = 0;
                do { x = B.l + Math.random() * Math.max(0, maxX - B.l); } while (!free(x) && ++n < 8);
                p.tx = x;
              }
              if (!free(p.tx)) { p.tx = null; p.wait = 600; }
            }
          } else {
            const speed = (B.mode === "roam" ? 0.06 : 0.028) * dt;
            const d = p.tx - p.x;
            if (Math.abs(d) <= speed) { p.x = p.tx; p.tx = null; p.wait = 1400 + Math.random() * 4200; }
            else { p.x += Math.sign(d) * speed; p.dir = Math.sign(d); }
          }
        }
        p.y += (B.y - p.y) * Math.min(1, dt / 160);
        p.hop = Math.max(0, p.hop - dt / 650);
        const arc = Math.sin(p.hop * Math.PI) * 40;
        const walking = p.tx != null && !busy && !env.current.sleeping;
        const mood = env.current.sleeping ? "sleep" : busy ? p.mood : walking ? "walk" : "idle";
        if (p.shown !== mood) { el.dataset.mood = mood; p.shown = mood; }
        el.style.setProperty("--ps", `${w}px`);
        el.classList.toggle("left", p.dir < 0);
        el.style.transform = `translate3d(${p.x.toFixed(1)}px, ${(p.y - w - arc).toFixed(1)}px, 0)`;
      });
      if (!env.current.sleeping && t - env.current.idleAt > NAP_AFTER_MS) {
        env.current.sleeping = true;
        setSleeping(true);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [signature]);

  if (!active.length) return null;
  return (
    <div ref={layer} className={`pet-layer ${sleeping ? "is-sleeping" : ""}`} aria-hidden="true" style={hidden ? { visibility: "hidden" } : undefined}>
      {active.map((member, slot) => (
        <div
          key={`${slot}-${member.k}-${member.look}-${member.acc}`}
          ref={(el) => { els.current[slot] = el; }}
          className="pet"
          data-mood="idle"
          title={petName(member)}
          onClick={() => react(slot, "pet")}
        >
          <div className="pet-in" dangerouslySetInnerHTML={{ __html: sprites[slot] ?? "" }} />
          <span className="pet-shadow" />
          {bubbles[slot] && <span key={bubbles[slot]!.key} className="pet-bub">{bubbles[slot]!.text}</span>}
          {(particles[slot] ?? []).map((f) => (
            <span
              key={f.id}
              className={`pet-fx ${f.kind}`}
              style={{ ["--dx" as string]: `${f.dx}px`, ["--d" as string]: `${f.delay}ms`, background: f.kind === "confetti" ? f.color : undefined }}
              dangerouslySetInnerHTML={f.kind === "heart" ? { __html: heartSvg(f.color) } : undefined}
            />
          ))}
        </div>
      ))}
    </div>
  );
}
