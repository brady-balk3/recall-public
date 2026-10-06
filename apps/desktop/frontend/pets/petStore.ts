// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Crew state (who your pets are) and the reactions app events send them.
 * Local to this PC; nothing about pets ever reaches the engine.
 */
import { create } from "zustand";
import { PET_LOOKS, PETS, type PetGear, type PetKind } from "./petArt";

export interface CrewMember {
  k: PetKind;
  look: number;
  /** Empty means "use the look's default name". */
  name: string;
  acc: PetGear;
}

export interface CrewState {
  crew: CrewMember[];
  show: boolean;
  /** Wander along the window's bottom edge while a scan runs. */
  roam: boolean;
}

export const CREW_STORAGE_KEY = "recall-crew";

export const DEFAULT_CREW: CrewState = {
  crew: [{ k: "capybara", look: 0, name: "", acc: "none" }, { k: "axolotl", look: 0, name: "", acc: "none" }],
  show: true,
  roam: true,
};

const KINDS = new Set(PETS.map(([k]) => k));
const GEAR = new Set<PetGear>(["none", "bandana", "headset", "cap", "crown"]);

function sanitize(raw: unknown): CrewState {
  if (!raw || typeof raw !== "object") return DEFAULT_CREW;
  const value = raw as Partial<CrewState>;
  const crew = Array.isArray(value.crew)
    ? value.crew
        .filter((c): c is CrewMember => !!c && KINDS.has((c as CrewMember).k))
        .slice(0, 2)
        .map((c) => ({
          k: c.k,
          look: Number.isInteger(c.look) ? ((c.look % PET_LOOKS[c.k].length) + PET_LOOKS[c.k].length) % PET_LOOKS[c.k].length : 0,
          name: typeof c.name === "string" ? c.name.slice(0, 16) : "",
          acc: GEAR.has(c.acc) ? c.acc : "none",
        }))
    : DEFAULT_CREW.crew;
  return {
    crew: crew.length ? crew : DEFAULT_CREW.crew.slice(0, 1),
    show: typeof value.show === "boolean" ? value.show : true,
    roam: typeof value.roam === "boolean" ? value.roam : true,
  };
}

export function readCrew(): CrewState {
  try {
    const raw = localStorage.getItem(CREW_STORAGE_KEY);
    return raw ? sanitize(JSON.parse(raw)) : DEFAULT_CREW;
  } catch {
    return DEFAULT_CREW;
  }
}

interface CrewStore extends CrewState {
  update: (patch: Partial<CrewState>) => void;
  setMember: (slot: number, member: CrewMember) => void;
  patchMember: (slot: number, patch: Partial<CrewMember>) => void;
  removeMember: (slot: number) => void;
}

const persist = (state: CrewState) => {
  try { localStorage.setItem(CREW_STORAGE_KEY, JSON.stringify({ crew: state.crew, show: state.show, roam: state.roam })); } catch {}
};

export const useCrew = create<CrewStore>((set, get) => ({
  ...readCrew(),
  update: (patch) => { set(patch); persist(get()); },
  setMember: (slot, member) => {
    const crew = [...get().crew];
    crew[Math.min(slot, crew.length)] = member;
    set({ crew: crew.slice(0, 2) });
    persist(get());
  },
  patchMember: (slot, patch) => {
    const crew = get().crew.map((c, i) => (i === slot ? { ...c, ...patch } : c));
    set({ crew });
    persist(get());
  },
  removeMember: (slot) => {
    if (get().crew.length <= 1) return;
    set({ crew: get().crew.filter((_, i) => i !== slot) });
    persist(get());
  },
}));

/* ── reactions ── */

export type PetReaction = "keep" | "wild" | "scan" | "unlock" | "pet";
export interface PetEvent { what: PetReaction; text?: string }

type Listener = (event: PetEvent) => void;
const listeners = new Set<Listener>();

/** Tell the crew something happened. Together they react; only one talks. */
export function petsReact(what: PetReaction, text?: string) {
  listeners.forEach((listener) => listener({ what, text }));
}

export function onPetEvent(listener: Listener) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** A twin starts on a different look so the two can be told apart. */
export function lookForNewMember(crew: CrewMember[], slot: number, k: PetKind) {
  const twin = crew.find((c, i) => i !== slot && c.k === k);
  return twin ? (twin.look + 1) % PET_LOOKS[k].length : 0;
}
