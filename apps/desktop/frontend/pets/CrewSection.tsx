// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/** Settings > Your crew: two slots, a roster of twelve, looks, names and gear. */
import { useState } from "react";
import { StudioSwitch } from "../components/StudioControls";
import { Lock, Plus, X } from "../lib/icons";
import { PETS, PET_GEAR, PET_LOOKS, PET_UNLOCK_2, lookSwatch, petDef, petLook, petName, petSvg, type PetKind } from "./petArt";
import { lookForNewMember, petsReact, useCrew } from "./petStore";

export function CrewSection({ keepers }: { keepers: number }) {
  const { crew, show, roam, update, setMember, patchMember, removeMember } = useCrew();
  const [slot, setSlot] = useState(0);
  const two = keepers >= PET_UNLOCK_2;

  const pick = (k: PetKind) => {
    let target = slot;
    if (target === 1 && !two) target = 0;
    if (target > crew.length) target = crew.length;
    setMember(target, { k, look: lookForNewMember(crew, target, k), name: "", acc: crew[target]?.acc ?? "none" });
    setSlot(target);
    window.setTimeout(() => petsReact("pet"), 250);
  };

  const target = crew[slot] ? `Pick someone to swap in for ${petName(crew[slot])}` : `Pick someone for slot ${slot + 1}`;

  return (
    <>
      <div className="crew-slots">
        {[0, 1].map((i) => {
          const member = crew[i];
          if (i === 1 && !two) {
            return (
              <div key={i} className="crew-slot locked">
                <span className="cs-lock" aria-hidden="true"><Lock size={22} /></span>
                <div>
                  <b>Second slot</b>
                  <small>Opens at {PET_UNLOCK_2} keepers. You're at <span className="num">{keepers}</span>.</small>
                  <span className="bar" aria-hidden="true"><i style={{ transform: `scaleX(${Math.min(1, keepers / PET_UNLOCK_2)})` }} /></span>
                </div>
              </div>
            );
          }
          if (!member) {
            return (
              <button key={i} type="button" className={`crew-slot empty ${slot === i ? "on" : ""}`} onClick={() => setSlot(i)}>
                <span className="cs-add" aria-hidden="true"><Plus size={20} /></span>
                <div><b>Add a friend</b><small>Pick anyone from the roster below, even another {petDef(crew[0].k)[1].toLowerCase()}. They'll keep {petName(crew[0])} company.</small></div>
              </button>
            );
          }
          const looks = PET_LOOKS[member.k];
          return (
            <div key={i} className={`crew-slot ${slot === i ? "on" : ""}`} onClick={() => setSlot(i)} role="group" aria-label={`Crew slot ${i + 1}: ${petName(member)}`}>
              <div className="cs-pet" dangerouslySetInnerHTML={{ __html: petSvg(member.k, member.acc, member.look) }} />
              <div className="cs-body">
                <div className="cs-row">
                  <input
                    className="field cs-name"
                    value={member.name || petLook(member.k, member.look).name}
                    maxLength={16}
                    aria-label={`Name for slot ${i + 1}`}
                    onChange={(event) => patchMember(i, { name: event.target.value })}
                  />
                  <span className="t3 cs-kind">{petLook(member.k, member.look).n} {petDef(member.k)[1].toLowerCase()}</span>
                  <span className="sp" />
                  {crew.length > 1 && (
                    <button type="button" className="icon-btn" aria-label={`Remove ${petName(member)}`} onClick={(event) => { event.stopPropagation(); removeMember(i); setSlot(0); }}>
                      <X size={14} aria-hidden="true" />
                    </button>
                  )}
                </div>
                <div className="cs-row">
                  <span className="cs-lab">Look</span>
                  <div className="cs-looks" role="radiogroup" aria-label="Look">
                    {looks.map((look, j) => (
                      <button
                        key={look.n}
                        type="button"
                        role="radio"
                        aria-checked={member.look === j}
                        aria-label={look.n}
                        title={look.n}
                        className={`look-dot ${member.look === j ? "on" : ""}`}
                        style={{ background: lookSwatch(look) }}
                        onClick={(event) => {
                          event.stopPropagation();
                          const renamed = member.name.trim() && member.name !== petLook(member.k, member.look).name;
                          patchMember(i, { look: j, name: renamed ? member.name : "" });
                        }}
                      />
                    ))}
                  </div>
                </div>
                <div className="cs-row">
                  <span className="cs-lab">Gear</span>
                  <div className="cs-accs">
                    {PET_GEAR.map(([gear, label, need]) =>
                      keepers >= need ? (
                        <button key={gear} type="button" className="chip" aria-pressed={member.acc === gear} onClick={(event) => { event.stopPropagation(); patchMember(i, { acc: gear }); }}>{label}</button>
                      ) : (
                        <span key={gear} className="chip locked" title={`Unlocks at ${need} keepers`}>{label} · {need}</span>
                      ),
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="crew-roster-h"><b>Roster</b><small className="t3">{target}. Two of the same is fine.</small></div>
      <div className="crew-roster">
        {PETS.map(([k, label]) => {
          const n = crew.filter((c) => c.k === k).length;
          return (
            <button key={k} type="button" className={`roster-pet ${n ? "in" : ""}`} onClick={() => pick(k)} aria-label={`${label}, ${PET_LOOKS[k].length} looks`}>
              <span className="pet" dangerouslySetInnerHTML={{ __html: petSvg(k) }} />
              <small>{label}</small>
              {n > 1 && <span className="rp-n num">×2</span>}
              <span className="rp-looks">{PET_LOOKS[k].length} looks</span>
            </button>
          );
        })}
      </div>

      <div className="setrow">
        <div className="st"><b>Show my crew</b><small>Turn them off and they'll wait for you here.</small></div>
        <StudioSwitch checked={show} onChange={(value) => update({ show: value })} label="Show my crew" />
      </div>
      <div className="setrow">
        <div className="st"><b>Wander during scans</b><small>They come out along the bottom of the window while something is scanning, and stay home in Review and the Cutting Room.</small></div>
        <StudioSwitch checked={roam} onChange={(value) => update({ roam: value })} label="Wander during scans" />
      </div>
    </>
  );
}
