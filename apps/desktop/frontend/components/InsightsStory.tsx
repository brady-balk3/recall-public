// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
import { useState } from "react";
import { BarChart3, Check, Clock, Film, Scissors, Target } from "../lib/icons";
import { clueLabel, learningStateDescription, learningStateLabel } from "../lib/copy";
import { fmtClock } from "../lib/format";
import { type Clip, type Session, useJobStore } from "../lib/store";
import StudioEmptyState from "./StudioEmptyState";

type Props = { sessions: Session[]; totalClips: number; totalKept: number; keepRate: number; onStart: () => void };
const recommendationRankOf = (clip: Clip) => Math.max(0, Math.min(99, clip.deckScore ?? clip.confidence ?? 0));
const durationOf = (clip: Clip) => Math.max(0, clip.duration || ((clip.end_time ?? 0) - (clip.start_time ?? 0)));

export default function InsightsStory({ sessions, totalClips, totalKept, keepRate, onStart }: Props) {
  const { learningStatus } = useJobStore();
  const completed = sessions.filter((session) => session.status === "completed");
  const allClips = sessions.flatMap((session) => session.clips);
  const reviewedSeconds = sessions.reduce((sum, session) => sum + (session.vodDuration ?? 0), 0);
  const exported = sessions.reduce((sum, session) => sum + session.exportedClipIds.length, 0);
  const keptSeconds = sessions.reduce((sum, session) => sum + session.clips.filter((clip) => session.savedClipIds.includes(clip.id)).reduce((clipSum, clip) => clipSum + durationOf(clip), 0), 0);
  const keptClips = sessions.flatMap((session) => session.clips.filter((clip) => session.savedClipIds.includes(clip.id)));
  const strongest = [...keptClips, ...allClips].sort((a, b) => recommendationRankOf(b) - recommendationRankOf(a))[0];
  const latest = [...completed].sort((a, b) => b.updatedAt - a.updatedAt)[0];
  const first = [...completed].sort((a, b) => a.createdAt - b.createdAt)[0];
  const momentCounts = new Map<string, number>();
  allClips.forEach((clip) => { const key = (clip.momentType || clip.reason || "Other").split("+")[0].trim() || "Other"; momentCounts.set(key, (momentCounts.get(key) ?? 0) + 1); });
  const signalCounts = new Map<string, number>();
  allClips.flatMap((clip) => clip.signals ?? []).forEach((signal) => signalCounts.set(signal, (signalCounts.get(signal) ?? 0) + 1));
  const strongestMoment = [...momentCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "No clear pattern yet";
  const topSignal = [...signalCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  const patternParts = [strongestMoment !== "Other" ? strongestMoment : null, topSignal ? clueLabel(topSignal) : null].filter(Boolean) as string[];
  const learnedPattern = patternParts.filter((part, index) => patternParts.findIndex((candidate) => candidate.toLowerCase() === part.toLowerCase()) === index).join(" + ") || "Recall needs more decisions";
  const learningCount = learningStatus?.label_count ?? learningStatus?.labels ?? totalKept;
  const learningGoal = learningStatus?.min_labels ?? 25;
  const learningLabel = completed.length < 2 ? "Early pattern" : learningStateLabel(learningStatus?.state);
  const dateLabel = (value?: number) => value ? new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "Not yet";
  const story = [
    first && { icon: Film, tone: "accent", date: `Since ${dateLabel(first.createdAt)}`, title: "Your Recall journey began", copy: `${completed.length} VOD${completed.length === 1 ? " has" : "s have"} now been analyzed in your library.` },
    latest && { icon: Target, tone: "accent", date: "Across your library", title: `${totalClips} potential moment${totalClips === 1 ? "" : "s"} surfaced`, copy: "Recall has continuously narrowed your footage into moments worth reviewing." },
    totalKept > 0 && { icon: Check, tone: "success", date: "Your creative decisions", title: `You kept ${totalKept} clip${totalKept === 1 ? "" : "s"}`, copy: strongest ? `${strongest.title} is one of the clearest signals of what you value.` : "Your keep decisions are shaping future ranking." },
    exported > 0 && { icon: Scissors, tone: "accent", date: "Finished work", title: `You exported ${exported} clip${exported === 1 ? "" : "s"}`, copy: "Recall prepared your selected moments for vertical short-form publishing." },
    learningCount > 0 && { icon: BarChart3, tone: "learning", date: "Your evolving profile", title: "Recall formed an emerging preference", copy: `${learnedPattern} is appearing most often in your strongest decisions over time.` },
  ].filter(Boolean) as { icon: typeof Film; tone: string; date: string; title: string; copy: string }[];

  if (!completed.length) return <div className="story-insights-empty"><StudioEmptyState variant="insights" icon={<BarChart3 />} title="Your Recall story starts with one scan" body="After you review a VOD, this view connects the moments Recall found with the creative decisions you made." action={<button type="button" className="cta-accent" onClick={onStart}><Film size={14} /><span>Scan your first VOD</span></button>} /></div>;

  return <div className="story-insights">
    <header className="story-insights-toolbar"><p>Your cumulative story with Recall—what you have made, kept, and taught it over time.</p><button type="button"><Clock size={13} />All time<span>⌄</span></button></header>
    <p className="story-impact">Recall has reviewed <b>{fmtClock(reviewedSeconds)}</b> of footage and helped you turn <b>{totalClips} moment{totalClips === 1 ? "" : "s"}</b> into <b>{exported} ready-to-post clip{exported === 1 ? "" : "s"}.</b></p>
    <div className="story-flow" aria-label="Recall workflow impact"><FlowStep icon={Film} value={fmtClock(reviewedSeconds)} label="Footage reviewed" /><i>→</i><FlowStep icon={Target} value={String(totalClips)} label="Moments found" tone="accent" /><i>→</i><FlowStep icon={Check} value={String(totalKept)} label="Clips kept" tone="success" /><i>→</i><FlowStep icon={Scissors} value={String(exported)} label="Clips exported" tone="accent" /></div>
    <SessionYieldChart sessions={completed} />
    <div className="story-insights-board">
      <section className="story-history"><div className="story-section-head"><h2>Your Recall story</h2><p>The meaningful steps in your creative workflow.</p></div><div className="story-timeline">{story.map(({ icon: Icon, tone, date, title, copy }) => <article className={`story-chapter tone-${tone}`} key={title}><span><Icon size={15} /></span><time>{date}</time><strong>{title}</strong><p>{copy}</p></article>)}</div></section>
      <aside className="story-learning"><section><div className="story-section-head"><h2>What Recall is learning</h2></div><span className="story-confidence">{learningLabel} · {learningCount} of {learningGoal} decisions</span><div className="story-pattern"><Target size={20} /><div><h3>{learnedPattern}</h3><p>Your strongest decisions are beginning to favor this combination.</p></div></div><h4>Why Recall believes this</h4><ul><li>Your kept clips carry more weight than found moments.</li><li>Exports confirm which edits you considered finished.</li><li>Trims and titles show how you shape the final story.</li></ul></section><div className="story-working"><strong>What is working</strong><b>{strongestMoment}</b><span>{strongest ? `${strongest.title} is one of Recall's strongest recommendations from its session.` : "Keep reviewing to reveal a dependable pattern."}</span></div><div className="story-next"><strong>Try next</strong><b>{completed.length < 2 ? "Review one more varied session" : "Keep challenging the pattern"}</b><span>{completed.length < 2 ? "Different footage will show whether this preference holds." : "Varied sessions help Recall avoid narrowing your style too far."}</span></div><p className="story-caveat">{completed.length < 2 ? "Recall is still learning. One more varied session will make this more reliable." : learningStateDescription(learningStatus?.state)}</p></aside>
    </div>
    <footer className="story-ledger"><div><Clock size={16} /><span><b>Your impact so far</b><small>Real work completed with Recall</small></span></div><Ledger value={fmtClock(reviewedSeconds)} label="reviewed" /><Ledger value={String(totalClips)} label="moments found" /><Ledger value={`${keepRate}%`} label="keep rate" /><Ledger value={fmtClock(keptSeconds)} label="highlight runtime" /><Ledger value={String(exported)} label="exports" tone="success" /></footer>
  </div>;
}

function SessionYieldChart({ sessions }: { sessions: Session[] }) {
  const recent = [...sessions].sort((a, b) => a.updatedAt - b.updatedAt).slice(-8).map((session) => {
    const clipIds = new Set(session.clips.map((clip) => clip.id));
    return {
      id: session.id,
      name: session.name,
      date: new Date(session.updatedAt).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
      found: session.clips.length,
      kept: session.savedClipIds.filter((id) => clipIds.has(id)).length,
      exported: session.exportedClipIds.filter((id) => clipIds.has(id)).length,
    };
  });
  const [activeId, setActiveId] = useState<string | null>(recent[recent.length - 1]?.id ?? null);
  const active = recent.find((item) => item.id === activeId) ?? recent[recent.length - 1];
  if (!recent.length) return null;

  const width = 720;
  const height = 220;
  const left = 34;
  const bottom = 27;
  const top = 12;
  const plotHeight = height - bottom - top;
  const plotWidth = width - left - 8;
  const maxValue = Math.max(1, ...recent.flatMap((item) => [item.found, item.kept, item.exported]));
  const ceiling = Math.max(4, Math.ceil(maxValue / 4) * 4);
  const groupWidth = plotWidth / recent.length;
  const barWidth = Math.max(6, Math.min(17, (groupWidth - 18) / 3));
  const barGap = 3;
  const barY = (value: number) => top + plotHeight - value / ceiling * plotHeight;
  const barH = (value: number) => value / ceiling * plotHeight;

  return <section className="story-yield" aria-labelledby="story-yield-title">
    <header><div><h2 id="story-yield-title">Output across recent sessions</h2><p>Found, kept, and exported counts from your completed work.</p></div><div className="story-yield-legend"><span className="is-found"><i />Found</span><span className="is-kept"><i />Kept</span><span className="is-exported"><i />Exported</span></div></header>
    <div className="story-yield-body">
      <div className="story-yield-plot">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Output counts across recent completed sessions">
          {[0, 1, 2, 3, 4].map((step) => { const value = ceiling / 4 * step; const y = barY(value); return <g key={step}><line x1={left} x2={width - 8} y1={y} y2={y} /><text x={left - 8} y={y + 3}>{Math.round(value)}</text></g>; })}
          {recent.map((item, index) => {
            const center = left + groupWidth * index + groupWidth / 2;
            const start = center - (barWidth * 3 + barGap * 2) / 2;
            const selected = item.id === active?.id;
            return <g key={item.id} className={selected ? "is-active" : ""} tabIndex={0} role="img" aria-label={`${item.name}: ${item.found} found, ${item.kept} kept, ${item.exported} exported`} onMouseEnter={() => setActiveId(item.id)} onFocus={() => setActiveId(item.id)}>
              <rect className="is-hit" x={left + groupWidth * index + 2} y={top} width={groupWidth - 4} height={plotHeight} rx={5} />
              <rect className="is-found" x={start} y={barY(item.found)} width={barWidth} height={barH(item.found)} rx={3} />
              <rect className="is-kept" x={start + barWidth + barGap} y={barY(item.kept)} width={barWidth} height={barH(item.kept)} rx={3} />
              <rect className="is-exported" x={start + (barWidth + barGap) * 2} y={barY(item.exported)} width={barWidth} height={barH(item.exported)} rx={3} />
              <text className="is-date" x={center} y={height - 7}>{item.date}</text>
            </g>;
          })}
        </svg>
      </div>
      {active && <aside aria-live="polite"><span>{active.date}</span><strong>{active.name}</strong><div><b>{active.found}<small>found</small></b><b>{active.kept}<small>kept</small></b><b>{active.exported}<small>exported</small></b></div></aside>}
    </div>
  </section>;
}

function FlowStep({ icon: Icon, value, label, tone = "neutral" }: { icon: typeof Film; value: string; label: string; tone?: string }) {
  return <div className={`story-flow-step tone-${tone}`}><span><Icon size={16} /></span><div><b>{value}</b><small>{label}</small></div></div>;
}

function Ledger({ value, label, tone }: { value: string; label: string; tone?: string }) {
  return <span className={tone ? `tone-${tone}` : ""}><b>{value}</b><small>{label}</small></span>;
}
