// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Synthetic library for UI work without a backend or any real footage.
 *
 * Dev only: main.tsx loads this when the URL has `?fixtures`, behind
 * `import.meta.env.DEV`, so it never reaches a production bundle. Every name,
 * number and image here is made up; thumbnails are generated SVG gradients.
 */
import { useJobStore, type Clip, type Job, type Session } from "../lib/store";

const TITLES = [
  "Late night ranked",
  "Co-op chaos with the squad",
  "Sunday just chatting",
  "Speedrun practice, again",
  "Horror night part 3",
  "Customs with viewers",
  "Road to diamond",
  "Cozy farm stream",
];

const HOOKS = [
  "No way that just worked",
  "Chat called it first",
  "The one-tap nobody expected",
  "Wait, who left the door open",
  "Clutch in the last second",
  "That laugh took a minute",
  "Accidental genius",
  "The jump scare that got me",
  "Three for one",
  "We do not talk about this",
];

/** A 9:16 frame: a two-hue gradient with a soft glow, like a blurred game still. */
function frame(seed: number, label: string) {
  const hue = (seed * 47) % 360;
  const hue2 = (hue + 40 + (seed % 3) * 25) % 360;
  const cx = 30 + ((seed * 13) % 40);
  const cy = 25 + ((seed * 7) % 45);
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='360' height='640' viewBox='0 0 360 640'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='hsl(${hue} 55% 42%)'/><stop offset='1' stop-color='hsl(${hue2} 60% 18%)'/></linearGradient>
<radialGradient id='r' cx='${cx}%' cy='${cy}%' r='55%'><stop offset='0' stop-color='hsl(${(hue + 180) % 360} 80% 70%)' stop-opacity='.55'/><stop offset='1' stop-color='#000' stop-opacity='0'/></radialGradient></defs>
<rect width='360' height='640' fill='url(#g)'/><rect width='360' height='640' fill='url(#r)'/>
<rect y='420' width='360' height='220' fill='#000' opacity='.25'/>
<text x='24' y='60' font-family='system-ui,sans-serif' font-size='34' font-weight='800' fill='#fff' opacity='.9'>${label}</text>
</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

/**
 * A synthetic 9:16 test-pattern MP4 for the in-app editor, generated locally
 * (gitignored; see frontend/dev/media). Clips just lack playable media without it:
 *   ffmpeg -f lavfi -i testsrc2=size=1080x1920:rate=30:duration=12 -f lavfi -i sine=frequency=330:duration=12  *     -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest frontend/dev/media/fixture-clip.mp4
 */
const FIXTURE_VIDEO = "/frontend/dev/media/fixture-clip.mp4";

function makeClip(sessionIndex: number, clipIndex: number): Clip {
  const seed = sessionIndex * 17 + clipIndex * 5 + 3;
  const start = 600 + clipIndex * 740 + (seed % 90);
  const duration = 18 + (seed % 27);
  const confidence = Math.max(42, 96 - clipIndex * 4 - (seed % 9));
  return {
    id: `fx-${sessionIndex}-${clipIndex}`,
    title: HOOKS[(sessionIndex + clipIndex) % HOOKS.length],
    clipNumber: clipIndex + 1,
    timestamp: new Date(start * 1000).toISOString().slice(11, 19),
    confidence,
    deckScore: confidence,
    duration,
    start_time: start,
    end_time: start + duration,
    thumbUrl: frame(seed, `#${String(clipIndex + 1).padStart(2, "0")}`),
    hookLine: HOOKS[(sessionIndex + clipIndex) % HOOKS.length],
    videoUrl: FIXTURE_VIDEO,
    mediaState: "ready",
  } as Clip;
}

function makeSession(index: number, now: number): Session {
  const clipCount = 9 + ((index * 5) % 7);
  const clips = Array.from({ length: clipCount }, (_, c) => makeClip(index, c));
  // Older sessions are further along: more decided, more kept, more exported.
  const decided = index === 0 ? 0 : Math.min(clipCount, 3 + index * 2);
  const saved: string[] = [];
  clips.slice(0, decided).forEach((clip, c) => {
    if (c % 3 === 2) clip.passed = true;
    else {
      clip.kept = true;
      saved.push(clip.id);
    }
  });
  const exported = index > 3 ? saved.slice(0, Math.ceil(saved.length / 2)) : [];
  const createdAt = now - (index + 1) * 26 * 3_600_000;
  return {
    id: `fx-session-${index}`,
    name: TITLES[(index + 1) % TITLES.length],
    sourceUrl: "",
    createdAt,
    updatedAt: createdAt,
    status: "completed",
    savedClipIds: saved,
    exportedClipIds: exported,
    clips,
    clipsHydrated: true,
    vodDuration: 3 * 3600 + ((index * 1234) % 7200),
  } as Session;
}

export function seedFixtures() {
  const now = Date.now();
  const sessions = TITLES.slice(1).map((_, i) => makeSession(i, now));
  const scanning: Session = {
    id: "fx-scanning",
    name: TITLES[0],
    sourceUrl: "",
    createdAt: now - 20 * 60_000,
    updatedAt: now,
    status: "detecting",
    savedClipIds: [],
    exportedClipIds: [],
    clips: [],
  } as Session;
  // Mid "Find moments": the whole VOD read, a first-pass curve in, three
  // moments pinned, so the live strip has something real-shaped to draw.
  const vod = 3 * 3600 + 26 * 60 + 16;
  const t = Array.from({ length: 400 }, (_, i) => (i / 399) * vod);
  const peaks = [1780, 4085, 5190, 7010, 9900, 11800];
  const r = t.map((x, i) => 0.12 + 0.08 * ((i * 37) % 11) / 11 + peaks.reduce((sum, c, k) => sum + (0.45 + k * 0.09) * Math.exp(-(((x - c) / 80) ** 2)), 0));
  const steps = [
    "Downloading Twitch VOD: 100%",
    "Recognized Fortnite from Twitch's VOD info...",
    "Transcribing audio · 1:12:40 / 3:26:16 · 20.4×",
    "Transcribing audio · 3:26:16 / 3:26:16 · 20.8×",
    "Reading Twitch chat reactions...",
    "Scoring reactions and selecting clips...",
    "Local vision model is checking each candidate...",
  ];
  const job: Job = {
    id: "fx-scanning",
    url: "https://www.twitch.tv/videos/1",
    status: "detecting",
    progress: 67,
    stage: { label: "Finding exciting moments", progress: 67 },
    createdAt: now - 20 * 60_000,
    currentPhase: "Reaction",
    message: "Local vision model is checking each candidate...",
    vodDuration: vod,
    scannedSeconds: vod,
    transcribedSeconds: vod,
    transcriptionSpeed: 20.8,
    elapsedSeconds: 20 * 60,
    elapsedSyncedAt: now,
    etaSeconds: 11 * 60,
    etaSyncedAt: now,
    etaConfidence: "medium",
    totalWorkers: 6,
    activeWorkers: 1,
    scanHealth: { device: "cuda", visual: { status: "healthy" } },
    events: steps.map((message, i) => ({ jobId: "fx-scanning", eventType: "progress", message, at: now - (steps.length - i) * 150_000 })),
    liveTimeline: { version: 2, t, r, game_markers: [], preliminary: true, segments: [{ start: 0, end: 7820, game: "Fortnite" }, { start: 7820, end: 9050, game: "Just Chatting" }, { start: 9050, end: vod, game: "Fortnite" }] },
    liveLanes: {
      version: 1,
      duration: vod,
      speech: Array.from({ length: 240 }, (_, i) => (i % 17 === 0 ? 0.05 : 0.35 + 0.6 * Math.abs(Math.sin(i * 0.37)) ** 2)),
      chat: Array.from({ length: 240 }, (_, i) => Math.min(1, 0.12 + 0.1 * Math.abs(Math.sin(i * 1.3)) + peaks.reduce((sum, c) => sum + 0.8 * Math.exp(-((((i / 240) * vod - c) / 160) ** 2)), 0))),
      chat_scope: "full_vod",
      speech_regions: null,
    },
    candidates: peaks.concat([2600, 6100, 8400, 10500]).sort((a, b) => a - b).map((c, i) => ({
      start: c - 20,
      end: c + 20,
      status: i < 3 ? (i === 1 ? "post" : i === 2 ? "skip" : "post") : i === 3 ? "checking" : "waiting",
    })),
    foundClips: [
      { clipId: "fx-f1", title: HOOKS[0], timestamp: "0:29:30", duration: 32, score: 82 },
      { clipId: "fx-f2", title: HOOKS[1], timestamp: "1:07:55", duration: 28, score: 74 },
      { clipId: "fx-f3", title: HOOKS[4], timestamp: "1:26:20", duration: 36, score: 77, pending: true, captioned: true },
      { clipId: "fx-f4", title: "New moment", timestamp: "1:56:50", duration: 30, score: 71, pending: true },
    ],
  } as Job;
  useJobStore.setState((state) => ({
    sessions: [scanning, ...sessions],
    jobs: [job],
    settings: { ...state.settings, displayName: state.settings.displayName || "Sam" },
  }));
}

/** Dev-only triggers for the crew and the bigger beats (`window.recallDev` in the console). */
export async function exposeDevTriggers() {
  const [{ petsReact }, { useMoments }] = await Promise.all([import("../pets/petStore"), import("../moments/milestones")]);
  const clips = () => useJobStore.getState().sessions.flatMap((s) => s.clips.filter((c) => s.savedClipIds.includes(c.id)));
  (window as unknown as { recallDev: unknown }).recallDev = {
    react: petsReact,
    celebrate: (n = 50) => useMoments.getState().celebrate(n, clips().slice(0, 5)),
    recap: () => useMoments.getState().openRecap(),
  };
}
