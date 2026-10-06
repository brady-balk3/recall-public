// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brady Balk
/**
 * Icon set: Phosphor duotone for content glyphs, Phosphor bold for the
 * single-stroke connective chrome (plus, carets, check, x, arrows), where a
 * duotone fill has nothing to fill and reads as a thin line.
 *
 * Every export keeps the name and prop shape (`size`, `className`, `color`,
 * `style`, `aria-*`, ...) the app has always used, so call sites never change
 * when the underlying set does.
 *
 * The window controls (minimize, maximize) stay hand-drawn line strokes: they
 * are OS chrome, not app vocabulary, and should look like the OS's.
 *
 * Never map anything to a sparkle glyph. "Sparkles" is kept as a name for old
 * call sites and renders a lightning bolt.
 *
 * Phosphor Icons, MIT. Bundled, so the packaged app works offline.
 */
import { forwardRef, type ComponentType, type SVGProps } from "react";
import {
  ArrowClockwiseIcon,
  ArrowCounterClockwiseIcon,
  ArrowLeftIcon,
  ArrowsDownUpIcon,
  BooksIcon,
  BugIcon,
  CaretDownIcon,
  CaretLeftIcon,
  CaretRightIcon,
  ChartBarIcon,
  CheckCircleIcon,
  CheckIcon,
  ClockIcon,
  ClosedCaptioningIcon,
  CopyIcon,
  DownloadSimpleIcon,
  FileTextIcon,
  FilmStripIcon,
  FloppyDiskIcon,
  FolderOpenIcon,
  GearSixIcon,
  HardDrivesIcon,
  HouseIcon,
  InfoIcon,
  KeyboardIcon,
  LightningIcon,
  MagnifyingGlassIcon,
  PauseIcon,
  PlayIcon,
  PlusIcon,
  RepeatIcon,
  ScissorsIcon,
  SpeakerHighIcon,
  SpeakerXIcon,
  TargetIcon,
  TrashIcon,
  TrophyIcon,
  VideoCameraIcon,
  WarningIcon,
  XIcon,
  PulseIcon,
  PlayCircleIcon,
  ExportIcon as ExportIconGlyph,
  BookOpenIcon,
  BroadcastIcon,
  SquaresFourIcon,
  StackIcon,
  QueueIcon,
  TwitchLogoIcon,
  LinkIcon as LinkGlyph,
  FolderIcon,
  LockSimpleIcon,
  PawPrintIcon,
  type IconProps as PhosphorProps,
  type IconWeight,
} from "@phosphor-icons/react";

export type IconProps = Omit<SVGProps<SVGSVGElement>, "ref"> & {
  /** Pixel size for width & height. Default 24. */
  size?: number | string;
  /** Accepted for compatibility with older call sites; Phosphor has fixed weights. */
  strokeWidth?: number | string;
  /** Accepted for compatibility with older call sites; no-op. */
  absoluteStrokeWidth?: boolean;
};

const phosphor = (Glyph: ComponentType<PhosphorProps>, weight: IconWeight, name: string) => {
  const Icon = forwardRef<SVGSVGElement, IconProps>(function RecallIcon(
    { size = 24, strokeWidth: _strokeWidth, absoluteStrokeWidth: _absolute, ...rest },
    ref,
  ) {
    return <Glyph ref={ref} size={size} weight={weight} {...(rest as PhosphorProps)} />;
  });
  Icon.displayName = name;
  return Icon;
};
const duotone = (Glyph: ComponentType<PhosphorProps>, name: string) => phosphor(Glyph, "duotone", name);
const bold = (Glyph: ComponentType<PhosphorProps>, name: string) => phosphor(Glyph, "bold", name);

/* ── Connective chrome: Phosphor bold ── */
export const ArrowLeft = bold(ArrowLeftIcon, "ArrowLeft");
export const ChevronDown = bold(CaretDownIcon, "ChevronDown");
export const ChevronLeft = bold(CaretLeftIcon, "ChevronLeft");
export const ChevronRight = bold(CaretRightIcon, "ChevronRight");
export const X = bold(XIcon, "X");
export const Plus = bold(PlusIcon, "Plus");
export const Check = bold(CheckIcon, "Check");

/* ── Window controls: OS-style line strokes ── */
const lineIcon = (markup: string, name: string) => {
  const Icon = forwardRef<SVGSVGElement, IconProps>(function WindowIcon(
    { size = 24, strokeWidth: _strokeWidth, absoluteStrokeWidth: _absolute, width, height, ...rest },
    ref,
  ) {
    return (
      <svg
        ref={ref}
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        width={width ?? size}
        height={height ?? size}
        dangerouslySetInnerHTML={{ __html: markup }}
        {...rest}
      />
    );
  });
  Icon.displayName = name;
  return Icon;
};
const stroke = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
export const Minus = lineIcon(`<path ${stroke} d="M5 12h14"/>`, "Minus");
export const Maximize2 = lineIcon(`<rect x="4.5" y="4.5" width="15" height="15" rx="2" ${stroke}/>`, "Maximize2");

/* ── Content: Phosphor duotone ── */
export const CheckCircle2 = duotone(CheckCircleIcon, "CheckCircle2");
export const Home = duotone(HouseIcon, "Home");
export const Film = duotone(FilmStripIcon, "Film");
export const Captions = duotone(ClosedCaptioningIcon, "Captions");
export const Scissors = duotone(ScissorsIcon, "Scissors");
export const Play = duotone(PlayIcon, "Play");
export const Pause = duotone(PauseIcon, "Pause");
export const Sparkles = duotone(LightningIcon, "Sparkles");
export const Trophy = duotone(TrophyIcon, "Trophy");
export const Target = duotone(TargetIcon, "Target");
export const Search = duotone(MagnifyingGlassIcon, "Search");
export const Settings = duotone(GearSixIcon, "Settings");
export const Library = duotone(BooksIcon, "Library");
export const BarChart3 = duotone(ChartBarIcon, "BarChart3");
export const Trash2 = duotone(TrashIcon, "Trash2");
export const Video = duotone(VideoCameraIcon, "Video");
export const Clock = duotone(ClockIcon, "Clock");
export const FolderOpen = duotone(FolderOpenIcon, "FolderOpen");
export const HardDrive = duotone(HardDrivesIcon, "HardDrive");
export const Download = duotone(DownloadSimpleIcon, "Download");
export const Copy = duotone(CopyIcon, "Copy");
export const Save = duotone(FloppyDiskIcon, "Save");
export const RefreshCw = duotone(ArrowClockwiseIcon, "RefreshCw");
export const Repeat = duotone(RepeatIcon, "Repeat");
export const RotateCcw = duotone(ArrowCounterClockwiseIcon, "RotateCcw");
export const Volume2 = duotone(SpeakerHighIcon, "Volume2");
export const VolumeX = duotone(SpeakerXIcon, "VolumeX");
export const Info = duotone(InfoIcon, "Info");
export const AlertTriangle = duotone(WarningIcon, "AlertTriangle");
export const ArrowUpDown = duotone(ArrowsDownUpIcon, "ArrowUpDown");
export const Bug = duotone(BugIcon, "Bug");
export const FileText = duotone(FileTextIcon, "FileText");
export const Keyboard = duotone(KeyboardIcon, "Keyboard");

/* ── Shell vocabulary ── */
export const Pulse = duotone(PulseIcon, "Pulse");
export const PlayCircle = duotone(PlayCircleIcon, "PlayCircle");
export const ExportIcon = duotone(ExportIconGlyph, "Export");
export const BookOpen = duotone(BookOpenIcon, "BookOpen");
export const Broadcast = duotone(BroadcastIcon, "Broadcast");
export const Grid = duotone(SquaresFourIcon, "Grid");
export const Layers = duotone(StackIcon, "Layers");
export const Queue = duotone(QueueIcon, "Queue");
export const Twitch = duotone(TwitchLogoIcon, "Twitch");
export const LinkIcon = duotone(LinkGlyph, "Link");
export const Folder = duotone(FolderIcon, "Folder");
export const Lock = duotone(LockSimpleIcon, "Lock");
export const Paw = duotone(PawPrintIcon, "Paw");
