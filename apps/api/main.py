# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
# Frozen workers execute this entry point as __main__. Dispatch them before
# importing application services: JobManager startup recovery would otherwise
# mark the parent process's live scan as interrupted.
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import sys
    if "--install-models" in sys.argv:
        if not getattr(sys, "frozen", False):
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from core.installer_models import main as install_models_main
        raise SystemExit(install_models_main())

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncio
import re
import sys
import os
import hmac
import math
import json as _json
import shutil
import threading
from pathlib import Path

# Ensure the root project path is available for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.database import DatabaseManager
from core.job_manager import JobManager
from core.recall_sessions import (
    RecallSessionConflict,
    RecallSessionError,
    RecallSessionNotFound,
    RecallSessionService,
)
from core.stream_memory import StreamMemoryService
from core.stream_memory_continuous import StreamMemoryContinuousIndexer
from core import compilation_recipes
from core.compilation_projects import (
    CompilationProjectConflict,
    CompilationProjectNotFound,
    CompilationProjectService,
)
from core.compilation_preparation import (
    CompilationPreparationConflict,
    CompilationPreparationService,
    CompilationPreparationStorageError,
)
from core.bundle_paths import get_data_dir, get_model_setup_dir, is_frozen
from core.model_setup import ModelSetupController, ModelSetupConflict, speech_plan, speech_groups
from core.model_activation import initialize_speech_models
from apps.api.services import (
    ClipService,
    ExportOperationService,
    JobService,
    LearningService,
    SettingsService,
    SystemService,
)

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: reconcile the unified Recall/source lifecycle. Guarded because the
    # service singletons aren't created in spawned worker processes (__mp_main__).
    svc = globals().get("system_service")
    if svc is not None:
        try:
            svc.storage()
        except Exception as exc:
            print(f"Storage lifecycle startup reconciliation skipped: {exc}")
    # Feedback may have accumulated while personalization was deliberately
    # frozen for an eval run. Do not wait for one more Keep/Pass click before
    # measuring it: rebuild the shadow challenger in the background as soon as
    # the normal product engine starts. It cannot affect selection until a
    # separate future promotion step validates it.
    learning = globals().get("learning_service")
    if learning is not None:
        try:
            status = learning.status()
            if (
                status.get("can_train")
                and not status.get("has_challenger")
                and not status.get("frozen")
            ):
                threading.Thread(
                    target=learning.maybe_train_background,
                    name="recall-personal-ranker-startup",
                    daemon=True,
                ).start()
        except Exception as exc:
            print(f"Personal ranker startup activation skipped: {exc}")
    memory_indexer = globals().get("stream_memory_continuous_indexer")
    if memory_indexer is not None:
        memory_indexer.start()
    try:
        yield
    finally:
        if memory_indexer is not None:
            memory_indexer.stop()


app = FastAPI(
    title="Recall Backend API",
    description="Headless service for Recall",
    lifespan=lifespan,
)

# The API binds 127.0.0.1 but is still reachable from any web page the user has
# open (a page can fetch http://127.0.0.1:8000/...). CORS headers only gate
# whether that page can *read* the response — a "simple" cross-origin POST still
# executes server-side, so CORS alone can't protect destructive endpoints
# (/system/wipe, /clips/copy-to-folder, /jobs/run, ...). The Origin guard below
# rejects the request itself before it runs.
#
# Legitimate callers: the packaged renderer loads over file:// (Origin "null"),
# the dev renderer is served by Vite on localhost, and non-browser clients
# (curl, the Electron main) send no Origin at all. Everything else is a real
# website and is refused.
_LOOPBACK_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")
_API_TOKEN = os.environ.get("RECALL_API_TOKEN", "").strip()


def _origin_allowed(origin: Optional[str]) -> bool:
    if origin is None:
        return True  # non-browser client; not a cross-site browser request
    if origin == "null":
        return True  # packaged file:// renderer
    return bool(_LOOPBACK_ORIGIN_RE.match(origin))


def _api_token_allowed(request: Request) -> bool:
    """Authenticate the Electron-owned API when a launch token is configured.

    Direct source launches intentionally remain tokenless for CLI/test workflows.
    Electron always supplies a random per-launch token and random loopback port.
    Query-string auth is supported because media elements and EventSource cannot
    attach an Authorization header.
    """
    if not _API_TOKEN:
        return True
    supplied = request.headers.get("x-recall-token") or request.query_params.get("token") or ""
    return hmac.compare_digest(supplied.encode("utf-8"), _API_TOKEN.encode("utf-8"))


@app.middleware("http")
async def guard_cross_origin(request: Request, call_next):
    if not _origin_allowed(request.headers.get("origin")):
        return JSONResponse(status_code=403, content={"detail": "Cross-origin request blocked"})
    if not _api_token_allowed(request):
        return JSONResponse(status_code=401, content={"detail": "Invalid Recall API token"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    # Echo back "null" for the packaged file:// renderer; match localhost (any
    # port) for the Vite dev server. No wildcard, and no credentials (the app
    # uses no cookies — the old allow_credentials=True + "*" was spec-invalid).
    allow_origins=["null"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Writable runtime data root. In dev this is <project_root>/data (same as
# project_root below); in the packaged exe it's a data/ folder next to the
# executable, since project_root's __file__-relative math doesn't resolve
# correctly once this module is frozen.
data_root = get_data_dir()

exports_dir = os.path.join(data_root, "exports")
os.makedirs(exports_dir, exist_ok=True)
# Rendered clip MP4s and their thumb_*.jpg posters. Served by an explicit route
# (not a StaticFiles mount) so
# the response carries Cache-Control: no-cache — clips are re-rendered in place
# (same filename) when edited or re-processed, and a fingerprint-cached video
# would otherwise keep playing the STALE render (old captions/framing) until the
# app restarted. no-cache lets the browser reuse bytes via ETag revalidation
# (304 when unchanged) while always picking up a changed file. FileResponse
# honors Range (206 + Content-Range) exactly like the StaticFiles mount did.
@app.get("/media/{filename}")
def media_file(filename: str):
    media_name = filename.lower()
    supported = media_name.endswith(".mp4") or (
        media_name.startswith("thumb_") and media_name.endswith(".jpg")
    )
    if (not supported
            or any(char in filename for char in '/\\:')
            or ".." in filename):
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(exports_dir, filename)
    # Sidecars are private application state. Also reject links/junctions that
    # would serve a file outside the export folder, including Windows aliases.
    resolved_root = os.path.realpath(exports_dir)
    resolved_path = os.path.realpath(path)
    try:
        inside = os.path.commonpath([resolved_root, resolved_path]) == resolved_root
    except ValueError:
        inside = False
    if not inside:
        raise HTTPException(status_code=404, detail="Not found")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, headers={"Cache-Control": "no-cache"})

# Skip the heavy singletons in parallel-perception worker processes: Windows
# `spawn` re-imports this entry module as "__mp_main__", which would otherwise
# re-run DB migrations in every worker (wasteful + SQLite lock races).
if __name__ != "__mp_main__":
    model_setup_root = Path(get_model_setup_dir()).resolve()
    speech_runtime_source = initialize_speech_models(
        model_setup_root, speech_plan()["plan_id"], speech_groups()
    )
    db = DatabaseManager()
    job_manager = JobManager(db)
    learning_service = LearningService(db)
    recall_session_service = RecallSessionService(db)
    stream_memory_service = StreamMemoryService(db)
    compilation_project_service = CompilationProjectService(db, stream_memory_service)
    compilation_preparation_service = CompilationPreparationService(
        db, compilation_project_service, job_manager
    )
    job_service = JobService(db, job_manager, recall_session_service)
    clip_service = ClipService(db, learning_service, job_manager)
    export_operation_service = ExportOperationService(clip_service)
    stream_memory_continuous_indexer = StreamMemoryContinuousIndexer(
        stream_memory_service,
        lambda: (
            job_manager.has_active_jobs()
            or export_operation_service.has_active_operations()
        ),
    )
    system_service = SystemService(db, job_manager)
    settings_service = SettingsService(db)
    model_setup_controller = ModelSetupController(
        str(model_setup_root), speech_runtime_source
    )

class DeleteClipsRequest(BaseModel):
    clip_ids: List[str] = Field(min_length=1, max_length=500)

class CopyClipsRequest(BaseModel):
    clip_ids: List[str] = Field(min_length=1, max_length=500)
    dest_folder: str = Field(min_length=1, max_length=4096)
    preset: Optional[str] = None
    filename_template: Optional[str] = None

class CompileReelRequest(BaseModel):
    clip_ids: List[str] = Field(min_length=2, max_length=500)
    dest_folder: str = Field(min_length=1, max_length=4096)
    filename: Optional[str] = None

class RunJobRequest(BaseModel):
    source_path: str
    source_type: str = "file"
    session_name: Optional[str] = None
    source_date: Optional[str] = None
    recall_session_id: Optional[str] = None
    vod_started_at_utc: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class RecallSessionStartRequest(BaseModel):
    source_platform: str = Field(default="unknown", min_length=1, max_length=32)
    source_ref: Optional[str] = Field(default=None, max_length=2048)
    title: Optional[str] = Field(default=None, max_length=200)
    started_at_utc: Optional[str] = None
    initial_stream_time_seconds: Optional[float] = Field(default=None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecallSessionStopRequest(BaseModel):
    ended_at_utc: Optional[str] = None


class RecallSessionEventRequest(BaseModel):
    kind: str = Field(default="remember", min_length=1, max_length=32)
    source: str = Field(default="manual", min_length=1, max_length=32)
    occurred_at_utc: Optional[str] = None
    stream_offset_seconds: Optional[float] = Field(default=None, ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    payload: Dict[str, Any] = Field(default_factory=dict)


class RecallSessionClockAnchorRequest(BaseModel):
    wall_time_utc: str
    stream_time_seconds: float = Field(ge=0)
    anchor_type: str = Field(default="player_position", min_length=1, max_length=32)
    uncertainty_seconds: float = Field(default=1.0, ge=0)
    source: str = Field(default="manual", min_length=1, max_length=32)


class RecallSessionRegionPlanRequest(BaseModel):
    vod_started_at_utc: Optional[str] = None
    vod_duration_seconds: Optional[float] = Field(default=None, ge=0)
    pre_roll_seconds: float = Field(default=90.0, ge=0, le=600)
    post_roll_seconds: float = Field(default=30.0, ge=0, le=600)

class DownloadVodRequest(BaseModel):
    url: str

class EditClipRequest(BaseModel):
    start_time: float
    end_time: float
    fade_in: float = 0.0
    fade_out: float = 0.0
    video_fade_in: Optional[float] = None
    video_fade_out: Optional[float] = None
    audio_fade_in: Optional[float] = None
    audio_fade_out: Optional[float] = None

    @model_validator(mode="after")
    def validate_times(self):
        values = [self.start_time, self.end_time, self.fade_in, self.fade_out]
        values.extend(v for v in (
            self.video_fade_in, self.video_fade_out,
            self.audio_fade_in, self.audio_fade_out,
        ) if v is not None)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Clip times and fades must be finite, non-negative numbers.")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time.")
        if self.end_time - self.start_time > 180:
            raise ValueError("Edited clips cannot be longer than 180 seconds.")
        return self

class CaptionEditRequest(BaseModel):
    # Empty text is meaningful: it clears the correction and restores the
    # machine decode. Length is capped well above any real caption so a
    # runaway paste cannot land a megabyte in the row that feeds every render.
    text: str = ""

    @model_validator(mode="after")
    def validate_text(self):
        if len(self.text) > 5000:
            raise ValueError("Caption text is too long for one clip.")
        return self

class ManualClipRequest(BaseModel):
    # Two creation shapes share this request. Legacy: a center `timestamp`
    # plus `duration` (review-timeline "clip this moment" affordance). Precise:
    # explicit `start`/`end` from the Cutting Room, used exactly as given.
    timestamp: Optional[float] = None
    duration: float = 30.0
    start: Optional[float] = None
    end: Optional[float] = None
    title: Optional[str] = Field(default=None, max_length=200)
    tags: List[str] = Field(default_factory=list, max_length=20)
    layout: str = "auto"
    focus_x: float = 0.5
    captions_enabled: Optional[bool] = None
    video_fade_in: float = 0.0
    video_fade_out: float = 0.0
    audio_fade_in: float = 0.0
    audio_fade_out: float = 0.0
    # Set only when the VOD Editor was opened from a Stream Memory result.
    # ClipService resolves this id against the local index and verifies that
    # the finished cut still overlaps the remembered source moment.
    memory_entry_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    memory_query: Optional[str] = Field(default=None, max_length=240)
    # "compilation" marks a machine cut made for a montage: it stays out of
    # Clip Library and never trains the ranker as a creator judgement.
    origin: Optional[str] = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_times(self):
        if self.layout not in {"auto", "vertical_split", "gameplay_pip", "full_gameplay"}:
            raise ValueError("layout must be auto, vertical_split, gameplay_pip, or full_gameplay.")
        if not math.isfinite(self.focus_x) or not 0.0 <= self.focus_x <= 1.0:
            raise ValueError("focus_x must be between 0 and 1.")
        fade_values = (
            self.video_fade_in, self.video_fade_out,
            self.audio_fade_in, self.audio_fade_out,
        )
        if any(not math.isfinite(value) or value < 0 for value in fade_values):
            raise ValueError("Fade durations must be finite, non-negative numbers.")
        if self.memory_query and not self.memory_entry_id:
            raise ValueError("memory_query requires memory_entry_id.")
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together.")
        if self.start is not None and self.end is not None:
            if any(not math.isfinite(v) or v < 0 for v in (self.start, self.end)):
                raise ValueError("start and end must be finite, non-negative numbers.")
            if self.end <= self.start:
                raise ValueError("end must be greater than start.")
            # Same ceiling as EditClipRequest — a "clip" longer than 3 minutes
            # is a re-upload, not a highlight.
            if self.end - self.start > 180:
                raise ValueError("Manual clips cannot be longer than 180 seconds.")
            return self
        if self.timestamp is None or not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite, non-negative number.")
        if not math.isfinite(self.duration) or not 5 <= self.duration <= 180:
            raise ValueError("duration must be between 5 and 180 seconds.")
        return self

class LabelClipRequest(BaseModel):
    # null retracts the matching event and returns the clip to neutral. This is
    # required for undoing Pass: not-passed is not the same thing as Kept.
    label: Optional[float] = Field(default=None, ge=0, le=1)
    event: str = "manual"

class UpdateClipRequest(BaseModel):
    title: Optional[str] = None
    kept: Optional[bool] = None
    passed: Optional[bool] = None
    maybe: Optional[bool] = None
    exported: Optional[bool] = None


class ClipStateBatchRequest(BaseModel):
    clip_ids: List[str] = Field(min_length=1, max_length=500)
    kept: Optional[bool] = None
    passed: Optional[bool] = None
    maybe: Optional[bool] = None
    exported: Optional[bool] = None


class ProbeRequest(BaseModel):
    source_path: str
    source_type: str = "file"


class PinSourceRequest(BaseModel):
    pinned: bool


class StreamMemoryReindexRequest(BaseModel):
    force: bool = False


class StreamMemoryAliasRequest(BaseModel):
    canonical_term: str = Field(min_length=1, max_length=80)
    alias: str = Field(min_length=1, max_length=80)


class StreamMemoryFeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=240)
    entry_id: str = Field(min_length=1, max_length=100)
    verdict: str = Field(pattern="^(relevant|not_relevant|clear)$")


class StreamMemoryInteractionEntryRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=100)


class StreamMemoryInteractionRequest(BaseModel):
    search_id: str = Field(min_length=1, max_length=100)
    event_type: str = Field(pattern="^(impression|open)$")
    entries: List[StreamMemoryInteractionEntryRequest] = Field(min_length=1, max_length=100)


class StreamMemoryMissedQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=240)
    expected_description: str = Field(min_length=1, max_length=500)


class StreamMemoryEvaluationTargetRequest(BaseModel):
    entry_id: str = Field(min_length=1, max_length=100)


class GenerateCompilationProjectRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    query: str = Field(default="", max_length=240)
    game: Optional[str] = Field(default=None, max_length=120)
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    max_clips: int = Field(default=12, ge=2, le=40)
    max_per_session: int = Field(default=3, ge=1, le=8)
    target_duration_seconds: Optional[float] = Field(default=None, ge=30, le=7200)


class GenerateRecipeProjectRequest(BaseModel):
    recipe_id: str = Field(min_length=1, max_length=64)
    game: Optional[str] = Field(default=None, max_length=120)
    # Lets the creator rename before generating; the recipe supplies a default.
    title: Optional[str] = Field(default=None, max_length=160)


class GenerateThemeProjectRequest(BaseModel):
    """One montage from one theme, drawn across every scanned session."""

    title: str = Field(min_length=1, max_length=160)
    query: str = Field(default="", max_length=240)
    game: Optional[str] = Field(default=None, max_length=120)
    date_from: Optional[str] = Field(default=None, max_length=10)
    date_to: Optional[str] = Field(default=None, max_length=10)
    target_duration_seconds: float = Field(default=60.0, ge=30, le=7200)
    max_per_session: int = Field(default=3, ge=1, le=8)
    # Detected moments need their recording on disk to be cut. Off by default
    # so a montage is buildable now rather than after a queue of downloads.
    ready_only: bool = True


class CompilationAudioRequest(BaseModel):
    """Silence a montage in one action, optionally sparing its last moment."""

    muted: bool = True
    # The reaction at the end is usually the reason the montage exists.
    keep_finale: bool = True


class SaveCompilationReelRequest(BaseModel):
    dest_folder: str = Field(min_length=1, max_length=1000)


class GenerateMatchProjectRequest(BaseModel):
    """Build a compilation from one match inside one scanned session."""

    job_id: str = Field(min_length=1, max_length=100)
    match_index: int = Field(ge=1)
    title: Optional[str] = Field(default=None, max_length=160)
    max_clips: int = Field(default=12, ge=2, le=40)
    target_duration_seconds: Optional[float] = Field(default=None, ge=30, le=7200)


class UpdateCompilationProjectRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class CompilationProjectItemRequest(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    included: bool = True
    # A muted moment plays silent in the montage; its clip keeps its own audio.
    muted: bool = False


class UpdateCompilationProjectItemsRequest(BaseModel):
    items: List[CompilationProjectItemRequest] = Field(min_length=1, max_length=40)


class AddCompilationProjectClipRequest(BaseModel):
    clip_id: str = Field(min_length=1, max_length=100)
    memory_entry_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    memory_query: Optional[str] = Field(default=None, max_length=240)


class ExportCompilationProjectRequest(BaseModel):
    dest_folder: str = Field(min_length=1, max_length=4096)
    filename: Optional[str] = Field(default=None, max_length=240)


class StartCompilationPreparationRequest(BaseModel):
    source_keys: Optional[List[str]] = Field(default=None, max_length=40)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/memory/stats")
def stream_memory_stats():
    result = stream_memory_service.stats()
    indexer = globals().get("stream_memory_continuous_indexer")
    if indexer is not None:
        result["continuous_indexing"] = indexer.status()
    return result


@app.get("/memory/search")
def search_stream_memory(
    q: str = Query(min_length=1, max_length=240),
    kind: str = Query(default="all", pattern="^(all|transcript|clip|evidence)$"),
    decision: str = Query(
        default="all",
        pattern="^(all|kept|passed|maybe|unreviewed)$",
    ),
    exported: str = Query(
        default="all",
        pattern="^(all|exported|not_exported)$",
    ),
    origin: str = Query(default="all", pattern="^(all|creator_marker)$"),
    game: Optional[str] = Query(default=None, max_length=120),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(default=40, ge=1, le=100),
    mode: str = Query(default="hybrid", pattern="^(hybrid|keyword)$"),
):
    try:
        return stream_memory_service.search(
            q,
            kind=kind,
            decision=decision,
            exported=exported,
            origin=origin,
            game=game,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            mode=mode,
            track_interactions=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/memory/aliases")
def list_stream_memory_aliases():
    return stream_memory_service.list_aliases()


@app.post("/memory/aliases", status_code=201)
def add_stream_memory_alias(request: StreamMemoryAliasRequest):
    try:
        return stream_memory_service.add_alias(request.canonical_term, request.alias)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/memory/aliases/{alias_id}")
def delete_stream_memory_alias(alias_id: str):
    try:
        return stream_memory_service.delete_alias(alias_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/memory/feedback")
def set_stream_memory_feedback(request: StreamMemoryFeedbackRequest):
    try:
        return stream_memory_service.set_feedback(
            request.query,
            request.entry_id,
            request.verdict,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/memory/interactions")
def record_stream_memory_interactions(request: StreamMemoryInteractionRequest):
    try:
        return stream_memory_service.record_interactions(
            request.search_id,
            request.event_type,
            [entry.model_dump() for entry in request.entries],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/memory/learning")
def get_stream_memory_learning():
    return stream_memory_service.learning_summary()


@app.delete("/memory/learning")
def clear_stream_memory_learning():
    return stream_memory_service.clear_learning_history()


@app.post("/memory/learning/evaluate")
def run_stream_memory_learning_evaluation():
    if job_manager.has_active_jobs():
        raise HTTPException(
            status_code=409,
            detail="Wait for the active scan to finish before running the Memory learning comparison.",
        )
    try:
        return stream_memory_service.run_learning_shadow_evaluation()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/memory/evaluation")
def get_stream_memory_evaluation():
    return stream_memory_service.evaluation_summary()


@app.post("/memory/evaluation/misses", status_code=201)
def save_stream_memory_missed_query(request: StreamMemoryMissedQueryRequest):
    try:
        return stream_memory_service.save_missed_query(
            request.query,
            request.expected_description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/memory/evaluation/cases/{case_id}")
def delete_stream_memory_evaluation_case(case_id: str):
    try:
        return stream_memory_service.delete_evaluation_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/memory/evaluation/cases/{case_id}/target")
def resolve_stream_memory_evaluation_case(
    case_id: str,
    request: StreamMemoryEvaluationTargetRequest,
):
    try:
        return stream_memory_service.resolve_evaluation_case(case_id, request.entry_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/memory/evaluation/run")
def run_stream_memory_evaluation():
    if job_manager.has_active_jobs():
        raise HTTPException(
            status_code=409,
            detail="Wait for the active scan to finish before measuring Memory search.",
        )
    try:
        return stream_memory_service.run_evaluation()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/memory/reindex")
def reindex_stream_memory(request: StreamMemoryReindexRequest):
    # Database/cache-only maintenance: no models are loaded and no scan truth
    # is changed. Missing old transcripts are reported honestly as clip-only.
    return stream_memory_service.reindex_completed(force=request.force)


@app.post("/memory/semantic/reindex")
def reindex_stream_memory_semantics():
    # Keep CPU and disk maintenance out of the dependable scan lane. The old
    # concept index remains usable (and marked stale) until an idle rebuild.
    if job_manager.has_active_jobs():
        raise HTTPException(
            status_code=409,
            detail="Wait for the active scan to finish before updating the concept index.",
        )
    try:
        return stream_memory_service.build_semantic_index()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _compilation_project_call(operation):
    try:
        return operation()
    except CompilationProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CompilationProjectConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _compilation_preparation_call(operation):
    try:
        return operation()
    except CompilationProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CompilationPreparationStorageError as exc:
        raise HTTPException(
            status_code=507,
            detail={"message": str(exc), "plan": exc.plan},
        ) from exc
    except CompilationPreparationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/compilation-projects")
def list_compilation_projects():
    return {"schema_version": 1, "projects": compilation_project_service.list()}


@app.get("/compilation-projects/recipes")
def list_compilation_recipes():
    """Named projects the creator can ask for, with the games they can target."""
    from datetime import date as _date
    return {
        "schema_version": 1,
        "recipes": compilation_recipes.available(db, today=_date.today()),
    }


@app.post("/compilation-projects/generate", status_code=201)
def generate_compilation_project(request: GenerateCompilationProjectRequest):
    return _compilation_project_call(
        lambda: compilation_project_service.generate(**request.model_dump())
    )


@app.post("/compilation-projects/generate-recipe", status_code=201)
def generate_compilation_project_from_recipe(request: GenerateRecipeProjectRequest):
    """Resolve a named recipe, then run the ordinary generator with its arguments."""
    from datetime import date as _date

    def run():
        arguments = compilation_recipes.resolve(
            request.recipe_id, today=_date.today(), game=request.game,
        )
        if request.title:
            arguments["title"] = request.title
        # A recipe is a named theme, so it builds the same montage the manual
        # theme form does. max_clips is retired here: length is the control.
        return compilation_project_service.generate_from_theme(
            title=arguments["title"],
            query=arguments.get("query") or "",
            game=arguments.get("game"),
            date_from=arguments.get("date_from"),
            date_to=arguments.get("date_to"),
            target_duration_seconds=arguments.get("target_duration_seconds") or 60.0,
            max_per_session=arguments.get("max_per_session") or 3,
        )

    return _compilation_project_call(run)


@app.get("/jobs/{job_id}/matches")
def list_session_matches(job_id: str):
    """The individual games Recall can see inside one scanned session.

    Read-only over detected events the scan already wrote: no source video is
    loaded, no model runs, and the scan pipeline is untouched.
    """
    return _compilation_project_call(
        lambda: compilation_project_service.match_options(job_id)
    )


@app.post("/compilation-projects/generate-theme", status_code=201)
def generate_compilation_project_from_theme(request: GenerateThemeProjectRequest):
    """Assemble one theme across sessions, ending on its strongest moment."""
    return _compilation_project_call(
        lambda: compilation_project_service.generate_from_theme(**request.model_dump())
    )


@app.post("/compilation-projects/generate-match", status_code=201)
def generate_compilation_project_from_match(request: GenerateMatchProjectRequest):
    """Assemble one match's strongest moments, with its win placed last."""
    return _compilation_project_call(
        lambda: compilation_project_service.generate_from_match(**request.model_dump())
    )


@app.get("/compilation-projects/{project_id}")
def get_compilation_project(project_id: str):
    return _compilation_project_call(lambda: compilation_project_service.get(project_id))


@app.patch("/compilation-projects/{project_id}")
def update_compilation_project(project_id: str, request: UpdateCompilationProjectRequest):
    return _compilation_project_call(
        lambda: compilation_project_service.update(project_id, title=request.title)
    )


@app.put("/compilation-projects/{project_id}/items")
def update_compilation_project_items(
    project_id: str,
    request: UpdateCompilationProjectItemsRequest,
):
    return _compilation_project_call(
        lambda: compilation_project_service.replace_items(
            project_id,
            [item.model_dump() for item in request.items],
        )
    )


@app.post("/compilation-projects/{project_id}/clips")
def add_compilation_project_clip(
    project_id: str,
    request: AddCompilationProjectClipRequest,
):
    def add_clip():
        reason = "Creator added this kept clip."
        if request.memory_entry_id:
            try:
                context = stream_memory_service.action_context(request.memory_entry_id)
            except KeyError as exc:
                raise CompilationProjectConflict(
                    "That Memory result is no longer in the local index. Search again and retry."
                ) from exc
            if context.get("clip_id") != request.clip_id:
                raise ValueError("That Memory result no longer points to this clip.")
            query = " ".join((request.memory_query or "").split())
            reason = (
                f"Added from Stream Memory search “{query}”."
                if query else "Added from Stream Memory."
            )
        elif request.memory_query:
            raise ValueError("memory_query requires memory_entry_id.")
        result = compilation_project_service.add_clip(
            project_id,
            request.clip_id,
            selection_reason=reason,
        )
        if request.memory_entry_id:
            try:
                stream_memory_service.record_action(
                    request.memory_query or "",
                    request.memory_entry_id,
                    "compilation_add",
                    dedupe_key=f"compilation_add:{project_id}:{request.clip_id}",
                    context={"project_id": project_id, "clip_id": request.clip_id},
                )
            except Exception as exc:
                print(f"Stream Memory compilation learning skipped: {exc}")
        return result

    return _compilation_project_call(add_clip)


@app.post("/compilation-projects/{project_id}/approve")
def approve_compilation_project(project_id: str):
    return _compilation_project_call(
        lambda: compilation_project_service.approve(project_id)
    )


@app.get("/compilation-projects/{project_id}/preparation")
def get_compilation_preparation(project_id: str):
    return _compilation_preparation_call(
        lambda: compilation_preparation_service.plan(project_id)
    )


@app.post("/compilation-projects/{project_id}/preparation", status_code=202)
def start_compilation_preparation(
    project_id: str,
    request: StartCompilationPreparationRequest,
):
    return _compilation_preparation_call(
        lambda: compilation_preparation_service.start(project_id, request.source_keys)
    )


@app.post("/compilation-projects/{project_id}/preparation/cancel")
def cancel_compilation_preparation(project_id: str):
    return _compilation_preparation_call(
        lambda: compilation_preparation_service.cancel(project_id)
    )


@app.post("/compilation-projects/{project_id}/cut-next")
def cut_next_compilation_moment(project_id: str):
    """Render one detected moment of this project into a real clip.

    One moment per call, because each cut is a genuine render: the creator
    keeps control of how far the batch goes and can stop after any of them.
    """
    def run():
        item = compilation_project_service.next_uncut_item(project_id)
        if item is None:
            return {
                "done": True,
                "cut_clip_id": None,
                "project": compilation_project_service.get(project_id),
            }
        if compilation_preparation_service._has_active_scan():  # noqa: SLF001
            # Same mutual exclusion the source-restore path enforces: a render
            # must not compete with a scan for the machine.
            raise HTTPException(
                status_code=409,
                detail="A scan is running. Cut these moments once it finishes.",
            )
        if not item.get("job_id"):
            raise HTTPException(
                status_code=409, detail="That moment has lost its source session."
            )
        with db.get_connection() as conn:
            source = conn.execute(
                "SELECT source_path, asset_path FROM jobs WHERE id = ?",
                (item["job_id"],),
            ).fetchone()
        paths = [source["source_path"], source["asset_path"]] if source else []
        if not any(path and os.path.isfile(str(path)) for path in paths):
            # Cutting renders from the recording, so say what is missing rather
            # than failing inside the renderer.
            raise HTTPException(
                status_code=409,
                detail=(
                    "The recording for this session is not on disk. "
                    "Restore its source above, then cut these moments."
                ),
            )
        cut = clip_service.create_manual(
            item["job_id"],
            ManualClipRequest(
                start=item["start_time"],
                end=item["end_time"],
                title=item["title"],
                tags=["compilation", item.get("moment_kind") or "moment"],
                # Kills render clean; the closing win keeps its captions. The
                # layout stays "auto" so every moment keeps its facecam.
                captions_enabled=item.get("captions_enabled"),
                origin="compilation",
            ),
        )
        project = compilation_project_service.attach_clip(
            project_id, item["id"], str(cut["id"])
        )
        return {"done": False, "cut_clip_id": cut["id"], "project": project}

    return _compilation_project_call(run)


@app.post("/compilation-projects/{project_id}/audio")
def set_compilation_audio(project_id: str, request: CompilationAudioRequest):
    """Mute or unmute the montage, keeping the closing moment audible."""
    return _compilation_project_call(
        lambda: compilation_project_service.set_audio(
            project_id, muted=request.muted, keep_finale=request.keep_finale,
        )
    )


@app.post("/compilation-projects/{project_id}/build-reel")
def build_compilation_reel(project_id: str):
    """Stitch this project's cut moments into one montage and keep it.

    The montage lands in Recall's own export directory rather than a folder the
    creator picks, because this is the thing they are about to watch and prune,
    not yet the file they are posting. Saving it out stays a separate,
    deliberate action.

    Order is the project's order, not the VOD's: the reel closes on the win,
    which is the whole point of the sequence.
    """
    def run():
        clip_ids, muted = compilation_project_service.reel_plan(project_id)
        result = clip_service.compile_reel(
            clip_ids, exports_dir, filename=f"recall_montage_{project_id[-12:]}",
            mute_clip_ids=muted,
        )
        if result.get("error") or not result.get("path"):
            raise HTTPException(
                status_code=409,
                detail=result.get("error") or "Recall could not stitch that montage.",
            )

        duration = 0.0
        with db.get_connection() as conn:
            for clip_id in result.get("succeeded_clip_ids") or clip_ids:
                row = conn.execute(
                    "SELECT start_time, end_time FROM clips WHERE id = ?", (clip_id,)
                ).fetchone()
                if row:
                    duration += max(
                        0.0, float(row["end_time"] or 0) - float(row["start_time"] or 0)
                    )
        return {
            "project": compilation_project_service.record_reel(
                project_id, path=result["path"], duration_seconds=round(duration, 3),
            ),
            "clips": int(result.get("clips") or len(clip_ids)),
            "errors": int(result.get("errors") or 0),
        }

    return _compilation_project_call(run)


@app.post("/compilation-projects/{project_id}/save-reel")
def save_compilation_reel(project_id: str, request: SaveCompilationReelRequest):
    """Copy the built montage into a folder the creator picked.

    The montage already exists, so saving it is a copy rather than a second
    render of the same sequence.
    """
    def run():
        project = compilation_project_service.get(project_id)
        source = project.get("reel_path")
        if not source or not os.path.isfile(str(source)):
            raise HTTPException(
                status_code=409,
                detail="Build the montage before saving it.",
            )
        if project.get("reel_stale"):
            raise HTTPException(
                status_code=409,
                detail="Rebuild the montage first — this file is the previous version.",
            )

        from core.export_presets import sanitize_filename

        folder = os.path.abspath(str(request.dest_folder))
        os.makedirs(folder, exist_ok=True)
        stem = sanitize_filename(project["title"], "recall_montage")
        destination = os.path.join(folder, f"{stem}.mp4")
        index = 2
        while os.path.exists(destination):
            destination = os.path.join(folder, f"{stem}_{index}.mp4")
            index += 1
        shutil.copy2(str(source), destination)
        return {"path": destination, "project_id": project_id}

    return _compilation_project_call(run)


@app.post("/compilation-projects/{project_id}/export", status_code=202)
def export_compilation_project(
    project_id: str,
    request: ExportCompilationProjectRequest,
):
    def start_export():
        project = compilation_project_service.get(project_id)
        clip_ids = compilation_project_service.approved_clip_ids(project_id)
        operation = export_operation_service.start_reel(
            clip_ids,
            request.dest_folder,
            filename=request.filename or project["title"],
        )
        return {"project_id": project_id, "project_status": project["status"], **operation}

    return _compilation_project_call(start_export)


@app.delete("/compilation-projects/{project_id}")
def delete_compilation_project(project_id: str):
    return _compilation_project_call(
        lambda: compilation_project_service.delete(project_id)
    )


def _recall_session_call(operation):
    try:
        return operation()
    except RecallSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RecallSessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RecallSessionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/recall-sessions")
def start_recall_session(request: RecallSessionStartRequest):
    return _recall_session_call(lambda: recall_session_service.start(
        source_platform=request.source_platform,
        source_ref=request.source_ref,
        title=request.title,
        started_at_utc=request.started_at_utc,
        initial_stream_time_seconds=request.initial_stream_time_seconds,
        metadata=request.metadata,
    ))


@app.get("/recall-sessions")
def list_recall_sessions(limit: int = 50):
    return recall_session_service.list_sessions(limit=limit)


@app.get("/recall-sessions/active")
def active_recall_session():
    return {"session": recall_session_service.get_active()}


@app.post("/recall-sessions/active/remember")
def remember_active_recall_session():
    return _recall_session_call(recall_session_service.record_active_remember)


@app.get("/recall-sessions/{session_id}")
def get_recall_session(session_id: str):
    return _recall_session_call(lambda: recall_session_service.get(session_id))


@app.post("/recall-sessions/{session_id}/stop")
def stop_recall_session(session_id: str, request: RecallSessionStopRequest):
    return _recall_session_call(lambda: recall_session_service.stop(
        session_id, ended_at_utc=request.ended_at_utc,
    ))


@app.post("/recall-sessions/{session_id}/events")
def add_recall_session_event(session_id: str, request: RecallSessionEventRequest):
    return _recall_session_call(lambda: recall_session_service.record_event(
        session_id,
        kind=request.kind,
        source=request.source,
        occurred_at_utc=request.occurred_at_utc,
        stream_offset_seconds=request.stream_offset_seconds,
        confidence=request.confidence,
        payload=request.payload,
    ))


@app.delete("/recall-sessions/{session_id}/events/{event_id}")
def delete_recall_session_event(session_id: str, event_id: str):
    return _recall_session_call(
        lambda: recall_session_service.delete_event(session_id, event_id),
    )


@app.post("/recall-sessions/{session_id}/clock-anchors")
def add_recall_session_clock_anchor(
    session_id: str, request: RecallSessionClockAnchorRequest,
):
    return _recall_session_call(lambda: recall_session_service.add_clock_anchor(
        session_id,
        wall_time_utc=request.wall_time_utc,
        stream_time_seconds=request.stream_time_seconds,
        anchor_type=request.anchor_type,
        uncertainty_seconds=request.uncertainty_seconds,
        source=request.source,
    ))


@app.post("/recall-sessions/{session_id}/region-plan")
def plan_recall_session_regions(
    session_id: str, request: RecallSessionRegionPlanRequest,
):
    return _recall_session_call(lambda: recall_session_service.build_region_plan(
        session_id,
        vod_started_at_utc=request.vod_started_at_utc,
        vod_duration_seconds=request.vod_duration_seconds,
        pre_roll_seconds=request.pre_roll_seconds,
        post_roll_seconds=request.post_roll_seconds,
    ))


def _maybe_train_ranker_background():
    learning_service.maybe_train_background()


def _configure_frozen_stdio() -> None:
    """Give the windowed PyInstaller engine a durable stdout/stderr sink.

    With ``console=False`` PyInstaller sets both streams to ``None``. Any print
    from Recall or a model library would otherwise terminate startup before
    Uvicorn can bind its port.
    """
    if not is_frozen() or (sys.stdout is not None and sys.stderr is not None):
        return
    log_dir = get_data_dir()
    os.makedirs(log_dir, exist_ok=True)
    stream = open(os.path.join(log_dir, "engine.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream

@app.post("/jobs/run")
def run_pipeline_job(request: RunJobRequest):
    try:
        return job_service.run(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/download")
def download_vod_job(request: DownloadVodRequest):
    return job_service.download(request)

@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return job_service.cancel(job_id)

@app.get("/settings")
def get_settings():
    """Persisted creator settings (scan prefs, theme, display name)."""
    return settings_service.get_all()


@app.put("/settings")
def update_settings(patch: Dict[str, Any]):
    """Upsert a partial settings patch; returns the full persisted set."""
    updated = settings_service.update(patch)
    if {"recallStorageCapGb", "sourceRetentionDays"}.intersection(patch):
        system_service.storage()
    return updated


@app.post("/system/wipe")
def wipe_system_data():
    return system_service.wipe()

@app.post("/system/cache/clear")
def clear_system_cache():
    return system_service.clear_cache()

@app.get("/system/storage")
def system_storage():
    """Recall storage target plus separately tracked downloaded sources."""
    return system_service.storage()


@app.get("/jobs/{job_id}/source-status")
def job_source_status(job_id: str):
    return system_service.source_status_for_job(job_id)


@app.post("/jobs/{job_id}/source/restore")
def restore_job_source(job_id: str):
    return system_service.restore_source_for_job(job_id)


@app.post("/source-assets/{source_key}/remove")
def remove_source_asset(source_key: str):
    return system_service.remove_source(source_key)


@app.post("/source-assets/{source_key}/pin")
def pin_source_asset(source_key: str, request: PinSourceRequest):
    return system_service.pin_source(source_key, request.pinned)


@app.post("/source-assets/{source_key}/restore")
def restore_source_asset(source_key: str):
    return system_service.restore_source(source_key)


@app.post("/source-assets/{source_key}/restore/cancel")
def cancel_source_restore(source_key: str):
    return system_service.cancel_restore(source_key)


@app.get("/system/capabilities")
def system_capabilities(refresh: bool = False):
    """Safe first-run hardware route; an explicit refresh requires an idle engine."""
    if refresh and job_manager.has_active_jobs():
        raise HTTPException(
            status_code=409,
            detail="Wait for the current scan to finish before checking hardware again.",
        )
    from core.hardware_capabilities import get_hardware_capabilities

    return get_hardware_capabilities(refresh=refresh)


@app.get("/system/model-setup")
def model_setup_status():
    return model_setup_controller.status()


@app.get("/system/model-setup/plan")
def model_setup_plan():
    return model_setup_controller.plan()


class StartModelSetupRequest(BaseModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")


@app.post("/system/model-setup/start")
def start_model_setup(request: StartModelSetupRequest):
    """Explicit preparation of staged speech assets, not full app readiness."""
    try:
        return model_setup_controller.start(request.plan_id)
    except ModelSetupConflict as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception:
        raise HTTPException(status_code=500, detail="Could not start model setup")


@app.post("/system/model-setup/{operation_id}/cancel")
def cancel_model_setup(operation_id: str):
    try:
        return model_setup_controller.cancel(operation_id)
    except ModelSetupConflict as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/system/model-setup/{operation_id}/activate")
def activate_model_setup(operation_id: str):
    try:
        return model_setup_controller.activate(operation_id)
    except ModelSetupConflict as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception:
        raise HTTPException(status_code=500, detail="Could not activate speech models")


@app.post("/system/assets/clear")
def clear_system_assets():
    """Delete all downloaded VODs; sessions and rendered clips stay."""
    return system_service.clear_assets()

@app.post("/probe")
def probe_source(request: ProbeRequest):
    """Pre-scan probe for the import preview: real duration + poster frame for a
    local file (URL sources degrade to nulls)."""
    return system_service.probe(request.source_path, request.source_type)

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    return job_service.delete(job_id)

@app.get("/jobs")
def list_jobs():
    return job_service.list()

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return job_service.get(job_id)

@app.get("/jobs/{job_id}/diagnostics")
def get_job_diagnostics(job_id: str):
    """Copyable plain-text diagnostics bundle (machine + job + events + log tail)."""
    return job_service.diagnostics(job_id)

@app.get("/jobs/{job_id}/source")
def get_job_source(job_id: str):
    """Full source VOD for the Cutting Room, served seekably.

    Starlette's FileResponse honors HTTP Range headers (206 + Content-Range),
    which is what lets a <video> element scrub through a multi-GB VOD without
    downloading it front to back.
    """
    return job_service.source_media(job_id)


@app.get("/jobs/{job_id}/media-status")
def job_media_status(job_id: str):
    """Which of this session's clip proxies survived on disk, and whether the
    missing ones can be rebuilt. Session records are never lost with the files,
    so this reports recoverability, not damage."""
    status = job_service.media_status(job_id)
    status["rebuild"] = clip_service.rebuild_status(job_id)
    return status


@app.post("/jobs/{job_id}/media/rebuild")
def rebuild_job_media(job_id: str):
    """Re-render this session's missing clip previews from its source VOD.

    Always creator-initiated. Returns ``source_required`` rather than starting a
    download on its own -- restoring a Twitch source is its own explicit step.
    """
    return clip_service.rebuild_media(job_id)


@app.get("/jobs/{job_id}/thumb")
def job_thumb(job_id: str):
    """Landscape still from the source VOD for session gallery cards.

    Grabs a real VOD frame (preferring the strongest clip's peak) so cards
    aren't forced to crop a 9:16 clip export into a 16:9 well.
    """
    path = job_service.poster_path(job_id)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/filmstrip")
def job_filmstrip(
    job_id: str,
    start: float = 0.0,
    end: Optional[float] = None,
    frames: int = 24,
    width: int = 1600,
    height: int = 64,
):
    """Cached source-frame contact sheet for both VOD editor timelines."""
    path = job_service.filmstrip_path(
        job_id,
        start=start,
        end=end,
        frames=frames,
        width=width,
        height=height,
    )
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/jobs/{job_id}/timeline")
def get_job_timeline(job_id: str):
    """Compact VOD-level reaction curve R(t) for the review timeline.

    Returns {"timeline": null} for legacy-engine jobs or jobs processed
    before timelines were persisted.
    """
    return job_service.timeline(job_id)


@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request):
    """Server-Sent Events stream of live job telemetry. Replays the last known
    snapshot immediately, then pushes structured events until the job ends."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    subscriber_id, q = job_manager.subscribe_async(job_id)

    async def event_gen():
        try:
            # Prime the stream with the current job snapshot so a late subscriber
            # (e.g. after a page reload) is immediately consistent.
            snapshot = job_manager.get_job(job_id)
            yield f"event: snapshot\ndata: {_json.dumps(snapshot)}\n\n"

            terminal = {"completed", "failed", "cancelled"}
            if snapshot and snapshot.get("status") in terminal:
                yield "event: end\ndata: {}\n\n"
                return

            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Nothing progressed for 15s (e.g. a long silent sub-stage
                    # like facecam refinement). Re-send a fresh snapshot so the
                    # clock/ETA on the theater screen keep counting down
                    # instead of freezing at whatever the last real event said.
                    # Skip the event-log query here — events arrive on the
                    # message channel; only the priming snapshot needs them.
                    snapshot = job_manager.get_job(job_id, include_events=False)
                    if snapshot and snapshot.get("status") in terminal:
                        yield "event: end\ndata: {}\n\n"
                        return
                    yield f"event: snapshot\ndata: {_json.dumps(snapshot)}\n\n"
                    continue
                yield f"event: message\ndata: {_json.dumps(ev)}\n\n"
                if ev.get("event_type") in ("job_completed", "job_failed", "job_cancelled"):
                    yield "event: end\ndata: {}\n\n"
                    return
        finally:
            job_manager.unsubscribe_async(job_id, subscriber_id)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/exports")
def list_exports():
    return clip_service.list_exports()

@app.post("/clips/delete-batch")
def delete_clips_batch(request: DeleteClipsRequest, background_tasks: BackgroundTasks):
    result = clip_service.delete_batch(request.clip_ids)
    background_tasks.add_task(_maybe_train_ranker_background)
    return result

@app.post("/clips/copy-to-folder")
def copy_clips_to_folder(request: CopyClipsRequest):
    result = clip_service.copy_to_folder(
        request.clip_ids, request.dest_folder,
        preset=request.preset, filename_template=request.filename_template,
    )
    return result


@app.post("/clips/export-operations", status_code=202)
def start_clip_export(request: CopyClipsRequest):
    """Start a serialized final-file export and return its live snapshot."""
    return export_operation_service.start(
        request.clip_ids,
        request.dest_folder,
        preset=request.preset,
        filename_template=request.filename_template,
    )


@app.post("/clips/reel-operations", status_code=202)
def start_reel_export(request: CompileReelRequest):
    """Start serialized reel preparation and return its live snapshot."""
    return export_operation_service.start_reel(
        request.clip_ids,
        request.dest_folder,
        filename=request.filename,
    )


@app.get("/clips/export-operations/{operation_id}")
def get_clip_export(operation_id: str):
    return export_operation_service.get(operation_id)


@app.post("/clips/export-operations/{operation_id}/cancel")
def cancel_clip_export(operation_id: str):
    return export_operation_service.cancel(operation_id)

@app.post("/clips/compile-reel")
def compile_reel(request: CompileReelRequest):
    """Concatenate kept clips into one highlight reel (plan 22 §3.2)."""
    result = clip_service.compile_reel(
        request.clip_ids, request.dest_folder, filename=request.filename,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/clips/{clip_id}/label")
def label_clip(clip_id: str, request: LabelClipRequest, background_tasks: BackgroundTasks):
    result = clip_service.label(clip_id, request)
    # Only explicit Keep/Pass verdicts supervise the personal challenger.
    # Export and Maybe remain useful product state, but rebuilding an identical
    # model for either event wastes work and used to let those later actions
    # mask the actual verdict in the training view.
    if request.event in ("saved", "passed"):
        background_tasks.add_task(_maybe_train_ranker_background)
    return result

@app.get("/clips")
def list_clips(job_id: Optional[str] = None):
    return clip_service.list_clips(job_id=job_id)

@app.get("/jobs/{job_id}/clips/more")
def more_clip_candidate_status(job_id: str, limit: int = 15):
    """Describe the next ceiling-cut review batch without changing the deck."""
    return clip_service.more_candidate_status(job_id, limit)

@app.post("/jobs/{job_id}/clips/more")
def materialize_more_clip_candidates(job_id: str, limit: int = 15):
    """Reveal a bounded second-look tier already preview-rendered at scan time."""
    return clip_service.materialize_more_candidates(job_id, limit)

@app.get("/clips/{clip_id}/preview")
def preview_clip(clip_id: str):
    return clip_service.preview(clip_id)

@app.post("/clips/{clip_id}/preview")
def prepare_clip_preview(clip_id: str):
    """Render an explicitly requested vertical proof for a Second-look clip."""
    return clip_service.prepare_preview(clip_id)

@app.get("/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str):
    """Poster JPEG for grid cards / filmstrips. Serves the pre-rendered file,
    or generates it on demand for clips made before thumbnails existed."""
    path = clip_service.thumbnail_path(clip_id)
    # no-cache: the poster is regenerated in place on re-render, so revalidate
    # rather than let a stale (or briefly-missing) thumbnail stick in the cache.
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

@app.patch("/clips/{clip_id}")
def update_clip(clip_id: str, request: UpdateClipRequest):
    result = clip_service.update(clip_id, request)
    if request.title is not None:
        try:
            stream_memory_service.refresh_clip(clip_id)
        except Exception as exc:
            print(f"Stream Memory clip refresh skipped: {exc}")
    return result

@app.post("/clips/state-batch")
def update_clip_state_batch(request: ClipStateBatchRequest):
    """Bulk kept/passed/exported update (localStorage → DB migration)."""
    return clip_service.update_state_batch(
        request.clip_ids, kept=request.kept, passed=request.passed,
        maybe=request.maybe, exported=request.exported
    )

@app.post("/clips/{clip_id}/edit")
def edit_clip(clip_id: str, request: EditClipRequest):
    result = clip_service.edit(clip_id, request)
    try:
        stream_memory_service.refresh_clip(clip_id)
    except Exception as exc:
        print(f"Stream Memory clip refresh skipped: {exc}")
    return result

@app.get("/clips/{clip_id}/caption")
def get_clip_caption(clip_id: str, transcribe: bool = False):
    """The words burned into this clip.

    A stored correction always comes back. Reading the machine's own words
    costs a per-clip ASR pass, so it needs ``transcribe=true`` -- otherwise the
    response is ``source: "none"`` and the caller decides whether to spend it.
    """
    return clip_service.get_caption(clip_id, transcribe=transcribe)

@app.put("/clips/{clip_id}/caption")
def save_clip_caption(clip_id: str, request: CaptionEditRequest):
    """Correct this clip's caption. Takes effect on the clip's next render."""
    return clip_service.save_caption(clip_id, request.text)

@app.post("/jobs/{job_id}/clips/manual")
def create_manual_clip(job_id: str, request: ManualClipRequest):
    result = clip_service.create_manual(job_id, request)
    try:
        stream_memory_service.refresh_job_clips(job_id)
    except Exception as exc:
        print(f"Stream Memory clip refresh skipped: {exc}")
    return result

@app.post("/jobs/{job_id}/clips/manual/framing")
def resolve_manual_clip_framing(job_id: str, request: ManualClipRequest):
    start, end = clip_service.manual_bounds(job_id, request)
    return clip_service.resolve_manual_framing(
        job_id, start, end,
        requested_layout=request.layout,
        focus_x=request.focus_x,
    )

@app.post("/jobs/{job_id}/clips/manual/preview")
def preview_manual_clip(job_id: str, request: ManualClipRequest):
    path = clip_service.preview_manual(job_id, request)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="recall-vod-editor-preview.mp4",
        background=BackgroundTask(clip_service.cleanup_manual_preview, path),
    )

@app.get("/ranker/status")
def ranker_status():
    """Learned-ranker state: label count, threshold, which model is active."""
    return learning_service.status()


@app.post("/ranker/train")
def ranker_train():
    """Fine-tune the per-install ranker on captured keep/reject labels."""
    return learning_service.train()


@app.post("/ranker/reset")
def ranker_reset():
    """Reset local personalization without deleting clips or sessions."""
    return learning_service.reset()


if __name__ == "__main__":
    _configure_frozen_stdio()

    import uvicorn
    if is_frozen():
        # Pass the app object directly: uvicorn's "module:attr" string form
        # re-imports "main" by name, which isn't reliable for a frozen entry
        # script. Reload is meaningless here too (no source file to watch).
        print("Recall API: production mode (reload disabled)")
        port = int(os.environ.get("RECALL_API_PORT", "8000"))
        # Media/EventSource URLs carry the launch token; never persist requests.
        uvicorn.run(app, host="127.0.0.1", port=port, reload=False, access_log=False)
    else:
        reload_enabled = os.environ.get("RECALL_API_RELOAD", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        port = int(os.environ.get("RECALL_API_PORT", "8000"))
        if not reload_enabled:
            # Electron owns source-backend startup and shutdown. Keep that path
            # to one directly observable process; developers launching main.py
            # themselves retain Uvicorn hot reload below.
            print("Recall API: source mode (reload disabled by launcher)")
            # Electron authenticates media/EventSource requests in the query
            # string. Suppress access logs so the per-launch token never lands
            # in the launcher terminal; startup failures still reach stderr.
            uvicorn.run(
                app, host="127.0.0.1", port=port, reload=False, access_log=False,
            )
            raise SystemExit(0)

        # Dev reload watches ONLY real source dirs. The default (recursive CWD)
        # would watch apps/api/build + dist (PyInstaller output) and restart the
        # server mid-scan whenever a packaging run or stray artifact touches a
        # .py file there — and would miss engine/pipeline edits entirely.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        source_dirs = [
            os.path.dirname(os.path.abspath(__file__)),  # apps/api itself
            os.path.join(repo_root, "engines"),
            os.path.join(repo_root, "core"),
            os.path.join(repo_root, "pipeline"),
        ]
        print(f"Recall API: dev mode (reload watching {len(source_dirs)} source dirs only)")
        uvicorn.run(
            "main:app", host="127.0.0.1", port=port, reload=True,
            access_log=False,
            reload_dirs=source_dirs,
            reload_excludes=["build/*", "dist/*", "__pycache__/*"],
        )
