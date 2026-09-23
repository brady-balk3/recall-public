# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Named compilation recipes.

Roadmap phase 7 asks for projects the creator can ask for by name -- Funniest
Moments, Best Plays, This Week on Stream -- rather than only hand-built date
filters. The generator already accepts a query, a game, a date window, and a
size, so a recipe is a resolved set of those arguments and nothing more. No new
selection machinery, no new storage, and every recipe still runs through the
same kept-clips-only path.

Recipes resolve against a supplied date so the caller can be deterministic in
tests; nothing here reads the clock on its own except through ``today``.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass(frozen=True)
class Recipe:
    id: str
    label: str
    description: str
    # Empty query means "every kept clip in the window", which is what a
    # date-scoped highlight reel wants. A query routes through Stream Memory.
    query: str = ""
    # ("days", n) is a rolling window ending today; ("month", offset) is a whole
    # calendar month, offset 0 being the current one.
    window: Optional[tuple[str, int]] = None
    max_clips: int = 12
    max_per_session: int = 3
    target_duration_seconds: Optional[float] = None
    # Recipes that only make sense once a game is chosen.
    needs_game: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        id="this_week",
        label="This Week on Stream",
        description="Everything you kept over the last seven days.",
        window=("days", 7),
        max_clips=10,
        tags=("recent",),
    ),
    Recipe(
        id="best_of_month",
        label="Best of {month}",
        description="This month's kept moments, newest streams included.",
        window=("month", 0),
        tags=("recent",),
    ),
    Recipe(
        id="best_of_last_month",
        label="Best of {month}",
        description="Last month, once the month is complete.",
        window=("month", -1),
        tags=("archive",),
    ),
    Recipe(
        id="funniest",
        label="Funniest Moments",
        description="Laughter, jokes, and the moments that broke you.",
        query="funny laughing hilarious joke",
        tags=("theme",),
    ),
    Recipe(
        id="best_plays",
        label="Best Plays",
        description="Clutches, wins, and eliminations.",
        query="clutch win elimination play",
        tags=("theme", "gameplay"),
    ),
    Recipe(
        id="rage",
        label="Rage Compilation",
        description="Losing it, repeatedly.",
        query="rage angry mad screaming",
        tags=("theme",),
    ),
    Recipe(
        id="scared",
        label="Scared Moments",
        description="Jumpscares and the reactions to them.",
        query="scared scream jumpscare startled",
        tags=("theme",),
    ),
    Recipe(
        id="chat_moments",
        label="Best Chat Moments",
        description="Moments your chat drove.",
        query="chat viewers stream chat reaction",
        tags=("theme", "chat"),
    ),
    Recipe(
        id="game_highlights",
        label="{game} Highlights",
        description="Kept moments from one game across every stream.",
        needs_game=True,
        max_clips=15,
        tags=("game",),
    ),
    Recipe(
        id="playthrough",
        label="{game} in 20 Minutes",
        description="One game start to finish, in stream order.",
        needs_game=True,
        max_clips=40,
        max_per_session=6,
        target_duration_seconds=1200.0,
        tags=("game", "story"),
    ),
)

BY_ID = {recipe.id: recipe for recipe in RECIPES}


def _month_window(today: date, offset: int) -> tuple[str, str, str]:
    """Whole calendar month at ``offset`` from today's month."""
    year, month = today.year, today.month + offset
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    last = calendar.monthrange(year, month)[1]
    return (
        date(year, month, 1).isoformat(),
        date(year, month, last).isoformat(),
        MONTHS[month - 1],
    )


def resolve(
    recipe_id: str,
    *,
    today: date,
    game: Optional[str] = None,
) -> dict[str, Any]:
    """Turn a recipe into keyword arguments for ``CompilationProjectService.generate``."""
    recipe = BY_ID.get(str(recipe_id or "").strip())
    if recipe is None:
        raise ValueError(f"Unknown compilation recipe: {recipe_id}")
    game = (game or "").strip() or None
    if recipe.needs_game and not game:
        raise ValueError(f"{recipe.label.replace('{game}', 'This recipe')} needs a game.")

    date_from = date_to = None
    month_name = ""
    if recipe.window:
        kind, amount = recipe.window
        if kind == "days":
            # Inclusive of today, so seven days means today and the six before.
            date_from = (today - timedelta(days=amount - 1)).isoformat()
            date_to = today.isoformat()
        elif kind == "month":
            date_from, date_to, month_name = _month_window(today, amount)
        else:  # pragma: no cover - guarded by the recipe table above
            raise ValueError(f"Unknown recipe window: {kind}")

    title = recipe.label.format(
        month=month_name or MONTHS[today.month - 1],
        game=(game or "").title(),
    )
    return {
        "title": title,
        "query": recipe.query,
        "game": game,
        "date_from": date_from,
        "date_to": date_to,
        "max_clips": recipe.max_clips,
        "max_per_session": recipe.max_per_session,
        "target_duration_seconds": recipe.target_duration_seconds,
    }


def available(db, *, today: date) -> list[dict[str, Any]]:
    """Recipes plus the games they can be pointed at.

    Deliberately does not run the searches to count candidates: eight Stream
    Memory queries per page load is real latency for a number the creator gets
    the moment they generate anyway.
    """
    with db.get_connection() as conn:
        games = [
            str(row[0])
            for row in conn.execute(
                """SELECT DISTINCT LOWER(m.game) FROM stream_memory_entries AS m
                   JOIN clips AS c ON c.id = m.clip_id
                   WHERE m.kind = 'clip' AND COALESCE(m.game, '') <> ''
                     AND COALESCE(c.kept, 0) = 1
                   ORDER BY 1"""
            ).fetchall()
            if row[0]
        ]
    out: list[dict[str, Any]] = []
    for recipe in RECIPES:
        if recipe.needs_game and not games:
            continue
        entry: dict[str, Any] = {
            "id": recipe.id,
            "description": recipe.description,
            "needs_game": recipe.needs_game,
            "tags": list(recipe.tags),
            "games": games if recipe.needs_game else [],
        }
        if recipe.needs_game:
            entry["label"] = recipe.label
        else:
            entry["label"] = resolve(recipe.id, today=today)["title"]
        out.append(entry)
    return out
