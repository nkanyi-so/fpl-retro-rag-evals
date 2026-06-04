"""Read-only access to the snapshotted FPL season data (the outcome oracle).

The decision-quality eval needs two things the corpus cannot give: real
per-gameweek points (the answer key) and a name -> player-id resolver (to turn a
free-text captain pick into something scoreable). Both come from a frozen snapshot
of the FPL API under `data/fpl/`, taken once and read from disk — the live API
wipes on season rollover, so the snapshot is the only reproducible source.

IMPORTANT — this is the GRADER, not an input to a decision. Points for GW N are an
*outcome*, known only after the deadline. Using them to score a pick is the answer
key, not temporal leakage; the temporal invariant constrains the corpus (inputs),
not this module. Anything here that reads GW N's result (`points`, `top_scorer`,
`pool_mean_points`) must never feed a decision for GW N.
"""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPL_DIR = ROOT / "data" / "fpl"

# Positions that are realistic captain options (captains are ~always outfield
# attackers; used for the template leader and the random-captain floor pool).
ATTACKING_TYPES = {3, 4}  # MID, FWD
OUTFIELD_TYPES = {2, 3, 4}  # DEF, MID, FWD


@lru_cache(maxsize=1)
def _bootstrap() -> dict:
    return json.loads((FPL_DIR / "bootstrap.json").read_text())


@lru_cache(maxsize=1)
def _players() -> dict[int, dict]:
    return {e["id"]: e for e in _bootstrap()["elements"]}


@lru_cache(maxsize=1)
def total_managers() -> int:
    return _bootstrap()["total_players"]


@lru_cache(maxsize=64)
def _live(gameweek: int) -> dict[int, dict]:
    """element_id -> stats dict for one gameweek (from the live snapshot)."""
    data = json.loads((FPL_DIR / "live" / f"gw{gameweek}.json").read_text())
    return {el["id"]: el["stats"] for el in data["elements"]}


def player(element_id: int) -> dict:
    return _players()[element_id]


def web_name(element_id: int) -> str:
    return _players()[element_id]["web_name"]


# --------------------------------------------------------------------------
# Name resolution: free-text captain pick -> element_id (or None if unsure).
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _name_index() -> dict[str, int]:
    """Map several normalized name forms to element_id.

    Covers web_name ("Haaland"), full name ("erling haaland"), and bare surname
    ("salah"), so a model echoing any of these from a note resolves cleanly.
    Collisions (a surname shared by two players) are dropped to force the fuzzy
    path rather than silently picking the wrong player.
    """
    forms: dict[str, list[int]] = {}

    def add(key: str, pid: int) -> None:
        k = key.strip().lower()
        if k:
            forms.setdefault(k, [])
            if pid not in forms[k]:
                forms[k].append(pid)

    for pid, e in _players().items():
        add(e["web_name"], pid)
        add(f"{e['first_name']} {e['second_name']}", pid)
        add(e["second_name"], pid)
    # only keep unambiguous keys
    return {k: v[0] for k, v in forms.items() if len(v) == 1}


def resolve_player(name: str | None) -> int | None:
    """Resolve a player name to an element_id, or None if not confident.

    Exact (normalized) match first; otherwise a tight fuzzy match against
    web_name. Returns None for empty input or anything below the fuzzy cutoff —
    the caller routes None to the extraction-error bucket, never to a 0-point
    football outcome.
    """
    if not name:
        return None
    key = name.strip().lower()
    index = _name_index()
    if key in index:
        return index[key]
    # fuzzy fallback against web_names only (tight cutoff)
    names = {e["web_name"].lower(): pid for pid, e in _players().items()}
    close = difflib.get_close_matches(key, list(names), n=1, cutoff=0.85)
    if close:
        return names[close[0]]
    # last resort: surname token match against web_name
    token = key.split()[-1]
    if token in names:
        return names[token]
    return None


# --------------------------------------------------------------------------
# Outcome oracle (post-deadline truth) — the answer key.
# --------------------------------------------------------------------------

def points(element_id: int, gameweek: int) -> int:
    """Actual FPL points scored by `element_id` in `gameweek` (0 if no record)."""
    stats = _live(gameweek).get(element_id)
    return int(stats["total_points"]) if stats else 0


def top_scorer(gameweek: int) -> tuple[int, int]:
    """The perfect-hindsight captain: (element_id, points) of the GW's top scorer."""
    live = _live(gameweek)
    pid = max(live, key=lambda p: live[p]["total_points"])
    return pid, int(live[pid]["total_points"])


def _starter_pool(gameweek: int, min_minutes_to_date: int = 270) -> list[int]:
    """Realistic captain universe for GW: attacking players who are regular
    starters going into the GW (minutes accrued in prior GWs). Outcome-blind —
    uses only pre-deadline minutes."""
    mins: dict[int, int] = {}
    for gw in range(1, gameweek):
        for pid, s in _live(gw).items():
            mins[pid] = mins.get(pid, 0) + s["minutes"]
    pool = [
        pid for pid, m in mins.items()
        if m >= min_minutes_to_date and player(pid)["element_type"] in ATTACKING_TYPES
    ]
    # GW1 has no prior minutes — fall back to priced-up attackers.
    if not pool:
        pool = [
            pid for pid, e in _players().items()
            if e["element_type"] in ATTACKING_TYPES and e["now_cost"] >= 70
        ]
    return pool


def pool_mean_points(gameweek: int) -> tuple[float, int]:
    """Random-captain floor: mean actual GW points over the starter pool.

    Returns (mean_points, pool_size). Deterministic expected value of a random
    pick — no sampling.
    """
    pool = _starter_pool(gameweek)
    if not pool:
        return 0.0, 0
    return sum(points(pid, gameweek) for pid in pool) / len(pool), len(pool)


# --------------------------------------------------------------------------
# Pre-deadline signals — safe to feed a decision (no GW-N outcome).
# --------------------------------------------------------------------------

def points_to_date(element_id: int, gameweek: int) -> int:
    """Total points in GWs strictly before `gameweek` (pre-deadline form)."""
    return sum(points(element_id, gw) for gw in range(1, gameweek))


def template_pick(gameweek: int) -> int:
    """The 'always-captain-template' baseline pick for `gameweek`.

    Defined as the season-to-date points leader among outfield players (the
    entrenched premium everyone owns). GW1 has no prior points, so it falls back
    to the highest-priced attacker. Both signals are pre-deadline.
    """
    if gameweek > 1:
        leaders = {
            pid: points_to_date(pid, gameweek)
            for pid, e in _players().items()
            if e["element_type"] in OUTFIELD_TYPES
        }
        return max(leaders, key=leaders.get)
    attackers = {
        pid: e["now_cost"]
        for pid, e in _players().items()
        if e["element_type"] in ATTACKING_TYPES
    }
    return max(attackers, key=attackers.get)
