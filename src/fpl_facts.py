"""Deterministic pre-deadline fact builder for corpus notes (the fact-first guard).

Every `fpl-derived` corpus note must be built from real facts that were knowable
BEFORE the gameweek's deadline. This module reconstructs exactly those facts from
the frozen snapshot, so a note's claims can be traced to a reproducible source
rather than written from memory:

  - the real deadline (from data/deadlines.json, regenerated from bootstrap),
  - a form-to-date window over the last N fully-complete gameweeks, with a check
    that the last fixture of that window kicked off before the deadline (no
    in-progress-round leakage),
  - fixture + FDR for each candidate's team that gameweek,
  - point-in-time ownership + price, taken from the per-player history snapshot,
  - the crowd's actual armband (`most_captained`) for the gameweek.

Ownership is only reconstructable for the players whose element-summary files were
snapshotted (the API wipes on rollover). Those ids are OWNERSHIP_PLAYERS; passing
any other id raises, so a note can never quote an ownership number we cannot
source. Everything else (form, fixture, FDR, most_captained) is available for all
players and all gameweeks.

This is a build-time helper for authoring notes, not part of the RAG/eval path.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpl_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FPL_DIR = ROOT / "data" / "fpl"
DEADLINES = ROOT / "data" / "deadlines.json"

# Players whose per-gameweek ownership (`selected`) is in the snapshot. A note may
# only quote ownership for these; anything else is not reconstructable post-rollover.
OWNERSHIP_PLAYERS = {
    21: "Rice", 82: "Semenyo", 236: "Neto", 237: "Enzo", 241: "Caicedo",
    381: "M.Salah", 414: "Foden", 430: "Haaland", 488: "Bruno G.", 582: "Kudus",
}

FDR_FIELD_H = "team_h_difficulty"
FDR_FIELD_A = "team_a_difficulty"


@lru_cache(maxsize=1)
def _bootstrap() -> dict:
    return json.loads((FPL_DIR / "bootstrap.json").read_text())


@lru_cache(maxsize=1)
def _teams() -> dict[int, str]:
    return {t["id"]: t["short_name"] for t in _bootstrap()["teams"]}


@lru_cache(maxsize=1)
def _events() -> dict[int, dict]:
    return {e["id"]: e for e in _bootstrap()["events"]}


@lru_cache(maxsize=1)
def _fixtures() -> list[dict]:
    return json.loads((FPL_DIR / "fixtures.json").read_text())


@lru_cache(maxsize=1)
def _deadlines() -> dict[str, str]:
    return json.loads(DEADLINES.read_text())


@lru_cache(maxsize=None)
def _history(element_id: int) -> dict[int, dict]:
    """round -> history row for one player (selected, value, etc.)."""
    data = json.loads((FPL_DIR / "element" / f"{element_id}.json").read_text())
    return {r["round"]: r for r in data["history"]}


def _last_kickoff(gameweek: int) -> str | None:
    ks = [f["kickoff_time"] for f in _fixtures()
          if f.get("event") == gameweek and f.get("kickoff_time")]
    return max(ks) if ks else None


def deadline(gameweek: int) -> str:
    return _deadlines()[str(gameweek)]


def form_window(gameweek: int, n: int = 4) -> dict:
    """The last `n` complete gameweeks before `gameweek`, with a leakage check.

    Returns the window GWs, whether the window's final round finished before the
    deadline (so its results were knowable pre-deadline), and the deadline.
    """
    gws = [gw for gw in range(max(1, gameweek - n), gameweek)]
    last_gw = gws[-1] if gws else None
    last_ko = _last_kickoff(last_gw) if last_gw else None
    dl = deadline(gameweek)
    # day-granular comparison matches the retrieval filter (date_int = YYYYMMDD)
    window_complete = bool(last_ko) and last_ko[:10] < dl
    return {
        "window_gws": gws,
        "last_window_gw": last_gw,
        "last_window_kickoff": last_ko,
        "deadline": dl,
        "window_complete_pre_deadline": window_complete,
    }


def fixture(gameweek: int, team_id: int) -> dict | None:
    for f in _fixtures():
        if f.get("event") != gameweek:
            continue
        if f["team_h"] == team_id:
            return {"home": True, "opp": _teams()[f["team_a"]], "fdr": f[FDR_FIELD_H]}
        if f["team_a"] == team_id:
            return {"home": False, "opp": _teams()[f["team_h"]], "fdr": f[FDR_FIELD_A]}
    return None


def ownership(element_id: int, gameweek: int) -> dict:
    """Point-in-time ownership going into `gameweek` (squads as locked at deadline)."""
    if element_id not in OWNERSHIP_PLAYERS:
        raise KeyError(
            f"element {element_id} has no snapshot ownership; a note may only quote "
            f"ownership for {sorted(OWNERSHIP_PLAYERS)}"
        )
    row = _history(element_id).get(gameweek)
    total = fpl_data.total_managers()
    if not row:
        return {"selected": None, "pct": None, "value": None}
    return {
        "selected": row["selected"],
        "pct": round(100 * row["selected"] / total, 1),
        "value": row["value"] / 10,
    }


def player_facts(element_id: int, gameweek: int, n: int = 4) -> dict:
    e = fpl_data.player(element_id)
    win = form_window(gameweek, n)["window_gws"]
    per = {gw: fpl_data.points(element_id, gw) for gw in win}
    fx = fixture(gameweek, e["team"])
    own = ownership(element_id, gameweek) if element_id in OWNERSHIP_PLAYERS else None
    return {
        "element_id": element_id,
        "name": e["web_name"],
        "team": _teams()[e["team"]],
        "pos": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]],
        "form_window_pts": per,
        "form_window_sum": sum(per.values()),
        "season_to_date_pts": fpl_data.points_to_date(element_id, gameweek),
        "fixture": fx,
        "ownership": own,
    }


def gw_facts(gameweek: int, element_ids: list[int], n: int = 4) -> dict:
    win = form_window(gameweek, n)
    mc = _events()[gameweek].get("most_captained")
    return {
        "gameweek": gameweek,
        "deadline": win["deadline"],
        "form_window": win,
        "most_captained": {"element_id": mc, "name": fpl_data.web_name(mc)} if mc else None,
        "players": [player_facts(pid, gameweek, n) for pid in element_ids],
    }


def _print(facts: dict) -> None:
    gw = facts["gameweek"]
    win = facts["form_window"]
    print(f"GW{gw}  deadline {facts['deadline']}")
    print(f"  form window: GW{win['window_gws'][0]}-{win['window_gws'][-1]} "
          f"(last round GW{win['last_window_gw']} last kickoff {win['last_window_kickoff']}) "
          f"-> complete pre-deadline: {win['window_complete_pre_deadline']}")
    mc = facts["most_captained"]
    print(f"  most_captained (crowd template): {mc['name']} ({mc['element_id']})" if mc else "  most_captained: n/a")
    for p in facts["players"]:
        fx = p["fixture"]
        fxs = (f"{'H' if fx['home'] else 'A'} {fx['opp']} (FDR {fx['fdr']})") if fx else "no fixture"
        own = p["ownership"]
        owns = f"own {own['pct']}% £{own['value']}m" if own and own["pct"] is not None else "own n/a"
        print(f"    {p['name']:<10} {p['pos']} {p['team']:<4} | {fxs:<18} | "
              f"form {p['form_window_pts']}={p['form_window_sum']} | season {p['season_to_date_pts']} | {owns}")


if __name__ == "__main__":
    # CLI: python src/fpl_facts.py <gameweek> <id> <id> ...
    gw = int(sys.argv[1])
    ids = [int(x) for x in sys.argv[2:]]
    _print(gw_facts(gw, ids))
