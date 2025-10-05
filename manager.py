from dotenv import load_dotenv
import os

# Env: Load .env into environment variables.
load_dotenv()

# Env Vars: Keys may be used later.
db_url = os.getenv("DATABASE_URL")
sleeper_key = os.getenv("SLEEPER_API_KEY")
espn_key = os.getenv("ESPN_API_KEY")
yahoo_key = os.getenv("YAHOO_API_KEY")

import argparse
import random
import yaml
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from sleeper_api import SleeperAPI
from optimizer import optimize_lineup
from projections import ProjectionModels
from features import FEATURE_COLS
from id_map import build_id_map_from_sleeper
from features_runtime import load_runtime_features

# Constants: Baselines and FLEX set.
POS_BASELINE = {"QB": 18.0, "RB": 12.0, "WR": 12.0, "TE": 9.0, "K": 8.0, "DEF": 7.0}
FLEX_SET = {"RB", "WR", "TE"}

# Utils: Load config file if present.
def load_config(path: str = "config.yml"):
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

# Utils: Normalize Sleeper roster positions.
def normalize_roster_slots(positions):
    """
    Roster Slots: Convert Sleeper roster positions to standard counts.
    """
    cnt = Counter()
    for p in positions or []:
        if p in ("BN", "IR", "TAXI"):
            continue
        if p in ("FLEX", "W/R/T", "RB/WR/TE"):
            cnt["FLEX"] += 1
        else:
            cnt[p] += 1
    return {k: int(v) for k, v in cnt.items()}

# Utils: Build constraints from league JSON.
def roster_constraints_from_league(league_json):
    return normalize_roster_slots(league_json.get("roster_positions", []))

# Utils: Normalize position strings.
def _norm_pos(x: str) -> str:
    """
    Position Norm: Map DST variants to DEF and uppercase.
    """
    x = (x or "").strip().upper().replace(" ", "")
    return {"D/ST": "DEF", "DST": "DEF"}.get(x, x)

# Utils: Build a small DataFrame for given player ids.
def build_player_df(players_map, ids):
    """
    Player DF: Turn a list of Sleeper ids into name/position rows.
    """
    rows = []
    for pid in ids or []:
        if not pid:
            continue
        p = players_map.get(pid, {})
        name = p.get("full_name") or f"{p.get('first_name','?')} {p.get('last_name','')}"
        rows.append({"player_id": pid, "name": name.strip(), "position": _norm_pos(p.get("position", "FLEX"))})
    return pd.DataFrame(rows)

# Projections: Simple fallback projection.
def naive_proj(df: pd.DataFrame):
    """
    Naive Proj: Baseline by position ± random noise.
    """
    df = df.copy()
    df["proj_mean"] = df["position"].map(POS_BASELINE).fillna(10.0) + [random.uniform(-1, 1) for _ in range(len(df))]
    df["proj_p20"] = df["proj_mean"] - 2.0
    df["proj_p80"] = df["proj_mean"] + 2.0
    return df

# Projections: Use trained models when available else naive.
def project_players(candidate_df: pd.DataFrame, models: ProjectionModels):
    """
    Project Players: Try model projections else fallback.
    """
    feats = candidate_df.copy()
    for col in FEATURE_COLS:
        if col not in feats.columns:
            feats[col] = 0.0
    cols = ["player_id", "name", "position"] + FEATURE_COLS
    feats = feats[[c for c in cols if c in feats.columns]]
    if models.has_models():
        return models.project(feats)
    else:
        return naive_proj(candidate_df)

# Free Agents: Filter a plausible FA pool.
def compute_free_agents(rosters, players_map, allowed_positions):
    """
    Free Agents: Unrostered, allowed positions, likely active/team-coded.
    """
    rostered = {pid for r in rosters for pid in (r.get("players") or []) if pid}
    rows = []
    for pid, meta in players_map.items():
        if not isinstance(meta, dict):
            continue
        if pid in rostered:
            continue
        pos = _norm_pos(meta.get("position", ""))
        team = meta.get("team")
        active = meta.get("active", True)
        if pos in allowed_positions and (team or pos in {"RB","WR","TE","QB","DEF","K"}):
            name = meta.get("full_name") or f"{meta.get('first_name','?')} {meta.get('last_name','')}"
            rows.append({"player_id": pid, "name": name.strip(), "position": pos})
    return pd.DataFrame(rows)

# Greedy: Deterministic fallback lineup.
def greedy_lineup(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    Greedy: Fill fixed slots then FLEX in projected order.
    """
    if df.empty:
        return df.copy()
    fixed_order = ["QB", "RB", "WR", "TE", "K", "DEF"]
    remaining = df.sort_values("projected", ascending=False).copy()
    picks = []
    for pos in fixed_order:
        need = int(constraints.get(pos, 0))
        if need <= 0:
            continue
        cand = remaining[remaining["position"] == pos].head(need)
        if not cand.empty:
            picks.append(cand)
            remaining = remaining.drop(cand.index)
    flex_need = int(constraints.get("FLEX", 0))
    if flex_need > 0:
        flex_cand = remaining[remaining["position"].isin(FLEX_SET)].head(flex_need)
        if not flex_cand.empty:
            picks.append(flex_cand)
            remaining = remaining.drop(flex_cand.index)
    if picks:
        return pd.concat(picks, ignore_index=False)
    total = sum(int(v) for v in constraints.values())
    return remaining.head(total)

# Filler: Ensure exact total and fixed mins.
def force_fill_to_slots(selected_df: pd.DataFrame, all_candidates: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    Force Fill: Hit exact total and satisfy fixed slots before FLEX.
    """
    need_total = sum(int(v) for v in constraints.values())
    chosen = (selected_df if selected_df is not None else pd.DataFrame(columns=all_candidates.columns)).copy()
    for c in {"player_id", "position", "projected"} - set(chosen.columns):
        chosen[c] = pd.Series(dtype=all_candidates[c].dtype if c in all_candidates.columns else "object")
    remaining = all_candidates[~all_candidates["player_id"].isin(chosen.get("player_id", pd.Series([], dtype=str)))].copy()
    remaining = remaining.sort_values("projected", ascending=False)

    def take_from_pool(pos_filter, count):
        nonlocal chosen, remaining
        if count <= 0:
            return
        if isinstance(pos_filter, (set, list, tuple)):
            cand = remaining[remaining["position"].isin(pos_filter)].head(count)
        else:
            cand = remaining[remaining["position"] == pos_filter].head(count)
        if not cand.empty:
            chosen = pd.concat([chosen, cand], ignore_index=True)
            remaining = remaining[~remaining["player_id"].isin(cand["player_id"])]

    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        want = int(constraints.get(pos, 0))
        if want <= 0:
            continue
        have = int((chosen["position"] == pos).sum())
        need = want - have
        if need > 0:
            take_from_pool(pos, need)

    flex_want = int(constraints.get("FLEX", 0))
    if flex_want > 0:
        have_flex_pool = int(chosen["position"].isin(FLEX_SET).sum())
        base_fixed = sum(int(constraints.get(p, 0)) for p in FLEX_SET)
        need_in_flex_set = base_fixed + flex_want - have_flex_pool
        if need_in_flex_set > 0:
            take_from_pool(FLEX_SET, need_in_flex_set)

    short = need_total - len(chosen)
    if short > 0:
        take_from_pool({"QB", "RB", "WR", "TE", "K", "DEF"}, short)

    if len(chosen) > need_total:
        chosen = chosen.sort_values("projected", ascending=False).head(need_total)
    return chosen

# Enforcer: Keep minimum counts per fixed slot.
def enforce_fixed_slots(optimized: pd.DataFrame, opt_input: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    Fixed Slots: Swap to ensure all fixed positions meet their minimums.
    """
    fixed_order = ["QB", "RB", "WR", "TE", "K", "DEF"]
    total_slots = sum(int(v) for v in constraints.values())

    def pos_count(df, pos):
        return int((df["position"] == pos).sum())

    chosen = optimized.copy()

    def remaining_pool():
        return opt_input[~opt_input["player_id"].isin(chosen["player_id"])].copy()

    for pos in fixed_order:
        need = int(constraints.get(pos, 0))
        if need <= 0:
            continue
        have = pos_count(chosen, pos)
        if have >= need:
            continue
        need_more = need - have
        pool = remaining_pool()
        add_pos = pool[pool["position"] == pos].sort_values("projected", ascending=False).head(need_more)
        if add_pos.empty:
            continue
        to_swap = []
        current = chosen.sort_values("projected", ascending=True).copy()
        removed_counts = Counter()
        for _, row in current.iterrows():
            p = row["position"]
            if p == pos:
                continue
            min_p = int(constraints.get(p, 0)) if p in fixed_order else 0
            future_p_count = pos_count(chosen, p) - removed_counts[p] - 1
            if p in fixed_order and future_p_count < min_p:
                continue
            to_swap.append(row["player_id"])
            removed_counts[p] += 1
            if len(to_swap) >= len(add_pos):
                break
        if len(to_swap) < len(add_pos):
            extra_needed = len(add_pos) - len(to_swap)
            extra_pool = current[~current["player_id"].isin(to_swap) & (current["position"] != pos)]
            to_swap += extra_pool.head(extra_needed)["player_id"].tolist()
        chosen = pd.concat([chosen[~chosen["player_id"].isin(to_swap)], add_pos], ignore_index=True) \
                 .sort_values("projected", ascending=False)
        if len(chosen) > total_slots:
            chosen = chosen.head(total_slots)
    return chosen

# Labels: Assign QB1/RB1/.../FLEX1 labels.
def assign_slots(selected_df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    Slot Labels: Tag starters with human-readable slot names.
    """
    if selected_df is None or selected_df.empty:
        return selected_df
    need_total = sum(int(v) for v in constraints.values())
    df = selected_df.copy().sort_values("projected", ascending=False)
    if "slot" not in df.columns:
        df["slot"] = ""

    def take_exact(pos: str, count: int) -> int:
        if count <= 0:
            return 0
        mask = (df["slot"] == "") & (df["position"] == pos)
        picked_idx = df[mask].head(count).index.tolist()
        for i, idx in enumerate(picked_idx, start=1):
            df.at[idx, "slot"] = f"{pos}{i}"
        return len(picked_idx)

    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        take_exact(pos, int(constraints.get(pos, 0)))

    flex_need = int(constraints.get("FLEX", 0))
    if flex_need > 0:
        mask_flex = (df["slot"] == "") & (df["position"].isin(FLEX_SET))
        flex_idx = df[mask_flex].head(flex_need).index.tolist()
        for i, idx in enumerate(flex_idx, start=1):
            df.at[idx, "slot"] = f"FLEX{i}"
        short = flex_need - len(flex_idx)
        if short > 0:
            backfill_idx = df[df["slot"] == ""].head(short).index.tolist()
            for k, idx in enumerate(backfill_idx, start=len(flex_idx) + 1):
                df.at[idx, "slot"] = f"FLEX{k}"

    leftover_idx = df.index[df["slot"] == ""].tolist()
    for j, idx in enumerate(leftover_idx, start=1):
        df.at[idx, "slot"] = f"BENCH{j}"

    starters = df[df["slot"].str.startswith(("QB", "RB", "WR", "TE", "K", "DEF", "FLEX"))]
    if len(starters) > need_total:
        starters = starters.head(need_total)
    return starters

# Roster: Resolve which roster belongs to you.
def resolve_my_roster(api: SleeperAPI, league_id: str, rosters, want_user: str | None, owner_id: str | None):
    """
    My Roster: Resolve a single roster by username or owner_id.
    """
    users = api.get_league_users(league_id)
    user_by_id = {u.get("user_id"): u for u in users}
    def norm(s): return (s or "").strip().lower()
    if owner_id:
        for r in rosters:
            if r.get("owner_id") == owner_id:
                return r
    if want_user:
        want = norm(want_user)
        for u in users:
            if norm(u.get("username")) == want or norm(u.get("display_name")) == want:
                uid = u.get("user_id")
                for r in rosters:
                    if r.get("owner_id") == uid:
                        return r
                break
    table = []
    for r in rosters:
        oid = r.get("owner_id")
        u = user_by_id.get(oid, {})
        table.append({
            "owner_id": oid,
            "username": u.get("username"),
            "display_name": u.get("display_name"),
            "roster_id": r.get("roster_id"),
            "players_count": len(r.get("players") or []),
        })
    print("\n[error] Could not resolve your roster. Available owners in this league:\n")
    print(pd.DataFrame(table).to_string(index=False))
    raise SystemExit("\nPass one of:  --owner_id <owner_id>   or   --user <username_or_display_name>\n")

# Run Week: Orchestrate data, features, projections, and lineup.
def run_week(league_id: str, week: int, user: str | None = None, owner_id: str | None = None,
             models_dir: str = "models", waivers_top: int = 10, test_features: bool = False):
    s = SleeperAPI()

    league = s.get_league(league_id)
    rosters = s.get_rosters(league_id)
    players_map = s.get_players()

    my_roster = resolve_my_roster(s, league_id, rosters, want_user=user, owner_id=owner_id)

    starters_ids = my_roster.get("starters", []) or []
    bench_ids = [pid for pid in (my_roster.get("players") or []) if pid not in starters_ids]

    starters_df = build_player_df(players_map, starters_ids)
    bench_df = build_player_df(players_map, bench_ids)
    candidate_df = pd.concat([starters_df, bench_df], ignore_index=True)
    candidate_df["position"] = candidate_df["position"].astype(str).map(_norm_pos)
    candidate_df = candidate_df[candidate_df["position"].isin(["QB", "RB", "WR", "TE", "K", "DEF"])].drop_duplicates("player_id")
    candidate_df = pd.concat([starters_df, bench_df], ignore_index=True)
    candidate_df = candidate_df[candidate_df["position"].isin(["QB","RB","WR","TE","K","DEF"])].drop_duplicates("player_id")

    # Runtime Features: Load per-week features with optional test forcing.
    raw_season = league.get("season")
    try:
        season = int(raw_season)
    except Exception:
        season = 2024
    if test_features:
        season_for_feats = 2024
        week_for_feats   = 10
        print(f"[test] Forcing runtime feature join to season={season_for_feats}, week={week_for_feats}")
    else:
        season_for_feats = season
        week_for_feats   = week

    rt_feats, _sched = load_runtime_features(season=season_for_feats, week=week_for_feats)

    if rt_feats is None or rt_feats.empty:
        last_err = getattr(rt_feats, "_last_error", None) if rt_feats is not None else None
        print(f"[warn] runtime features unavailable for tries around season={season_for_feats}, week={week_for_feats}; "
              f"falling back to baselines. Last error: {last_err}")
        candidate_df["is_home"] = 0.0
        candidate_df["opp_pass_rank"] = 16.0
        candidate_df["opp_rush_rank"] = 16.0
        candidate_df["rolling_fp_3"] = 0.0
        candidate_df["rolling_oppty_3"] = 0.0
        candidate_df["vegas_total"] = 44.0
        candidate_df["weather_wind"] = 0.0
    else:
        print(f"[info] Using runtime features (season={season_for_feats}, week={week_for_feats}, rows={len(rt_feats)})")

        # ---- PATCH START (Collision-safe join with rt_feats) ----
        # Id Map: Build Sleeper↔GSIS map.
        idmap = build_id_map_from_sleeper(players_map)
        if idmap is None or idmap.empty:
            idmap = pd.DataFrame(columns=["sleeper_id", "gsis_id", "name_key", "position"])
        idmap_subset = idmap[["sleeper_id", "gsis_id", "name_key", "position"]].rename(
            columns={"position": "position_idmap"}
        )

        # Merge 1: Attach ids without creating position collisions.
        cand = candidate_df.rename(columns={"position": "position_cand"}).merge(
            idmap_subset,
            left_on="player_id",
            right_on="sleeper_id",
            how="left",
        )

        # Position: Recreate a single position column.
        cand["position"] = cand.get("position_cand")
        if "position_idmap" in cand.columns:
            cand["position"] = cand["position"].fillna(cand["position_idmap"])

        # Keys: Build a robust name key on candidate side.
        name_src = cand.get("name").fillna("")
        pos_src  = cand.get("position").astype(str).str.upper()
        cand["name_key_cand"] = (
            name_src.astype(str)
            .str.lower()
            .str.replace(r"[^a-z\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            + "|"
            + pos_src
        )

        # Keys: Build a robust name key on rt_feats side.
        RUNTIME_COLS = ["is_home", "opp_pass_rank", "opp_rush_rank",
                        "rolling_fp_3", "rolling_oppty_3", "vegas_total", "weather_wind"]
        rt_feats = rt_feats.copy()
        rt_feats["name_key"] = (
            rt_feats.get("player_name", "").astype(str).str.lower()
            .str.replace(r"[^a-z\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            + "|"
            + rt_feats.get("position", "").astype(str).str.upper()
        )
        rt_keep = ["gsis_id", "name_key"] + [c for c in RUNTIME_COLS if c in rt_feats.columns]
        rt_sub  = rt_feats[rt_keep].copy()

        # Merge 2: Primary join on gsis_id (no collisions now).
        cand = cand.merge(rt_sub.drop(columns=["name_key"]), on="gsis_id", how="left")

        # Merge 3: Fallback join on name_key for missing rows.
        missing_mask = cand["is_home"].isna() if "is_home" in cand.columns else pd.Series(True, index=cand.index)
        if missing_mask.any():
            cand = cand.merge(
                rt_sub, left_on="name_key_cand", right_on="name_key", how="left", suffixes=("", "_rt2")
            )
            for col in RUNTIME_COLS:
                src = col if col in cand.columns else f"{col}_rt2"
                if src in cand.columns:
                    cand[col] = pd.to_numeric(cand.get(col), errors="coerce") \
                                    .fillna(pd.to_numeric(cand[src], errors="coerce"))
            cand = cand.drop(columns=["name_key"], errors="ignore")

        # Defaults: Fill missing features safely.
        defaults = {
            "is_home": 0.0,
            "opp_pass_rank": 16.0,
            "opp_rush_rank": 16.0,
            "rolling_fp_3": 0.0,
            "rolling_oppty_3": 0.0,
            "vegas_total": 44.0,
            "weather_wind": 0.0,
        }
        for k, v in defaults.items():
            cand[k] = pd.to_numeric(cand.get(k), errors="coerce").fillna(v)

        # Replace candidate_df with enriched version.
        candidate_df = cand
        # ---- PATCH END ----

    # Projections: Use models or baselines.
    models = ProjectionModels(models_dir=models_dir)
    proj_df = project_players(candidate_df, models)

    # Team Slots: Keep DEF/K baselines.
    team_mask = candidate_df["position"].isin(["DEF", "K"])
    if team_mask.any():
        proj_df.loc[team_mask, "proj_mean"] = candidate_df.loc[team_mask, "position"].map(POS_BASELINE).values

    # Floors: Replace non-finite or tiny projections with positional baselines.
    baseline_vec = candidate_df["position"].map(POS_BASELINE).fillna(8.0).values
    low_or_nan = ~np.isfinite(proj_df["proj_mean"]) | (proj_df["proj_mean"] < 1.0)
    if low_or_nan.any():
        proj_df.loc[low_or_nan, "proj_mean"] = baseline_vec[low_or_nan]

    # Optimizer Input: Merge projections and clean.
    opt_input = candidate_df.merge(proj_df[["player_id", "proj_mean"]], on="player_id", how="left")
    opt_input = opt_input.rename(columns={"proj_mean": "projected"})
    opt_input["projected"] = pd.to_numeric(opt_input["projected"], errors="coerce").fillna(8.0)
    opt_input["projected"] = opt_input["projected"].replace([np.inf, -np.inf], 8.0)
    opt_input["projected"] = opt_input["projected"].clip(lower=0.0)
    opt_input["position"] = opt_input["position"].astype(str).map(_norm_pos)

    # Constraints: Build from league config.
    constraints = roster_constraints_from_league(league)
    print("\n[info] Roster slots:", constraints)
    allowed_positions = set(constraints.keys()) | {"RB", "WR", "TE"}
    opt_input = opt_input[opt_input["position"].isin(allowed_positions)]

    # Injuries: Adjust projections if flagged.
    from injury_rules import apply_injury_adjustments
    opt_input = apply_injury_adjustments(
        opt_input.rename(columns={"proj_mean":"projected"}) if "proj_mean" in opt_input.columns else opt_input,
        players_map=players_map,
        injury_csv=None,
        verbose=True
    )
    if "projected_adj" in opt_input.columns:
        opt_input["projected"] = opt_input["projected_adj"]

    # Optimize: Try ILP, fallback to greedy.
    try:
        optimized = optimize_lineup(opt_input, constraints)
    except Exception as e:
        print("\n[warn] Optimizer failed, using greedy fallback:", e)
        optimized = greedy_lineup(opt_input, constraints)
    if (optimized is None) or optimized.empty or ("player_id" not in optimized.columns):
        print("\n[warn] Optimizer returned no lineup; using greedy fallback.")
        optimized = greedy_lineup(opt_input, constraints)

    # Columns: Ensure helpful columns exist.
    if not {"name", "position", "projected"}.issubset(optimized.columns):
        optimized = optimized.merge(
            opt_input[["player_id", "name", "position", "projected"]],
            on="player_id",
            how="left"
        )

    # Guards: Exact total and fixed mins.
    optimized = force_fill_to_slots(optimized, opt_input, constraints)
    optimized = enforce_fixed_slots(optimized, opt_input, constraints)

    # Flags: Mark current starters.
    opt_input["is_current_starter"] = opt_input["player_id"].isin(starters_ids)
    if "is_current_starter" not in optimized.columns:
        optimized = optimized.merge(
            opt_input[["player_id", "is_current_starter"]],
            on="player_id",
            how="left",
        )

    # Output: Assign slots and print starters.
    with_slots = assign_slots(optimized, constraints)
    print("\n=== Recommended Starters (with slots) ===")
    cols = [c for c in ["slot", "name", "position", "projected", "is_current_starter"] if c in with_slots.columns]
    print(with_slots.sort_values(["slot"]).to_string(index=False, columns=cols))

    # Output: Show bench moves.
    dropped = [pid for pid in starters_ids if pid not in set(with_slots["player_id"]) if pid]
    if dropped:
        dropped_df = build_player_df(players_map, dropped)
        print("\n=== Moved to Bench ===")
        print(dropped_df[["name", "position"]].to_string(index=False))

    # Waivers: Rank free agents vs replacement levels.
    from waivers import replacement_levels, rank_waivers
    try:
        fa_df = compute_free_agents(rosters, players_map, allowed_positions)
        print(f"[debug] Free-agent pool size: {0 if fa_df is None else len(fa_df)}")
        if fa_df is not None and not fa_df.empty:
            fa_proj = project_players(fa_df, models)
            fa = fa_df.merge(fa_proj[["player_id","proj_mean"]], on="player_id", how="left")

            POS_BASELINE_LOCAL = {"QB":18.0,"RB":12.0,"WR":12.0,"TE":9.0,"K":8.0,"DEF":7.0}
            fa["position"] = fa["position"].astype(str).str.upper()
            mask_team = fa["position"].isin(["DEF","K"])
            fa.loc[mask_team, "proj_mean"] = fa.loc[mask_team, "position"].map(POS_BASELINE_LOCAL).values
            fa["proj_mean"] = pd.to_numeric(fa["proj_mean"], errors="coerce")
            baseline_vec_fa = fa["position"].map(POS_BASELINE_LOCAL).fillna(8.0).values
            low_or_nan_fa = ~np.isfinite(fa["proj_mean"]) | (fa["proj_mean"] < 1.0)
            if low_or_nan_fa.any():
                fa.loc[low_or_nan_fa, "proj_mean"] = baseline_vec_fa[low_or_nan_fa]
            fa = fa.rename(columns={"proj_mean": "projected"})

            fa = apply_injury_adjustments(fa, players_map=players_map, verbose=False)
            if "projected_adj" in fa.columns:
                fa["projected"] = fa["projected_adj"]

            opt_input["projected"] = pd.to_numeric(opt_input["projected"], errors="coerce") \
                                        .replace([np.inf,-np.inf], np.nan).fillna(8.0).clip(lower=0.0)
            repl = replacement_levels(opt_input, quantile=0.25)
            print(f"[debug] Replacement rows by position:\n{repl.to_string(index=False)}")

            waiver_table = rank_waivers(
                fa, repl,
                budget_remaining=100,
                aggressiveness=0.35,
                top=waivers_top*2
            )
            if "PAR" in waiver_table.columns:
                waiver_table = waiver_table[waiver_table["PAR"] > 0].head(waivers_top)

            if waiver_table.empty:
                print("\n[waivers] No positive upgrades found right now.")
            else:
                print("\n=== Top Waiver Targets (PAR & FAAB Suggestion) ===")
                print(waiver_table.to_string(index=False))
        else:
            print("\n[waivers] No free agents found after filtering.")
    except Exception as e:
        print("\n[waivers] Skipped due to:", e)

    print("\nTip: pass --user <username_or_display_name> or --owner_id <owner_id> to bind to your exact roster.\n")

# CLI: Parse args and run.
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--league", required=True)
    p.add_argument("--week", required=True, type=int)
    p.add_argument("--user", required=False, help="Sleeper username or display_name (case-insensitive)")
    p.add_argument("--owner_id", required=False, help="Sleeper owner/user_id")
    p.add_argument("--models_dir", default="models")
    p.add_argument("--waivers", dest="waivers_top", type=int, default=10)
    p.add_argument("--test_features", action="store_true",
                   help="Force runtime feature join to season=2024, week=10 for sanity check")
    args = p.parse_args()
    run_week(args.league, args.week, user=args.user, owner_id=args.owner_id,
             models_dir=args.models_dir, waivers_top=args.waivers_top,
             test_features=args.test_features)
