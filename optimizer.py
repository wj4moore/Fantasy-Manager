# optimizer.py

# Solver: linear program to pick the best lineup.
import pulp
import pandas as pd

# Sets: which positions can occupy FLEX/SUPER_FLEX.
FLEX_SET = {"RB", "WR", "TE"}
SUPER_FLEX_SET = {"QB", "RB", "WR", "TE"}


# Main: optimize lineup under roster constraints (robust to shortages).
def optimize_lineup(players_df: pd.DataFrame, roster_constraints: dict) -> pd.DataFrame:
    """
    players_df columns (required): ['player_id', 'position', 'projected']
    roster_constraints example: {'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':2,'DEF':1}
    Returns the selected rows as a DataFrame (may be <= slots if infeasible).
    """
    df = players_df.copy().reset_index(drop=True)
    if df.empty:
        return df

    # Guard: ensure types/values are numeric and finite.
    df["projected"] = pd.to_numeric(df["projected"], errors="coerce").fillna(0.0)

    # LP: variables x_i ∈ {0,1} for each candidate.
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in df.index]
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)

    # Objective: maximize total projected points.
    prob += pulp.lpSum(x[i] * float(df.loc[i, "projected"]) for i in df.index)

    # Slots: total selected must equal total roster slots requested.
    total_slots = int(sum(int(v) for v in roster_constraints.values()))
    prob += pulp.lpSum(x) == total_slots

    # Fixed Caps: don't exceed requested count for each fixed position.
    # (Use ≤ to avoid infeasibility when the roster lacks enough players of a position.)
    for pos, cnt in roster_constraints.items():
        if pos in ("FLEX", "SUPER_FLEX"):
            continue
        cnt = int(cnt)
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == pos) <= cnt

    # FLEX Min: ensure enough total RB/WR/TE to cover fixed + FLEX.
    flex_required = int(roster_constraints.get("FLEX", 0))
    if flex_required > 0:
        fixed_in_flex_pool = sum(int(roster_constraints.get(p, 0)) for p in FLEX_SET)
        need_from_flex_pool = fixed_in_flex_pool + flex_required
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] in FLEX_SET) >= need_from_flex_pool

    # SUPER_FLEX Min: ensure enough total among QB+RB+WR+TE to cover fixed + SUPER_FLEX.
    sflex_required = int(roster_constraints.get("SUPER_FLEX", 0))
    if sflex_required > 0:
        fixed_in_sflex_pool = sum(int(roster_constraints.get(p, 0)) for p in SUPER_FLEX_SET)
        need_from_sflex_pool = fixed_in_sflex_pool + sflex_required
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] in SUPER_FLEX_SET) >= need_from_sflex_pool

    # Solve: silent CBC; falls back gracefully if infeasible.
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract: pick chosen rows (x_i > 0.5).
    selected = [df.loc[i].to_dict() for i in df.index if pulp.value(x[i]) > 0.5]
    return pd.DataFrame(selected)
