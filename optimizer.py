import pulp
import pandas as pd

FLEX_SET = {"RB", "WR", "TE"}
SUPER_FLEX_SET = {"QB", "RB", "WR", "TE"}

def optimize_lineup(players_df: pd.DataFrame, roster_constraints: dict) -> pd.DataFrame:
    """
    players_df: columns ['player_id','position','projected', ...]
    roster_constraints: e.g. {'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':1,'SUPER_FLEX':1}
    """
    df = players_df.copy().reset_index(drop=True)
    if df.empty:
        return df

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in df.index]

    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    # Objective: maximize projected points
    prob += pulp.lpSum(x[i] * float(df.loc[i, "projected"]) for i in df.index)

    # Total players chosen must equal total slots
    total_slots = sum(int(v) for v in roster_constraints.values())
    prob += pulp.lpSum(x) == total_slots

    # Fixed-position equality constraints
    for pos, cnt in roster_constraints.items():
        if pos in ("FLEX", "SUPER_FLEX"):
         continue
    prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == pos) == int(cnt)

    # FLEX: Exactly 'FLEX' more from RB/WR/TE on top of the fixed RB/WR/TE
    if "FLEX" in roster_constraints and roster_constraints["FLEX"] > 0:
        flex_required = int(roster_constraints["FLEX"])
        # Number selected from FLEX_SET must be >= fixed_in_set + flex_required
        fixed_in_set = sum(int(roster_constraints.get(p, 0)) for p in {"RB", "WR", "TE"})
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] in {"RB", "WR", "TE"}) >= fixed_in_set + flex_required
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [df.loc[i].to_dict() for i in df.index if pulp.value(x[i]) > 0.5]
    return pd.DataFrame(selected)
