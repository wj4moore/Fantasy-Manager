# 🏈 Fantasy Manager

Fantasy Manager is an advanced Python-based toolkit for **Sleeper fantasy football leagues**.  
It fetches live data, builds features, trains models, and recommends optimal lineups, waivers, and FAAB bids using modern data science workflows.

---

## 🚀 Features Overview

### 📊 Core Functionality
- **Real-time runtime features** via `nfl_data_py` and `nflreadpy`
- **Machine learning models** for fantasy point projections (Gradient Boosting)
- **Per-position models** (QB, RB, WR, TE)
- **Rolling opportunity and performance features**
- **Opponent strength (pass/rush rank)** and schedule context
- **Waiver wire evaluator** with FAAB suggestions
- **Injury adjustment engine** based on status/practice trends
- **Lineup optimizer** using linear programming (via PuLP)

---

## 🧱 Project Structure

```
Fantasy-Manager/
│
├── manager.py             # Main runtime script to fetch, project, and optimize
├── data_pipeline.py       # Utils: loads play-by-play data using nflreadpy
├── features.py            # Builds training features from pbp data
├── features_runtime.py    # Builds runtime (weekly) features for current season
├── projections.py         # Loads models and generates per-player projections
├── train.py               # Trains quantile models (p20/p50/p80) and saves joblibs
├── optimizer.py           # Optimizes lineup selection
├── waivers.py             # Computes waiver values, replacement levels, FAAB bids
├── id_map.py              # Maps Sleeper player IDs to nflverse/GSIS IDs
├── injury_rules.py        # Injury multipliers and adjustments
├── sleeper_api.py         # Sleeper API wrapper for leagues, rosters, players
├── .env                   # (optional) Environment variables like API keys
├── config.yml             # League configuration (see below)
└── models/                # Saved model files (.joblib)
```

---

## ⚙️ Required Files

### `config.yml`
This file contains basic settings used by `manager.py`:

```yaml
league_id: "123456789012345678"
season: 2025
week: auto
user: "YourSleeperUsername"
models_dir: "models"
faab_budget: 100
aggressiveness: 0.35
```

### `.env`
Used for any API keys (optional):

```
OWM_API_KEY=your_openweathermap_key_here
```

---

## 🧰 Installation

```bash
# Clone the repo
git clone https://github.com/wj4moore/Fantasy-Manager.git
cd Fantasy-Manager

# (Recommended) Create virtual environment
python -m venv .venv311
.venv311\Scripts\activate  # (Windows)
# or
source .venv311/bin/activate  # (Mac/Linux)

# Install dependencies
pip install -r requirements.txt

# Core libraries (if no requirements file)
pip install pandas numpy requests pulp joblib scikit-learn nfl_data_py "nflreadpy[pandas]" pyyaml
```

---

## 🧠 Training Models (Historic Seasons)

To generate your models from previous years’ play-by-play data:

```bash
python train.py --years 2019 2020 2021 2022 2023 2024
```

This builds position models in the `/models` directory:
```
models/
├── QB_mean.joblib
├── RB_mean.joblib
├── WR_mean.joblib
├── TE_mean.joblib
...
```

---

## 🏃 Running the Current Week

### Automatically detect current season/week
```bash
python manager.py --league <LEAGUE_ID> --user <YourSleeperUsername>
```

### Manually specify season/week
```bash
python manager.py --league <LEAGUE_ID> --season 2025 --week 5 --user <YourSleeperUsername>
```

---

## 📅 Testing Historical Data

To test a past season and week:
```bash
python manager.py --league <LEAGUE_ID> --season 2023 --week 10 --user <YourSleeperUsername>
```

---

## 🔍 Outputs

When running `manager.py`, you’ll see:

```
[info] Roster slots: {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2, 'DEF': 1}

=== Recommended Starters (with slots) ===
 slot          name           position  projected  is_current_starter
 QB1           J. Herbert     QB        18.5       True
 RB1           R. Stevenson   RB        14.3       True
 FLEX1         J. Palmer      WR        12.0       False
...

=== Moved to Bench ===
 name                position
 Kyren Williams       RB
 Marvin Harrison      WR

[waivers] No positive upgrades found right now.
```

---

## 💡 How It Works

### Step-by-step Flow
1. **Sleeper API** → Pulls league, rosters, and players.
2. **Runtime Features** → Loads nfl_data_py weekly stats and builds features.
3. **Model Inference** → Uses trained models to predict mean/p20/p80 projections.
4. **Injury Adjustments** → Multiplies projected stats by injury multipliers.
5. **Optimizer** → Solves best lineup via PuLP linear programming.
6. **Waivers** → Computes replacement-level baselines, PAR, and FAAB suggestions.

---

## 🩻 Injury and Practice Rules
- `injury_rules.py` automatically adjusts projections for `OUT`, `Q`, `D`, etc.
- Practice participation (`DNP`, `LP`, `FP`) can optionally be added via CSV.

---

## 💰 Waiver Logic
- Calculates **PAR (Points Above Replacement)** per position.
- Suggests FAAB bids using:
  ```
  bid = PAR * (aggressiveness * (budget_remaining / 25))
  ```
- Filtered for positive-upgrade candidates.

---

## 🧩 Helpful Commands

```bash
# Run test week with debug info
python manager.py --league <LEAGUE_ID> --season 2024 --week 10 --user <USERNAME> --debug

# Refresh runtime features
python manager.py --league <LEAGUE_ID> --refresh

# View model details
python -m joblib models/QB_mean.joblib
```

---

## 🧠 Troubleshooting

**404 or Missing Weekly Data:**  
→ nfl_data_py may not have current week data yet. Script will fallback to baselines.

**“No positive upgrades found”**  
→ Your current lineup already beats waiver projections.

**Large File Push Errors (Git)**  
→ Exclude `.venv` and `/data` folders in `.gitignore`:
```
.venv*
data/
models/
__pycache__/
.env
```

---

## 🧾 License
MIT License © 2025 William Moore

---

## 🙌 Acknowledgments
- [Sleeper API](https://docs.sleeper.app)
- [nfl_data_py](https://github.com/cooperdff/nfl_data_py)
- [nflreadpy](https://github.com/cooperdff/nflreadpy)
- [scikit-learn](https://scikit-learn.org/)
- [PuLP](https://coin-or.github.io/pulp/)
