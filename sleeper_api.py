import requests
import time

BASE = "https://api.sleeper.app/v1"

class SleeperAPI:
    def __init__(self, rate_limit_sleep=0.25):
        self.base = BASE
        self.sleep = rate_limit_sleep

    def _get(self, path: str):
        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url)
        r.raise_for_status()
        time.sleep(self.sleep)
        return r.json()

    # ---- Users & League ----
    def get_user(self, username: str):
        return self._get(f"user/{username}")

    def get_user_by_id(self, user_id: str):
        return self._get(f"user/{user_id}")

    def get_league(self, league_id: str):
        return self._get(f"league/{league_id}")

    def get_league_users(self, league_id: str):
        return self._get(f"league/{league_id}/users")

    def get_rosters(self, league_id: str):
        return self._get(f"league/{league_id}/rosters")

    # ---- Players ----
    def get_players(self):
        # Sleeper global players map
        return self._get("players/nfl")

    # ---- Misc ----
    def get_state(self, league_id: str):
        return self._get(f"league/{league_id}/state")

    def get_traded_picks(self, league_id: str):
        return self._get(f"league/{league_id}/traded_picks")