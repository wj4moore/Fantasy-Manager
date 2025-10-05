# sleeper_api.py
import requests
import time

# ==========================================================
# Utils: Lightweight wrapper for the Sleeper Fantasy Football API
# Handles rate-limiting, user, league, roster, and player endpoints.
# ==========================================================

BASE = "https://api.sleeper.app/v1"


class SleeperAPI:
    """Simple, rate-limited interface to the Sleeper public API."""

    def __init__(self, rate_limit_sleep: float = 0.25):
        self.base = BASE
        self.sleep = rate_limit_sleep

    # ----------------------------------------------------------
    # Utils: Core GET helper with retry-friendly rate limiting
    # ----------------------------------------------------------
    def _get(self, path: str):
        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url)
        r.raise_for_status()
        time.sleep(self.sleep)
        return r.json()

    # ----------------------------------------------------------
    # Users & League
    # ----------------------------------------------------------
    def get_user(self, username: str):
        """Fetch user profile by username."""
        return self._get(f"user/{username}")

    def get_user_by_id(self, user_id: str):
        """Fetch user profile by Sleeper user ID."""
        return self._get(f"user/{user_id}")

    def get_league(self, league_id: str):
        """Fetch metadata for a specific league."""
        return self._get(f"league/{league_id}")

    def get_league_users(self, league_id: str):
        """Return all users in a league."""
        return self._get(f"league/{league_id}/users")

    def get_rosters(self, league_id: str):
        """Return all rosters in a league."""
        return self._get(f"league/{league_id}/rosters")

    # ----------------------------------------------------------
    # Players
    # ----------------------------------------------------------
    def get_players(self):
        """Fetch global player map (NFL)."""
        return self._get("players/nfl")

    # ----------------------------------------------------------
    # Miscellaneous Endpoints
    # ----------------------------------------------------------
    def get_state(self, league_id: str):
        """Fetch league state (e.g., current week, season)."""
        return self._get(f"league/{league_id}/state")

    def get_traded_picks(self, league_id: str):
        """Fetch traded picks for a league."""
        return self._get(f"league/{league_id}/traded_picks")
