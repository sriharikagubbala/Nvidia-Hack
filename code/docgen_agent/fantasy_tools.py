import httpx
from langchain_core.tools import tool

SLEEPER_BASE = "https://api.sleeper.app/v1"

# Lookup player ID from full name
async def get_player_id(player_name: str) -> str:
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{SLEEPER_BASE}/players/nfl")
        players = res.json()
        for pid, info in players.items():
            full_name = info.get("full_name", "").lower()
            if full_name == player_name.lower():
                return pid
    return None


@tool
async def fetch_player_stats(player_name: str) -> str:
    """
    Fetch latest stats for a given NFL player using Sleeper API.
    """
    player_id = await get_player_id(player_name)
    if not player_id:
        return f"Player {player_name} not found."

    async with httpx.AsyncClient() as client:
        stats_res = await client.get(f"{SLEEPER_BASE}/stats/nfl/regular/2023/17")  # last week's stats
        stats = stats_res.json().get(player_id, {})

    if not stats:
        return f"No stats found for {player_name}."

    points = stats.get("pts_ppr", "N/A")
    receptions = stats.get("rec", "N/A")
    yards = stats.get("yds", "N/A")
    return f"{player_name} in Week 17: {points} PPR pts, {receptions} rec, {yards} yards"


@tool
async def recommend_players(position: str, round: int) -> str:
    """
    Recommend top players by position for a given round using ADP.
    """
    position = position.upper()

    async with httpx.AsyncClient() as client:
        adp_res = await client.get(f"{SLEEPER_BASE}/players/adp")
        adp_data = adp_res.json()

    picks = []
    for player in adp_data:
        pos = player.get("position")
        adp = player.get("adp", 999)
        if pos == position and round - 0.5 <= adp <= round + 0.5:
            name = player.get("full_name")
            team = player.get("team")
            picks.append(f"{name} ({team}) - ADP {adp:.1f}")

    if not picks:
        return f"No players found for {position} in Round {round}."

    return f"Top {position}s for Round {round}:\n" + "\n".join(picks[:5])
