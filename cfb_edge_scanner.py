"""
CFB Edge Scanner
================

Screens a week of college football games for gaps between an SP+ derived
projected spread and the posted market spread.

This is a SCREENING tool. It surfaces games worth a closer look. It is not a
betting signal, and the edges it reports are almost certainly known to the
market, since SP+ is public. Backtest before trusting anything here.

Setup
-----
    pip install streamlit requests pandas
    export CFBD_API_KEY="your key from collegefootballdata.com/key"
    streamlit run cfb_edge_scanner.py

Sign conventions (the part that's easy to get wrong)
----------------------------------------------------
CFBD reports `spread` from the HOME team's perspective, where a negative
number means the home team is favored. So a spread of -7 means the market
expects the home team to win by 7.

We convert everything to "projected home margin" (positive = home wins by
that much) so the model and the market are directly comparable.

    market_home_margin = -spread
    model_home_margin  = (home_sp_plus - away_sp_plus) + home_field_points
    edge               = model_home_margin - market_home_margin

Positive edge means the model likes the HOME side relative to the market.
Negative edge means it likes the AWAY side.
"""

import os
from typing import Optional, Sequence

import pandas as pd
import requests

API_BASE = "https://api.collegefootballdata.com"
DEFAULT_HFA = 2.5
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Pure logic (no network, no UI) -- this is the part worth unit testing
# ---------------------------------------------------------------------------

def market_home_margin(spread: Optional[float]) -> Optional[float]:
    """Convert a CFBD home-perspective spread into an expected home margin.

    A spread of -7 (home favored by 7) becomes a home margin of +7.
    """
    if spread is None:
        return None
    return -float(spread)


def model_home_margin(
    home_rating: Optional[float],
    away_rating: Optional[float],
    home_field_points: float = DEFAULT_HFA,
    neutral_site: bool = False,
) -> Optional[float]:
    """Projected home margin from two SP+ style ratings.

    Ratings are points-above-average, so their difference is already a margin
    on a neutral field. Home field is added only when the game is not neutral.
    """
    if home_rating is None or away_rating is None:
        return None
    margin = float(home_rating) - float(away_rating)
    if not neutral_site:
        margin += float(home_field_points)
    return margin


def compute_edge(
    model_margin: Optional[float],
    mkt_margin: Optional[float],
) -> Optional[float]:
    """Model margin minus market margin. Positive favors the home side."""
    if model_margin is None or mkt_margin is None:
        return None
    return model_margin - mkt_margin


def describe_pick(edge: Optional[float], home: str, away: str) -> str:
    """Human readable side the edge points toward."""
    if edge is None:
        return ""
    if edge > 0:
        return f"{home} (home)"
    if edge < 0:
        return f"{away} (away)"
    return "no lean"


def consensus_spread(
    lines: Sequence[dict],
    provider: Optional[str] = None,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Pull a spread and total out of a CFBD game's `lines` array.

    If `provider` is given, use only that book. Otherwise average across every
    book that reported a spread. Returns (spread, total, provider_label).
    """
    if not lines:
        return None, None, None

    if provider and provider != "Consensus":
        rows = [ln for ln in lines if (ln.get("provider") or "") == provider]
        label = provider
    else:
        rows = list(lines)
        label = "Consensus"

    spreads = [
        float(ln["spread"]) for ln in rows
        if ln.get("spread") is not None
    ]
    totals = [
        float(ln["overUnder"]) for ln in rows
        if ln.get("overUnder") is not None
    ]

    spread = sum(spreads) / len(spreads) if spreads else None
    total = sum(totals) / len(totals) if totals else None
    return spread, total, (label if spreads else None)


def build_board(
    games: Sequence[dict],
    ratings: dict,
    home_field_points: float = DEFAULT_HFA,
    provider: Optional[str] = None,
) -> pd.DataFrame:
    """Join lines with ratings and produce the scanner board."""
    rows = []
    for g in games:
        home = g.get("homeTeam")
        away = g.get("awayTeam")
        if not home or not away:
            continue

        spread, total, book = consensus_spread(g.get("lines") or [], provider)
        mkt = market_home_margin(spread)

        home_rating = ratings.get(home)
        away_rating = ratings.get(away)
        model = model_home_margin(
            home_rating,
            away_rating,
            home_field_points,
            neutral_site=bool(g.get("neutralSite")),
        )

        edge = compute_edge(model, mkt)

        rows.append({
            "Week": g.get("week"),
            "Away": away,
            "Home": home,
            "Neutral": bool(g.get("neutralSite")),
            "Book": book,
            "Mkt Spread": spread,
            "Mkt Home Margin": mkt,
            "Model Home Margin": round(model, 2) if model is not None else None,
            "Edge": round(edge, 2) if edge is not None else None,
            "Abs Edge": round(abs(edge), 2) if edge is not None else None,
            "Lean": describe_pick(edge, home, away),
            "Total": total,
            "Home SP+": home_rating,
            "Away SP+": away_rating,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Abs Edge", ascending=False, na_position="last")
    return df


# ---------------------------------------------------------------------------
# Network layer
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def fetch_lines(api_key: str, year: int, week: int, season_type: str) -> list:
    resp = requests.get(
        f"{API_BASE}/lines",
        params={"year": year, "week": week, "seasonType": season_type},
        headers=_headers(api_key),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_sp_ratings(api_key: str, year: int) -> dict:
    """Return {team_name: overall SP+ rating}."""
    resp = requests.get(
        f"{API_BASE}/ratings/sp",
        params={"year": year},
        headers=_headers(api_key),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    out = {}
    for row in resp.json():
        team = row.get("team")
        rating = row.get("rating")
        if team and rating is not None:
            out[team] = float(rating)
    return out


def list_providers(games: Sequence[dict]) -> list:
    seen = set()
    for g in games:
        for ln in (g.get("lines") or []):
            p = ln.get("provider")
            if p:
                seen.add(p)
    return ["Consensus"] + sorted(seen)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def resolve_api_key(st) -> str:
    """Find the key without ever putting it in the source.

    Order: Streamlit secrets (for hosted deploys), then an environment
    variable (for local runs), then whatever the user types in the sidebar.
    """
    try:
        key = st.secrets.get("CFBD_API_KEY", "")
        if key:
            return str(key)
    except Exception:
        # No secrets.toml configured -- normal when running locally.
        pass
    return os.environ.get("CFBD_API_KEY", "")


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="CFB Edge Scanner",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CFB Edge Scanner")
    st.caption(
        "SP+ projected spread vs the posted market number. A screen for "
        "finding games worth a closer look, not a betting signal."
    )

    with st.sidebar:
        st.header("Settings")

        stored_key = resolve_api_key(st)
        if stored_key:
            st.success("API key loaded from secrets", icon="🔑")
            api_key = stored_key
        else:
            api_key = st.text_input(
                "CFBD API key",
                value="",
                type="password",
                help="Get one free at collegefootballdata.com/key",
            )

        compact = st.toggle(
            "Compact view",
            value=False,
            help="Fewer columns. Turn this on when using a phone.",
        )

        year = st.number_input("Season", min_value=2015, max_value=2100,
                               value=2026, step=1)
        week = st.number_input("Week", min_value=1, max_value=20,
                               value=2, step=1)
        season_type = st.selectbox("Season type",
                                   ["regular", "postseason"], index=0)

        st.divider()

        hfa = st.slider(
            "Home field advantage (points)",
            min_value=0.0, max_value=5.0, value=DEFAULT_HFA, step=0.25,
            help="Applied only to non-neutral games. Long run CFB home field "
                 "sits around 2 to 3 points.",
        )
        threshold = st.slider(
            "Minimum edge to flag (points)",
            min_value=0.0, max_value=15.0, value=3.0, step=0.5,
        )

        run = st.button("Scan week", type="primary", use_container_width=True)

    if not run:
        st.info("Enter your API key and press **Scan week** in the sidebar.")
        with st.expander("How the edge is calculated"):
            st.markdown(
                "- CFBD reports `spread` from the home team's perspective, "
                "so `-7` means the home team is favored by 7.\n"
                "- Everything is converted to **projected home margin** "
                "(positive = home wins by that much).\n"
                "- `model = (home SP+ − away SP+) + home field`\n"
                "- `edge = model − market`\n"
                "- Positive edge leans home, negative leans away."
            )
        return

    if not api_key:
        st.error("An API key is required.")
        return

    try:
        with st.spinner("Pulling lines and ratings..."):
            games = fetch_lines(api_key, int(year), int(week), season_type)
            ratings = fetch_sp_ratings(api_key, int(year))
    except requests.HTTPError as exc:
        st.error(f"API request failed: {exc}")
        st.caption(
            "If this is a 401, check the key. If the shape of the response "
            "looks wrong, CFBD may have revised the endpoint. Compare against "
            "the current API reference."
        )
        return
    except requests.RequestException as exc:
        st.error(f"Network error: {exc}")
        return

    if not games:
        st.warning("No games returned for that week.")
        return

    if not ratings:
        st.warning(
            "No SP+ ratings returned. Early in the season these may not be "
            "published yet, and SP+ covers FBS only."
        )
        return

    provider = st.selectbox("Book", list_providers(games), index=0)

    board = build_board(games, ratings, hfa, provider)

    if board.empty:
        st.warning("Nothing to show.")
        return

    priced = board[board["Edge"].notna()]
    flagged = priced[priced["Abs Edge"] >= threshold]

    c1, c2, c3 = st.columns(3)
    c1.metric("Games returned", len(board))
    c2.metric("With a usable line", len(priced))
    c3.metric(f"Edge ≥ {threshold}", len(flagged))

    COMPACT_COLS = ["Away", "Home", "Mkt Spread", "Edge", "Lean"]

    def show(frame):
        cols = COMPACT_COLS if compact else list(frame.columns)
        st.dataframe(frame[cols], use_container_width=True, hide_index=True)

    st.subheader("Flagged games")
    if flagged.empty:
        st.info("Nothing clears the threshold. Try lowering it.")
    else:
        show(flagged)

    with st.expander("Full board"):
        show(board)

    unpriced = board[board["Edge"].isna()]
    if not unpriced.empty:
        with st.expander(f"No line or no rating ({len(unpriced)})"):
            st.dataframe(unpriced[["Away", "Home", "Mkt Spread",
                                   "Home SP+", "Away SP+"]],
                         use_container_width=True, hide_index=True)

    st.download_button(
        "Download board as CSV",
        board.to_csv(index=False).encode("utf-8"),
        file_name=f"cfb_edges_{year}_wk{week}.csv",
        mime="text/csv",
    )

    st.caption(
        "SP+ is public, so most of what shows up here is already in the "
        "price. Treat a flag as a prompt to investigate, and backtest any "
        "rule against historical closing numbers before acting on it."
    )


if __name__ == "__main__":
    main()
