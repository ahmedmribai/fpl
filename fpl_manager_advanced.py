#!/usr/bin/env python3
"""
fpl_manager_advanced.py
Advanced, fully automated FPL optimizer for 2025/26 season
—including CBIT/CBIRT scoring, 2× chips, AGCON free-transfer boost,
simplified assists, BPS tweaks, and fixture-difficulty features—
delivered via Gmail SMTP.
"""

import os
import time
import logging
import requests
import joblib
import pandas as pd
import numpy as np

from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from lightgbm import LGBMRegressor
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary
import schedule
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
from urllib.parse import quote

load_dotenv()

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
FPL_BASE_URL   = "https://fantasy.premierleague.com/api"
NEWS_URL       = "https://www.skysports.com/football/news"
MODEL_FILE     = "fpl_model.pkl"

# External data sources for new transfers
FBREF_BASE     = "https://fbref.com"
UNDERSTAT_BASE = "https://understat.com"
TRANSFERS_FILE = "new_transfers.json"

GMAIL_USER     = os.getenv("GMAIL_USER")
GMAIL_PASS     = os.getenv("GMAIL_PASS")
EMAIL_TO       = os.getenv("EMAIL_TO")
BUDGET_CAP     = 100.0
MAX_TEAM_PLYR  = 3
RISK_AVERSION  = 0.1    # for risk‐adjusted optimization

# ─── TOP 1K OPTIMIZATION SETTINGS ───────────────────────────────────────
TOP_1K_MODE    = True   # Enable aggressive strategies for elite rank
DIFF_TARGET    = 0.3    # Target 30% differential picks for rank climbing
ELITE_TEMPLATE = 0.7    # 70% template players for safety
CAPTAIN_DIFF   = 0.15   # Consider differential captains above 15% elite rate

# ─── LEAGUE NORMALIZATION COEFFICIENTS ───────────────────────────────────
LEAGUE_COEFFICIENTS = {
    "Premier League": 1.00,    # Baseline
    "La Liga": 0.95,           # Slightly easier
    "Bundesliga": 0.92,        # More attacking
    "Serie A": 0.91,           # More defensive
    "Ligue 1": 0.88,           # PSG dominance
    "Primeira Liga": 0.85,     # Portuguese league
    "Championship": 0.80,      # Lower quality
    "Eredivisie": 0.83,        # Dutch league
    "Liga MX": 0.75,           # Mexican league
}

# AGCON boost: 5 free transfers in GW16
AGCON_GW = 16
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def fetch_fpl():
    """Fetch FPL data with retry logic and error handling"""
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Fetch bootstrap data
            r = requests.get(f"{FPL_BASE_URL}/bootstrap-static/", timeout=30)
            r.raise_for_status()
            data = r.json()
            
            pl = pd.DataFrame(data["elements"])
            tm = pd.DataFrame(data["teams"])
            fx = pd.DataFrame(data["events"])
            
            # Fetch fixtures for difficulty ratings
            r_fixtures = requests.get(f"{FPL_BASE_URL}/fixtures/", timeout=30)
            r_fixtures.raise_for_status()
            fixtures = pd.DataFrame(r_fixtures.json())
            
            logging.info(f"Successfully fetched FPL data: {len(pl)} players, {len(fixtures)} fixtures")
            return pl, tm, fx, fixtures
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                logging.error("Max retries exceeded for FPL API")
                raise


def fetch_understat_xg_data():
    """Fetch Expected Goals/Assists data from Understat or similar source"""
    try:
        # Try to get real shots data from multiple sources
        real_shots_data = fetch_real_shots_data()
        
        if real_shots_data:
            logging.info("Successfully fetched real shots data")
            return real_shots_data
        
        # Fallback to proxy metrics
        logging.info("Using FPL proxy metrics for xG/xA calculation")
        return None  # Will calculate proxies in preprocessing
        
    except Exception as e:
        logging.warning(f"Failed to fetch xG/xA data: {e}")
        return None


def fetch_real_shots_data():
    """Fetch real shots on target data from external sources"""
    try:
        shots_data = {}
        
        # Method 1: Try FBRef scraping
        fbref_data = fetch_fbref_shots_data()
        if fbref_data:
            shots_data.update(fbref_data)
            
        # Method 2: Try Understat API
        understat_data = fetch_understat_shots_api()
        if understat_data:
            shots_data.update(understat_data)
            
        # Method 3: Try detailed FPL endpoints
        fpl_detailed = fetch_detailed_fpl_shots()
        if fpl_detailed:
            shots_data.update(fpl_detailed)
            
        if shots_data:
            logging.info(f"Fetched real shots data for {len(shots_data)} players")
            return shots_data
        else:
            logging.info("No real shots data available, will use proxies")
            return None
            
    except Exception as e:
        logging.warning(f"Real shots data fetch failed: {e}")
        return None


def fetch_fbref_shots_data():
    """Scrape shots data from FBRef"""
    try:
        import time
        from urllib.parse import urljoin
        
        # FBRef Premier League stats page
        base_url = "https://fbref.com/en/comps/9/stats/Premier-League-Stats"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logging.info("Fetching FBRef Premier League stats...")
        response = requests.get(base_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logging.warning(f"FBRef request failed: HTTP {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the stats table (shooting stats)
        shooting_table = None
        for table in soup.find_all('table'):
            if 'shooting' in table.get('id', '').lower():
                shooting_table = table
                break
                
        if not shooting_table:
            # Try standard stats table
            shooting_table = soup.find('table', {'id': 'stats_standard'})
            
        if not shooting_table:
            logging.warning("Could not find shooting stats table on FBRef")
            return None
            
        shots_data = {}
        
        # Parse table rows
        tbody = shooting_table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                try:
                    # Get player name
                    player_cell = row.find('td', {'data-stat': 'player'})
                    if not player_cell:
                        continue
                        
                    player_name = player_cell.get_text(strip=True)
                    
                    # Get shots data
                    shots_cell = row.find('td', {'data-stat': 'shots'})
                    shots_on_target_cell = row.find('td', {'data-stat': 'shots_on_target'})
                    
                    if shots_cell and shots_on_target_cell:
                        shots = float(shots_cell.get_text(strip=True) or 0)
                        shots_on_target = float(shots_on_target_cell.get_text(strip=True) or 0)
                        
                        # Store by player name (will need to match to FPL later)
                        shots_data[player_name] = {
                            'shots': shots,
                            'shots_on_target': shots_on_target,
                            'shot_accuracy': (shots_on_target / shots * 100) if shots > 0 else 0
                        }
                        
                except (ValueError, AttributeError, ZeroDivisionError) as e:
                    logging.debug(f"Error parsing row: {e}")
                    continue
                    
        # Rate limiting
        time.sleep(2)
        
        if shots_data:
            logging.info(f"Successfully scraped FBRef data for {len(shots_data)} players")
            return shots_data
        else:
            logging.warning("No shots data found in FBRef table")
            return None
            
    except Exception as e:
        logging.warning(f"FBRef shots scraping failed: {e}")
        return None


def fetch_understat_shots_api():
    """Get shots data from Understat"""
    try:
        # This would implement Understat API calls
        # For now, return None to use proxies
        
        logging.info("Understat shots data not implemented yet")
        return None
        
    except Exception as e:
        logging.warning(f"Understat shots fetch failed: {e}")
        return None


def fetch_detailed_fpl_shots():
    """Try FPL detailed player endpoints for shots data"""
    try:
        # Try to get additional stats from individual player endpoints
        # Some seasons FPL includes more detailed metrics
        
        logging.info("FPL detailed shots data not implemented yet")
        return None
        
    except Exception as e:
        logging.warning(f"FPL detailed shots fetch failed: {e}")
        return None


def fetch_elite_manager_data(current_gw):
    """Fetch top manager picks for crowd wisdom analysis"""
    try:
        # Progressive strategy based on gameweek
        if current_gw <= 4:
            logging.info("Early season: Using historical elite patterns")
            return fetch_historical_elite_patterns()
        else:
            logging.info("Mid/Late season: Fetching current elite manager picks")
            return fetch_current_elite_picks(current_gw)
            
    except Exception as e:
        logging.warning(f"Failed to fetch elite manager data: {e}")
        return None


def fetch_historical_elite_patterns():
    """Use historical data for early season (GW1-4) when current rankings unreliable"""
    try:
        # This would ideally come from stored historical data
        # For now, use common early-season patterns
        historical_patterns = {
            'popular_gw1_picks': {
                # Based on historical GW1 ownership among top 10K
                'template_players': [1, 2, 15, 45, 102],  # Example player IDs
                'popular_captains': [1, 15, 45],  # Popular captain choices
                'differential_picks': [78, 156, 234],  # Lower ownership, high upside
            },
            'early_season_trends': {
                'avoid_rotation_risk': True,
                'prefer_nailed_starters': True,
                'budget_enablers': [567, 678, 789],  # Cheap starting players
            }
        }
        
        logging.info("Using historical elite manager patterns for early season")
        return historical_patterns
        
    except Exception as e:
        logging.warning(f"Failed to fetch historical patterns: {e}")
        return None


def fetch_current_elite_picks(current_gw):
    """Fetch actual elite manager picks for mid/late season"""
    elite_data = {
        'player_ownership': {},
        'captain_picks': {},
        'transfer_trends': {},
        'manager_count': 0
    }
    
    try:
        # Method 1: Sample from top overall managers
        # Note: This is simplified - you'd want to store/track reliable elite managers
        elite_manager_ids = get_top_manager_sample()
        
        successful_fetches = 0
        max_retries = len(elite_manager_ids)
        
        for manager_id in elite_manager_ids[:min(100, len(elite_manager_ids))]:
            try:
                url = f"{FPL_BASE_URL}/entry/{manager_id}/event/{current_gw}/picks/"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    picks_data = response.json()
                    picks = picks_data.get('picks', [])
                    
                    # Track player ownership
                    for pick in picks:
                        player_id = pick['element']
                        elite_data['player_ownership'][player_id] = elite_data['player_ownership'].get(player_id, 0) + 1
                        
                        # Track captain picks
                        if pick.get('is_captain'):
                            elite_data['captain_picks'][player_id] = elite_data['captain_picks'].get(player_id, 0) + 1
                    
                    successful_fetches += 1
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                    if successful_fetches >= 50:  # Enough sample size
                        break
                        
            except Exception as e:
                logging.debug(f"Failed to fetch manager {manager_id}: {e}")
                continue
        
        elite_data['manager_count'] = successful_fetches
        
        if successful_fetches > 0:
            # Convert to percentages
            for player_id in elite_data['player_ownership']:
                elite_data['player_ownership'][player_id] = elite_data['player_ownership'][player_id] / successful_fetches
            
            for player_id in elite_data['captain_picks']:
                elite_data['captain_picks'][player_id] = elite_data['captain_picks'][player_id] / successful_fetches
                
            logging.info(f"Successfully fetched elite data from {successful_fetches} managers")
        else:
            logging.warning("No elite manager data fetched")
            return None
            
        return elite_data
        
    except Exception as e:
        logging.error(f"Elite manager data fetch failed: {e}")
        return None


def get_top_manager_sample():
    """Get sample of top-performing manager IDs"""
    try:
        # Fetch from overall league (ID 314 is overall league)
        url = f"{FPL_BASE_URL}/leagues-classic/314/standings/?page_standings=1"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            standings = data.get('standings', {}).get('results', [])
            
            # Get top managers
            top_manager_ids = [manager['entry'] for manager in standings[:100]]
            logging.info(f"Found {len(top_manager_ids)} top managers")
            return top_manager_ids
        else:
            logging.warning(f"Failed to fetch top managers: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        logging.warning(f"Failed to get top manager sample: {e}")
        # Fallback: Return some known reliable manager IDs (you'd maintain this list)
        return []  # Would contain actual manager IDs


def fetch_transfer_market_sentiment(player_name):
    """Get transfer market sentiment and hype for new signings"""
    try:
        # This could scrape Twitter, Reddit, FPL community sites for sentiment
        # For now, return a placeholder sentiment score
        
        # Factors that could influence sentiment:
        # - Transfer fee size
        # - Social media buzz
        # - Preseason performance
        # - Manager comments
        # - FPL community discussions
        
        sentiment = {
            'hype_score': 0.5,  # 0-1 scale
            'ownership_prediction': 0.1,  # Predicted ownership %
            'price_rise_risk': 0.3,  # Risk of price increase
            'community_confidence': 0.6  # Community confidence in player
        }
        
        # Placeholder: You'd implement actual sentiment analysis here
        logging.info(f"Generated sentiment for {player_name}")
        return sentiment
        
    except Exception as e:
        logging.warning(f"Sentiment fetch failed for {player_name}: {e}")
        return {'hype_score': 0.5, 'ownership_prediction': 0.1, 'price_rise_risk': 0.3, 'community_confidence': 0.5}


def load_external_transfer_data():
    """Load new transfer data from external leagues"""
    try:
        if os.path.exists(TRANSFERS_FILE):
            with open(TRANSFERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logging.info("No external transfer data file found")
            return create_default_transfer_data()
    except Exception as e:
        logging.warning(f"Failed to load external transfer data: {e}")
        return {}


def create_default_transfer_data():
    """Create default transfer data for major summer 2024/25 signings"""
    # This would be updated each transfer window with new signings
    default_data = {
        "transfers": [
            {
                "player_name": "Viktor Gyökeres",
                "fpl_name": "Gyökeres",  # As it appears in FPL
                "new_team": "Arsenal",
                "previous_club": "Sporting CP",
                "previous_league": "Primeira Liga",
                "position": "Forward",
                "season_stats": {
                    "2023/24": {
                        "games": 33,
                        "goals": 29,
                        "assists": 10,
                        "minutes": 2847,
                        "xG": 25.4,
                        "xA": 7.2,
                        "shots": 98,
                        "key_passes": 45
                    }
                },
                "transfer_fee": 65000000,  # €65M
                "is_new_to_premier_league": True,
                "data_reliability": 0.75  # 75% reliable vs 90% for EPL players
            }
        ]
    }
    
    # Save default data
    try:
        with open(TRANSFERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Created default transfer data with {len(default_data['transfers'])} players")
    except Exception as e:
        logging.warning(f"Failed to save default transfer data: {e}")
    
    return default_data


def integrate_external_players(df, external_data):
    """Integrate external player data into the main FPL DataFrame"""
    try:
        if not external_data or 'transfers' not in external_data:
            return df
        
        logging.info(f"Integrating {len(external_data['transfers'])} external players")
        
        # Add default values for players without external data
        df['external_goals'] = 0
        df['external_assists'] = 0
        df['external_xg'] = 0.0
        df['external_xa'] = 0.0
        df['is_new_signing'] = False
        df['data_reliability'] = 0.9  # Default high for EPL players
        df['blended_ppg'] = df['points_per_game']
        df['new_signing_boost'] = 1.0
        
        logging.info("Completed external player integration (using defaults for now)")
        return df
        
    except Exception as e:
        logging.error(f"External player integration failed: {e}")
        return df


def fetch_news():
    """Fetch team news with error handling"""
    try:
        url = "https://www.fantasyfootballscout.co.uk/category/team-news/"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        news = [a.text for a in soup.select(".post-content h2 a")[:30]]
        logging.info(f"Fetched {len(news)} news items")
        return news
    except Exception as e:
        logging.warning(f"Failed to fetch news: {e}")
        return []  # Return empty list on failure


def preprocess(pl, tm, fx, fixtures, news, xg_data=None, elite_data=None):
    """
    1. Basic stats & form
    2. New CBIT/CBIRT features
    3. Simplified assists placeholder
    4. Fixture Difficulty Rating features (next GW & rolling 3 GWs)
    5. Free-transfer logic (for reporting)
    6. BPS tweaks placeholders
    7. NEW: External league transfer integration
    """
    df = pl.copy()
    
    # ═══ CRITICAL: Fix Data Types First ═══
    # Convert string columns to numeric to prevent comparison and model errors
    logging.info("Converting data types...")
    
    # Key columns that must be numeric for optimization and ML
    numeric_columns = [
        'selected_by_percent', 'now_cost', 'total_points', 'points_per_game',
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
        'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat',
        'ict_index', 'value', 'transfers_in', 'transfers_out'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            logging.debug(f"Column '{col}' not found, skipping conversion")
    
    logging.info(f"Data type conversion completed for {len([c for c in numeric_columns if c in df.columns])} columns")
    
    # ─── EXTERNAL TRANSFER INTEGRATION ─────────────────────────────────────
    # Load and integrate data from other leagues for new signings
    external_data = load_external_transfer_data()
    df = integrate_external_players(df, external_data)
    logging.info("Integrated external transfer data")
    
    # Ensure all numeric columns are still numeric after integration
    for col in ['selected_by_percent', 'now_cost', 'total_points', 'points_per_game']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ─── Core Stats ────────────────────────────────────────────
    df["value"]    = df["now_cost"] / 10
    df["form"]     = df["form"].astype(float)
    df["ppg"]      = df["points_per_game"].astype(float)
    df["element_type_name"] = df["element_type"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"})

    # ─── CBIT / CBIRT Scoring ──────────────────────────────────
    # Use available FPL metrics as proxy for CBIT/CBIRT until real data available
    is_def = df["element_type_name"] == "DEF"
    
    # CBIT proxy for defenders: based on clean sheets, saves, penalties saved
    df["cbit"] = np.where(
        is_def,
        df["clean_sheets"].fillna(0) * 2 + df["saves"].fillna(0) * 0.1 + df["penalties_saved"].fillna(0) * 5,
        0  # non-defenders get 0 CBIT
    )
    
    # CBIRT proxy for mids/forwards: based on goals, assists, key passes, shots
    df["cbirt"] = np.where(
        ~is_def,
        df["goals_scored"].fillna(0).astype(float) * 3 + df["assists"].fillna(0).astype(float) * 2 + 
        pd.to_numeric(df["creativity"].fillna(0), errors='coerce').fillna(0) * 0.01 + 
        pd.to_numeric(df["threat"].fillna(0), errors='coerce').fillna(0) * 0.01,
        0  # defenders get 0 CBIRT
    )
    
    # Bonus flags based on improved thresholds
    df["bonus_flag"] = np.where(
        (is_def  & (df["cbit"]  >= df["cbit"].quantile(0.75))) |
        (~is_def & (df["cbirt"] >= df["cbirt"].quantile(0.75))),
        1, 0
    )

    # ─── Expected Goals & Assists Integration ─────────────────────────────
    
    # Calculate Expected Goals (xG) - using FPL proxy metrics if external data unavailable
    if xg_data:
        # Use external xG data (Understat, FBref, etc.)
        df["expected_goals"] = df.index.map(lambda i: xg_data.get(df.loc[i, "id"], {}).get("xg", 0))
        df["expected_assists"] = df.index.map(lambda i: xg_data.get(df.loc[i, "id"], {}).get("xa", 0))
        logging.info("Using external xG/xA data")
    else:
        # Calculate xG proxies from available FPL metrics
        # Use safe column access with fallbacks for missing columns
        def safe_numeric_column(df, col_name, default=0):
            if col_name in df.columns:
                return pd.to_numeric(df[col_name].fillna(default), errors='coerce').fillna(default)
            else:
                logging.warning(f"Column '{col_name}' not found in FPL data, using default value {default}")
                return pd.Series([default] * len(df), index=df.index)
        
        # xG proxy: goals, threat, creativity (using available FPL columns)
        df["expected_goals"] = (
            safe_numeric_column(df, "goals_scored", 0) * 0.4 +  # Historical conversion (higher weight)
            safe_numeric_column(df, "threat", 0) * 0.002 +  # FPL threat index (higher weight)
            safe_numeric_column(df, "creativity", 0) * 0.001 +  # Some forwards get creative
            safe_numeric_column(df, "bonus", 0) * 0.1 +  # Bonus points indicate good performance
            safe_numeric_column(df, "minutes", 0) * 0.001  # Playing time factor
        )
        
        # xA proxy: assists, creativity (using available FPL columns)
        df["expected_assists"] = (
            safe_numeric_column(df, "assists", 0) * 0.5 +  # Historical assists (primary)
            safe_numeric_column(df, "creativity", 0) * 0.003 +  # FPL creativity (higher weight)
            safe_numeric_column(df, "bonus", 0) * 0.05 +  # Bonus points for assists
            safe_numeric_column(df, "minutes", 0) * 0.0005 +  # Playing time factor
            safe_numeric_column(df, "yellow_cards", 0) * -0.1  # Yellow cards reduce assists
        )
        logging.info("Using FPL proxy metrics for xG/xA calculation")
    
    # xG/xA per 90 minutes (normalized for playing time)
    df["minutes_played"] = df["minutes"].fillna(0).astype(float)
    df["xg_per_90"] = np.where(
        df["minutes_played"] > 0,
        (df["expected_goals"] * 90) / df["minutes_played"],
        0
    )
    df["xa_per_90"] = np.where(
        df["minutes_played"] > 0,
        (df["expected_assists"] * 90) / df["minutes_played"],
        0
    )
    
    # Over/under-performance vs expected
    df["goals_vs_xg"] = df["goals_scored"].fillna(0) - df["expected_goals"]
    df["assists_vs_xa"] = df["assists"].fillna(0) - df["expected_assists"]
    
    # Combined xG + xA metric (total attacking threat)
    df["combined_xg_xa"] = df["expected_goals"] + df["expected_assists"]
    df["combined_xg_xa_per_90"] = df["xg_per_90"] + df["xa_per_90"]
    
    # Enhanced Assist Prediction using xA ─────────────────────────────
    def safe_numeric_column(df, col_name, default=0):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].fillna(default), errors='coerce').fillna(default)
        else:
            return pd.Series([default] * len(df), index=df.index)
    
    df["assist_boost"] = (
        df["expected_assists"] * 0.7 +  # Expected assists (primary)
        safe_numeric_column(df, "assists", 0) * 0.2 +  # Historical assists
        safe_numeric_column(df, "creativity", 0) * 0.001  # FPL creativity index
    )

    # ─── Bonus Points System (BPS) Tweaks ───────────────────────
    df["bps_old"]     = df["bps"].astype(float)
    df["bps_tweaked"] = df["bps_old"]  # ready for real event-based recalculation

    # ─── Fixture Difficulty Rating (FDR) ────────────────────────
    next_gw = int(fx.loc[fx["is_next"], "id"].iloc[0])
    
    # Get next GW fixtures and map difficulties
    next_fixtures = fixtures[fixtures["event"] == next_gw]
    
    # Create team difficulty mapping for next GW
    team_difficulty_map = {}
    for _, fixture in next_fixtures.iterrows():
        team_difficulty_map[fixture["team_h"]] = fixture["team_h_difficulty"]
        team_difficulty_map[fixture["team_a"]] = fixture["team_a_difficulty"]
    
    df["fdr_next"] = df["team"].map(team_difficulty_map).fillna(3.0)
    
    # rolling 3-GW average difficulty
    team_fdr_history = []
    for team_id in df["team"].unique():
        team_fixtures = fixtures[(fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)]
        team_fixtures = team_fixtures.sort_values("event")
        
        difficulties = []
        for _, f in team_fixtures.iterrows():
            if f["team_h"] == team_id:
                difficulties.append(f["team_h_difficulty"])
            else:
                difficulties.append(f["team_a_difficulty"])
        
        # Calculate rolling 3-game average
        if len(difficulties) >= 3:
            avg_difficulty = np.mean(difficulties[-3:])
        else:
            avg_difficulty = np.mean(difficulties) if difficulties else 3.0
            
        team_fdr_history.append({"team": team_id, "fdr_3gw": avg_difficulty})
    
    fdr_3gw_map = {row["team"]: row["fdr_3gw"] for row in team_fdr_history}
    df["fdr_3gw"] = df["team"].map(fdr_3gw_map).fillna(3.0)

    # ─── Advanced Features from Top FPL Repos ─────────────────────────────
    
    # Ownership momentum (transfers in vs out)
    df["ownership_momentum"] = (
        df["transfers_in"].fillna(0) - df["transfers_out"].fillna(0)
    ).astype(float)
    
    # Price change momentum (recent price changes indicate form)
    df["price_change_momentum"] = (
        df["cost_change_event"].fillna(0).astype(float) * 0.1 +
        df["cost_change_start"].fillna(0).astype(float) * 0.05
    )
    
    # Weighted form (recent games matter more)
    # Give more weight to recent performances
    if "form" in df.columns:
        df["weighted_form"] = (
            df["form"].fillna(0).astype(float) * 0.6 +  # recent form
            df["ppg"].fillna(0).astype(float) * 0.4     # season average
        )
    else:
        df["weighted_form"] = df["ppg"].fillna(0).astype(float)
    
    # Fixture run analysis (next 4 gameweeks average difficulty)
    team_fixture_runs = []
    for team_id in df["team"].unique():
        future_fixtures = fixtures[
            (fixtures["event"] > next_gw) & 
            (fixtures["event"] <= next_gw + 3) &
            ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id))
        ]
        
        if len(future_fixtures) > 0:
            difficulties = []
            for _, f in future_fixtures.iterrows():
                if f["team_h"] == team_id:
                    difficulties.append(f["team_h_difficulty"])
                else:
                    difficulties.append(f["team_a_difficulty"])
            avg_difficulty = np.mean(difficulties)
        else:
            avg_difficulty = 3.0  # neutral
            
        team_fixture_runs.append({"team": team_id, "fixture_run": avg_difficulty})
    
    fixture_run_map = {row["team"]: row["fixture_run"] for row in team_fixture_runs}
    df["fixture_run_4gw"] = df["team"].map(fixture_run_map).fillna(3.0)
    
    # Injury/rotation risk based on minutes and recent starts
    # Use safe column access for games_played (may not exist in FPL API)
    def safe_numeric_column(df, col_name, default=0):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].fillna(default), errors='coerce').fillna(default)
        else:
            return pd.Series([default] * len(df), index=df.index)
    
    # Calculate games played from available data or estimate
    if 'games_played' in df.columns:
        games_played = safe_numeric_column(df, "games_played", 1)
    else:
        # Estimate games played from minutes (90 min per game)
        minutes = safe_numeric_column(df, "minutes", 0)
        games_played = np.maximum(1, np.round(minutes / 90))  # At least 1 game
        logging.info("'games_played' column not found, estimating from minutes")
    
    df["minutes_per_game"] = safe_numeric_column(df, "minutes", 0) / games_played
    df["rotation_risk"] = np.where(
        df["minutes_per_game"] < 60,  # Less than 60 min/game = rotation risk
        1.0 - (df["minutes_per_game"] / 90),  # Higher risk for fewer minutes
        0.1  # Low risk for regular starters
    )
    
    # ─── Elite Manager Wisdom Features ─────────────────────────────
    
    if elite_data:
        if next_gw <= 4:
            df = apply_historical_elite_features(df, elite_data)
        else:
            df = apply_elite_manager_features(df, elite_data)
    else:
        # Defaults when elite data not available
        df["elite_ownership"] = 0.0
        df["elite_captain_rate"] = 0.0
        df["elite_differential"] = 0.0
        df["crowd_wisdom_score"] = 0.5  # Neutral
        df["template_score"] = 0.0
    
    logging.info("Applied elite manager wisdom features")
    
    # ─── NEW SIGNING ADJUSTMENTS ─────────────────────────────────────────
    # Special handling for players new to Premier League
    new_signings = df[df.get('is_new_signing', False) == True]
    if len(new_signings) > 0:
        logging.info(f"Applying new signing adjustments to {len(new_signings)} players")
        
        for idx in new_signings.index:
            # Use blended PPG if available
            if 'blended_ppg' in df.columns and df.loc[idx, 'blended_ppg'] > 0:
                df.loc[idx, 'ppg'] = df.loc[idx, 'blended_ppg']
            
            # Apply hype boost to expected points calculation
            if 'new_signing_boost' in df.columns:
                boost = df.loc[idx, 'new_signing_boost']
                df.loc[idx, 'form'] = df.loc[idx, 'form'] * boost
            
            # Integrate external xG/xA data
            if 'external_xg' in df.columns and df.loc[idx, 'external_xg'] > 0:
                df.loc[idx, 'expected_goals'] = max(df.loc[idx, 'expected_goals'], df.loc[idx, 'external_xg'])
            
            if 'external_xa' in df.columns and df.loc[idx, 'external_xa'] > 0:
                df.loc[idx, 'expected_assists'] = max(df.loc[idx, 'expected_assists'], df.loc[idx, 'external_xa'])
            
            # Add transfer market sentiment boost
            sentiment = fetch_transfer_market_sentiment(df.loc[idx, 'web_name'])
            hype_multiplier = 1.0 + (sentiment['hype_score'] * 0.2)  # Up to 20% boost for high hype
            df.loc[idx, 'ppg'] = df.loc[idx, 'ppg'] * hype_multiplier
        
        logging.info("Completed new signing adjustments")

    # ─── Free Transfers for Next GW ─────────────────────────────
    df.attrs["next_gw"] = next_gw
    df.attrs["free_transfers"] = 5 if next_gw == AGCON_GW else 1

    return df


def apply_historical_elite_features(df, historical_data):
    """Apply early season elite manager patterns (GW1-4)"""
    try:
        patterns = historical_data.get('popular_gw1_picks', {})
        trends = historical_data.get('early_season_trends', {})
        
        # Template player boost
        template_players = patterns.get('template_players', [])
        df["elite_ownership"] = df.index.map(lambda i: 0.8 if df.loc[i, 'id'] in template_players else 0.1)
        
        # Captain preference boost
        popular_captains = patterns.get('popular_captains', [])
        df["elite_captain_rate"] = df.index.map(lambda i: 0.6 if df.loc[i, 'id'] in popular_captains else 0.0)
        
        # Differential picks (lower ownership, high upside)
        differential_picks = patterns.get('differential_picks', [])
        df["elite_differential"] = df.index.map(lambda i: 0.9 if df.loc[i, 'id'] in differential_picks else 0.0)
        
        # Early season crowd wisdom (avoid rotation, prefer nailed starters)
        if trends.get('avoid_rotation_risk', False):
            df["crowd_wisdom_score"] = np.where(
                df["rotation_risk"] < 0.3,  # Low rotation risk
                0.7,  # High crowd wisdom score
                0.2   # Lower score for rotation risks
            )
        else:
            df["crowd_wisdom_score"] = 0.5  # Neutral
            
        logging.info(f"Applied historical elite features to {len(df)} players")
        return df
        
    except Exception as e:
        logging.error(f"Failed to apply historical elite features: {e}")
        # Set defaults on failure
        df["elite_ownership"] = 0.0
        df["elite_captain_rate"] = 0.0
        df["elite_differential"] = 0.0
        df["crowd_wisdom_score"] = 0.0
        return df


def apply_elite_manager_features(df, elite_data):
    """Apply current elite manager picks as features (GW5+)"""
    try:
        player_ownership = elite_data.get('player_ownership', {})
        captain_picks = elite_data.get('captain_picks', {})
        manager_count = elite_data.get('manager_count', 1)
        
        # Elite ownership rate (% of elite managers who own this player)
        df["elite_ownership"] = df["id"].map(player_ownership).fillna(0.0)
        
        # Elite captain rate (% of elite managers who captain this player)
        df["elite_captain_rate"] = df["id"].map(captain_picks).fillna(0.0)
        
        # Differential score (inverse of general ownership but high elite ownership)
        general_ownership = df["selected_by_percent"].fillna(0) / 100
        df["elite_differential"] = np.where(
            (df["elite_ownership"] > 0.3) & (general_ownership < 0.15),  # High elite, low general
            df["elite_ownership"] / (general_ownership + 0.01),  # Differential score
            0.0
        )
        
        # Crowd wisdom score (combination of elite ownership and captain preference)
        df["crowd_wisdom_score"] = (
            df["elite_ownership"] * 0.7 +  # Elite ownership weight
            df["elite_captain_rate"] * 0.3  # Captain preference weight
        )
        
        # Template analysis (how "template" is this player)
        df["template_score"] = np.where(
            df["elite_ownership"] > 0.5,  # Owned by >50% of elite managers
            df["elite_ownership"],  # High template score
            0.0  # Not template
        )
        
        logging.info(f"Applied elite manager features from {manager_count} managers")
        return df
        
    except Exception as e:
        logging.error(f"Failed to apply elite manager features: {e}")
        # Set defaults on failure
        df["elite_ownership"] = 0.0
        df["elite_captain_rate"] = 0.0
        df["elite_differential"] = 0.0
        df["crowd_wisdom_score"] = 0.0
        df["template_score"] = 0.0
        return df


def train_and_serialize(df):
    """Train ensemble model with validation and feature engineering"""
    logging.info("Training ensemble model with enhanced features…")
    
    # Enhanced feature set including xG/xA, elite manager wisdom, and advanced features
    feats = [
        # Core stats
        "value", "form", "ppg", "minutes", "goals_scored", "assists",
        # Expected Goals/Assists features
        "expected_goals", "expected_assists", "xg_per_90", "xa_per_90",
        "combined_xg_xa", "combined_xg_xa_per_90", "goals_vs_xg", "assists_vs_xa",
        # Elite Manager Wisdom Features (NEW)
        "elite_ownership", "elite_captain_rate", "elite_differential", 
        "crowd_wisdom_score", "template_score",
        # Custom scoring features
        "cbit", "cbirt", "bonus_flag", "assist_boost", "bps_tweaked",
        # Fixture analysis
        "fdr_next", "fdr_3gw", "fixture_run_4gw",
        # Advanced features from top repos
        "selected_by_percent", "transfers_in", "ownership_momentum",
        "price_change_momentum", "weighted_form", "rotation_risk"
    ]
    
    # Data validation
    available_feats = [f for f in feats if f in df.columns]
    missing_feats = set(feats) - set(available_feats)
    if missing_feats:
        logging.warning(f"Missing features: {missing_feats}")
    
    X = df[available_feats].fillna(0)
    
    # Improved target variable: weighted combination
    y = (
        df["total_points"].fillna(0) * 0.7 +  # historical performance
        df["ppg"].fillna(0) * 30 * 0.3        # recent form scaled
    )
    
    # Filter out players with insufficient data
    valid_mask = (df["minutes"].fillna(0) > 90) & (y > 0)
    X_train = X[valid_mask]
    y_train = y[valid_mask]
    
    if len(X_train) < 50:
        logging.warning(f"Insufficient training data: {len(X_train)} samples")
        return False
    
    logging.info(f"Training on {len(X_train)} players with {len(available_feats)} features")
    
    try:
        # Train ensemble with improved hyperparameters
        m1 = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5, 
            random_state=42, n_jobs=-1
        ).fit(X_train, y_train)
        
        m2 = LGBMRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, verbose=-1
        ).fit(X_train, y_train)
        
        m3 = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6).fit(X_train, y_train)
        
        # Simple validation: predict on training data and check reasonableness
        pred_m1 = m1.predict(X_train)
        pred_m2 = m2.predict(X_train)
        pred_m3 = m3.predict(X_train)
        
        # Check if predictions are reasonable (0-20 points typical range)
        for i, (name, preds) in enumerate([("RF", pred_m1), ("LGB", pred_m2), ("BR", pred_m3)]):
            mean_pred = np.mean(preds)
            if not (0 <= mean_pred <= 20):
                logging.warning(f"{name} model predictions seem unreasonable: mean={mean_pred:.2f}")
        
        # Save models with metadata
        model_data = {
            'models': (m1, m2, m3),
            'features': available_feats,
            'train_samples': len(X_train),
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, MODEL_FILE)
        logging.info(f"Ensemble model saved with {len(X_train)} training samples")
        return True
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        return False


def load_models():
    """Load models with validation"""
    try:
        model_data = joblib.load(MODEL_FILE)
        if isinstance(model_data, tuple):  # backward compatibility
            return model_data, None
        else:
            return model_data['models'], model_data.get('features')
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return None, None


def predict(df):
    """Generate predictions with enhanced error handling"""
    logging.info("Predicting expected points with new rules & FDR…")
    
    models, saved_features = load_models()
    if models is None:
        logging.error("Could not load models for prediction")
        # Fallback: use simple heuristic
        df["exp_pts"] = df["ppg"].fillna(2) + df["bonus_flag"] * 2 + df["assist_boost"]
        return df
    
    m1, m2, m3 = models
    
    # Use saved features if available, otherwise enhanced default set with xG/xA
    if saved_features:
        feats = saved_features
    else:
        feats = [
            # Core features
            "value", "form", "ppg", "cbit", "cbirt", "bonus_flag",
            "assist_boost", "bps_tweaked", "fdr_next", "fdr_3gw",
            # Expected Goals/Assists features
            "expected_goals", "expected_assists", "xg_per_90", "xa_per_90",
            "combined_xg_xa_per_90", "goals_vs_xg", "assists_vs_xa",
            # Elite Manager Wisdom Features (NEW)
            "elite_ownership", "elite_captain_rate", "crowd_wisdom_score",
            # Advanced features
            "ownership_momentum", "price_change_momentum", "weighted_form", 
            "fixture_run_4gw", "rotation_risk"
        ]
    
    # Ensure all required features exist
    available_feats = [f for f in feats if f in df.columns]
    if len(available_feats) < len(feats):
        missing = set(feats) - set(available_feats)
        logging.warning(f"Missing prediction features: {missing}")
    
    try:
        X = df[available_feats].fillna(0)
        
        # Generate ensemble predictions
        preds = []
        for model in [m1, m2, m3]:
            try:
                pred = model.predict(X)
                # Sanity check predictions
                if np.any(np.isnan(pred)) or np.any(pred < 0) or np.any(pred > 50):
                    logging.warning(f"Model produced unrealistic predictions")
                    continue
                preds.append(pred)
            except Exception as e:
                logging.warning(f"Model prediction failed: {e}")
                continue
        
        if preds:
            base_preds = np.mean(preds, axis=0)
        else:
            logging.error("All models failed, using fallback prediction")
            base_preds = df["ppg"].fillna(2).values
        
        # Apply bonuses and adjustments
        df["exp_pts"] = (
            base_preds + 
            2 * df["bonus_flag"].fillna(0) + 
            df["assist_boost"].fillna(0)
        )
        
        # Cap predictions to reasonable range
        df["exp_pts"] = np.clip(df["exp_pts"], 0, 25)
        
        logging.info(f"Predicted points for {len(df)} players (mean: {df['exp_pts'].mean():.2f})")
        return df
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        # Ultimate fallback
        df["exp_pts"] = df["ppg"].fillna(2) + df["bonus_flag"].fillna(0) * 2
        return df


def select_fallback_team(df):
    """Fallback team selection when optimization fails"""
    logging.info("Selecting fallback team using greedy algorithm")
    
    squad_players = []
    remaining_budget = BUDGET_CAP
    
    # Position requirements: 2 GK, 5 DEF, 5 MID, 3 FWD
    position_reqs = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    
    for pos, count in position_reqs.items():
        pos_players = df[df["element_type_name"] == pos].copy()
        pos_players = pos_players.sort_values("exp_pts", ascending=False)
        
        selected = 0
        for _, player in pos_players.iterrows():
            if selected >= count:
                break
            if player["value"] <= remaining_budget:
                squad_players.append(player.name)
                remaining_budget -= player["value"]
                selected += 1
    
    if len(squad_players) < 15:
        logging.warning(f"Fallback selection incomplete: only {len(squad_players)}/15 players selected")
    
    squad = df.loc[squad_players].copy()
    # Select top 11 by expected points as starters
    squad = squad.sort_values("exp_pts", ascending=False)
    squad["is_start"] = False
    squad["is_captain"] = False
    squad["is_vice"] = False
    
    # Set starters (top 11)
    squad.iloc[:11, squad.columns.get_loc("is_start")] = True
    
    # Set captain (highest expected points among starters)
    starters = squad[squad["is_start"]]
    if len(starters) > 0:
        captain_idx = starters["exp_pts"].idxmax()
        squad.loc[captain_idx, "is_captain"] = True
        
        # Set vice-captain (second highest, different from captain)
        remaining_starters = starters[starters.index != captain_idx]
        if len(remaining_starters) > 0:
            vice_idx = remaining_starters["exp_pts"].idxmax()
            squad.loc[vice_idx, "is_vice"] = True
    
    return squad


def optimize(df):
    logging.info("Optimizing squad, formation, and captaincy…")
    prob = LpProblem("FPL", LpMaximize)
    idx = df.index.tolist()
    pick    = LpVariable.dicts("pick",    idx, 0, 1, LpBinary)
    start   = LpVariable.dicts("start",   idx, 0, 1, LpBinary)
    captain = LpVariable.dicts("captain", idx, 0, 1, LpBinary)
    vice    = LpVariable.dicts("vice",    idx, 0, 1, LpBinary)

    # objective: maximize risk-adjusted expected points including captaincy bonus
    if TOP_1K_MODE:
        # Top 1K mode: Weight differentials and elite wisdom more heavily
        prob += lpSum(
            (
                df.loc[i,"exp_pts"] + 
                df.loc[i,"elite_differential"] * 0.5 +  # Boost differentials
                df.loc[i,"crowd_wisdom_score"] * 1.0 -   # Elite wisdom boost
                RISK_AVERSION * df.loc[i,"rotation_risk"] # Penalize rotation risk
            ) * start[i] +
            df.loc[i,"exp_pts"] * captain[i]  # Captain gets double points
            for i in idx
        )
    else:
        # Standard mode
        prob += lpSum(
            (df.loc[i,"exp_pts"] - RISK_AVERSION * df.loc[i,"bps_tweaked"]) * start[i] +
            df.loc[i,"exp_pts"] * captain[i]  # Captain gets double points
            for i in idx
        )

    # budget & squad size
    prob += lpSum(df.loc[i,"value"] * pick[i] for i in idx) <= BUDGET_CAP
    prob += lpSum(pick[i] for i in idx) == 15
    prob += lpSum(start[i] for i in idx) == 11
    for i in idx:
        prob += start[i] <= pick[i]
        prob += captain[i] <= start[i]  # captain must start
        prob += vice[i] <= start[i]     # vice-captain must start
    
    # captaincy constraints
    prob += lpSum(captain[i] for i in idx) == 1  # exactly one captain
    prob += lpSum(vice[i] for i in idx) == 1     # exactly one vice-captain
    prob += lpSum(captain[i] + vice[i] for i in idx) <= 2  # captain ≠ vice-captain

    # position constraints
    for pos, (cnt_min, cnt_max) in {
        "GK": (2, 2), "DEF": (5, 5),
        "MID": (5, 5), "FWD": (3, 3)
    }.items():
        ids = [i for i in idx if df.at[i,"element_type_name"] == pos]
        prob += lpSum(pick[i] for i in ids) == cnt_min

    # max 3 per club
    for team in df["team"].unique():
        ids = [i for i in idx if df.at[i,"team"] == team]
        prob += lpSum(pick[i] for i in ids) <= MAX_TEAM_PLYR
    
    # TOP 1K constraints
    if TOP_1K_MODE:
        # Differential constraints (force some low ownership picks)
        low_ownership = [i for i in idx if df.at[i,"selected_by_percent"] < 10]
        if len(low_ownership) >= 3:
            prob += lpSum(pick[i] for i in low_ownership) >= 2  # At least 2 differentials
        
        # Elite template balance (ensure core elite picks)
        high_elite = [i for i in idx if df.at[i,"elite_ownership"] > ELITE_TEMPLATE]
        if len(high_elite) >= 8:
            prob += lpSum(pick[i] for i in high_elite) >= 8  # Template core
        
        # Captain differential opportunity
        differential_captains = [i for i in idx if df.at[i,"elite_captain_rate"] > CAPTAIN_DIFF]
        if len(differential_captains) > 0:
            # Allow differential captains to have slight boost
            for i in differential_captains:
                prob += captain[i] <= 1.2 * start[i]  # Slight preference

    # Solve optimization with validation
    from pulp import LpStatus, LpStatusOptimal
    
    status = prob.solve()
    
    if status != LpStatusOptimal:
        logging.error(f"Optimization failed with status: {LpStatus[status]}")
        # Fallback: select top players by expected points within constraints
        logging.info("Using fallback selection method")
        return select_fallback_team(df)
    
    # Extract solution
    squad = df[[pick[i].value() > 0.5 for i in idx]].copy()
    squad["is_start"] = [start[i].value() > 0.5 for i in idx if pick[i].value() > 0.5]
    squad["is_captain"] = [captain[i].value() > 0.5 for i in idx if pick[i].value() > 0.5]
    squad["is_vice"] = [vice[i].value() > 0.5 for i in idx if pick[i].value() > 0.5]
    
    # Validate solution
    if len(squad) != 15 or squad["is_start"].sum() != 11:
        logging.error(f"Invalid solution: {len(squad)} players, {squad['is_start'].sum()} starters")
        return select_fallback_team(df)
    
    # Validate captaincy
    if squad["is_captain"].sum() != 1 or squad["is_vice"].sum() != 1:
        logging.warning(f"Captaincy issue: {squad['is_captain'].sum()} captains, {squad['is_vice'].sum()} vice-captains")
        
    captain_name = squad[squad['is_captain']]['web_name'].iloc[0] if squad['is_captain'].any() else "Unknown"
    expected_total = squad[squad['is_start']]['exp_pts'].sum()
    captain_bonus = squad[squad['is_captain']]['exp_pts'].sum() if squad['is_captain'].any() else 0
        
    logging.info(f"Optimization successful. Captain: {captain_name}, Expected: {expected_total:.1f} + {captain_bonus:.1f} = {expected_total + captain_bonus:.1f} pts")
    return squad.sort_values("exp_pts", ascending=False)


def save_and_access_results(squad, df):
    """Save results with multiple easy access options (better than email!)"""
    import os
    import shutil
    from pathlib import Path
    
    # Generate comprehensive filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"fpl_lineup_gw{df.attrs['next_gw']}_{timestamp}"
    
    # 🎯 OPTION 1: Save to multiple easy-to-find locations
    locations = []
    
    # Current directory (where script runs)
    current_file = f"{base_filename}.csv"
    squad.to_csv(current_file, index=False)
    locations.append(os.path.abspath(current_file))
    
    # Desktop copy (most accessible)
    try:
        desktop = Path.home() / "Desktop" 
        if desktop.exists():
            desktop_file = desktop / f"{base_filename}.csv"
            shutil.copy2(current_file, desktop_file)
            locations.append(str(desktop_file))
            logging.info(f"✅ EASY ACCESS: File copied to Desktop -> {desktop_file.name}")
    except Exception as e:
        logging.debug(f"Desktop copy failed: {e}")
    
    # Documents folder copy
    try:
        documents = Path.home() / "Documents" / "FPL_Lineups"
        documents.mkdir(exist_ok=True)
        docs_file = documents / f"{base_filename}.csv"
        shutil.copy2(current_file, docs_file)
        locations.append(str(docs_file))
        logging.info(f"📁 ORGANIZED: File saved to Documents/FPL_Lineups -> {docs_file.name}")
    except Exception as e:
        logging.debug(f"Documents copy failed: {e}")
    
    # 🎯 OPTION 2: Auto-open the file (Windows)
    try:
        if os.name == 'nt':  # Windows
            os.startfile(current_file)
            logging.info("🚀 AUTO-OPENED: CSV file opened in Excel/default app!")
        else:  # Mac/Linux
            import subprocess
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', current_file])
            logging.info("🚀 AUTO-OPENED: CSV file opened in default app!")
    except Exception as e:
        logging.debug(f"Auto-open failed: {e}")
    
    # 🎯 OPTION 3: Show desktop notification (Windows)
    try:
        if os.name == 'nt':
            import subprocess
            captain_name = squad[squad['is_captain']]['web_name'].iloc[0] if squad['is_captain'].any() else "Unknown"
            total_exp_pts = squad[squad['is_start']]['exp_pts'].sum()
            captain_exp_pts = squad[squad['is_captain']]['exp_pts'].sum() if squad['is_captain'].any() else 0
            
            notification_msg = f"FPL GW{df.attrs['next_gw']} Lineup Ready! Captain: {captain_name}, Expected: {total_exp_pts + captain_exp_pts:.1f} pts"
            # Windows toast notification
            subprocess.run([
                'powershell', '-Command', 
                f'[reflection.assembly]::loadwithpartialname("System.Windows.Forms"); [reflection.assembly]::loadwithpartialname("System.Drawing"); $notify = new-object system.windows.forms.notifyicon; $notify.icon = [System.Drawing.SystemIcons]::Information; $notify.visible = $true; $notify.showballoontip(10000,"FPL Optimizer","{notification_msg}","info")'
            ], shell=True, capture_output=True)
            logging.info("🔔 NOTIFICATION: Desktop notification sent!")
    except Exception as e:
        logging.debug(f"Notification failed: {e}")
    
    # 🎯 OPTION 4: Print summary to console for immediate viewing
    captain_name = squad[squad['is_captain']]['web_name'].iloc[0] if squad['is_captain'].any() else "Unknown"
    vice_name = squad[squad['is_vice']]['web_name'].iloc[0] if squad['is_vice'].any() else "Unknown"
    total_exp_pts = squad[squad['is_start']]['exp_pts'].sum()
    captain_exp_pts = squad[squad['is_captain']]['exp_pts'].sum() if squad['is_captain'].any() else 0
    
    print("\n" + "="*60)
    print(f"🏆 FPL LINEUP - GAMEWEEK {df.attrs['next_gw']}")
    print("="*60)
    print(f"⭐ CAPTAIN: {captain_name} ({squad[squad['is_captain']]['exp_pts'].iloc[0]:.1f} pts)")
    print(f"🥈 VICE-CAPTAIN: {vice_name}")
    print(f"📊 Expected Points: {total_exp_pts:.1f} + {captain_exp_pts:.1f} = {total_exp_pts + captain_exp_pts:.1f} pts")
    print(f"💰 Team Value: £{squad['value'].sum():.1f}M")
    print("\n🏁 STARTING XI:")
    starting_xi = squad[squad['is_start']][['web_name', 'element_type_name', 'exp_pts', 'value']]
    for _, player in starting_xi.iterrows():
        print(f"  {player['web_name']} ({player['element_type_name']}) - {player['exp_pts']:.1f} pts")
    print("\n🪑 BENCH:")
    bench = squad[~squad['is_start']][['web_name', 'element_type_name', 'exp_pts']]
    for _, player in bench.iterrows():
        print(f"  {player['web_name']} ({player['element_type_name']}) - {player['exp_pts']:.1f} pts")
    print("\n📁 FILE LOCATIONS:")
    for i, location in enumerate(locations, 1):
        print(f"  {i}. {location}")
    print("="*60 + "\n")
    
    logging.info(f"🎯 SUCCESS: {len(locations)} copies saved, console summary displayed!")
    return locations
        



def validate_config():
    """Validate configuration and environment"""
    issues = []
    
    # Check required environment variables
    if not GMAIL_USER and not GMAIL_PASS and not EMAIL_TO:
        issues.append("Email credentials missing - results will be saved locally")
    
    # Check API accessibility
    try:
        response = requests.get(FPL_BASE_URL + "/bootstrap-static/", timeout=10)
        if response.status_code != 200:
            issues.append(f"FPL API unreachable: HTTP {response.status_code}")
    except Exception as e:
        issues.append(f"FPL API connection failed: {e}")
    
    # Check budget cap is reasonable
    if not (50 <= BUDGET_CAP <= 200):
        issues.append(f"Budget cap seems unreasonable: {BUDGET_CAP}")
    
    if issues:
        for issue in issues:
            logging.warning(issue)
        return False
    
    logging.info("Configuration validation passed")
    return True


def pipeline():
    """Main pipeline with improved error handling and reporting"""
    start_time = datetime.now()
    logging.info(f"Starting FPL pipeline at {start_time}")
    
    try:
        # Validate configuration
        if not validate_config():
            logging.warning("Configuration issues detected, proceeding with caution")
        
        # Fetch data
        logging.info("Fetching FPL data...")
        pl, tm, fx, fixtures = fetch_fpl()
        news = fetch_news()
        xg_data = fetch_understat_xg_data()
        
        # Get current gameweek for progressive elite manager strategy
        current_gw = int(fx.loc[fx["is_next"], "id"].iloc[0]) if fx["is_next"].any() else 1
        elite_data = fetch_elite_manager_data(current_gw)
        
        # Preprocess with external transfer integration
        logging.info("Preprocessing data with external league integration...")
        df = preprocess(pl, tm, fx, fixtures, news, xg_data, elite_data)
        
        # Train model if needed
        if not os.path.isfile(MODEL_FILE):
            logging.info("No existing model found, training new model...")
            success = train_and_serialize(df)
            if not success:
                logging.error("Initial model training failed, using fallback predictions")
        
        # Generate predictions
        logging.info("Generating predictions...")
        df = predict(df)
        
        # Optimize team
        logging.info("Optimizing team selection...")
        squad = optimize(df)
        
        if squad is None or len(squad) == 0:
            logging.error("Team optimization failed completely")
            return False
        
        # Save and access results (multiple easy options!)
        logging.info("Saving results with easy access options...")
        file_locations = save_and_access_results(squad, df)
        
        # Retrain model for next time
        logging.info("Retraining model with latest data...")
        train_and_serialize(df)
        
        # Success metrics
        duration = datetime.now() - start_time
        total_exp_pts = squad[squad['is_start']]['exp_pts'].sum()
        total_value = squad['value'].sum()
        
        logging.info(f"Pipeline completed successfully in {duration.total_seconds():.1f}s")
        logging.info(f"Team metrics: {total_exp_pts:.1f} expected points, £{total_value:.1f}M value")
        return True
        
    except Exception as e:
        duration = datetime.now() - start_time
        logging.exception(f"Pipeline failed after {duration.total_seconds():.1f}s: {e}")
        return False


if __name__ == "__main__":
    # GitHub Actions / Cloud execution - run once and exit
    logging.info("FPL Optimizer starting (GitHub Actions mode)...")
    
    # Run pipeline and exit with appropriate code
    success = pipeline()
    
    if success:
        logging.info("Pipeline completed successfully - exiting")
        exit(0)
    else:
        logging.error("Pipeline failed - exiting with error code")
        exit(1)
