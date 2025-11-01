"""
FTC DECODE Scouting System V8.0 - Complete Server Edition (clean, API-field names)
- Restores full UI
- Uses FTC API scoring field names where available (autoPoints, dcPoints, endgamePoints, penaltyPointsCommitted, totalPoints)
- Removes 'Lift up over base' / lift_over_base completely
"""
import json
import os
import numpy as np
import requests
import base64
from flask import Flask, render_template_string, jsonify, request

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.secret_key = os.urandom(24)

# FTC API Configuration (can override via env vars)
FTC_API_BASE = 'https://ftc-api.firstinspires.org/v2.0'
FTC_API_USERNAME = os.environ.get('FTC_API_USERNAME', 'harry')
FTC_API_KEY = os.environ.get('FTC_API_KEY', '4B5F8571-EB4C-4F87-9DC1-0F3C5AFAF010')

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    config_path = 'config.json'
    default_config = {
        "scoring_weights": {
            "auto_weight": 0.3,
            "teleop_weight": 0.5,
            "endgame_weight": 0.2,
            "hanging_success_weight": 0.2,
            "reliability_multiplier": 3,
            "cycle_time_factor": 50
        },
        "thresholds": {
            "strong_auto": 20,
            "fast_cycle": 12,
            "high_hanging": 80,
            "high_reliability": 4,
            "high_scoring": 100,
            "consistency_high": 70,
            "consistency_medium": 40
        },
        "alliance": {
            "min_combined_score": 120,
            "consistency_threshold": 70
        }
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            return default_config
    else:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config

CONFIG = load_config()

# In-memory event store
EVENT_STORE = {
    'event_code': None,
    'season': 2025,
    'team_number': None,
    'raw_matches': [],
    'processed_entries': {},
    'teams_processed': {}
}

# ============================================================================
# UTILS
# ============================================================================

def safe_int(v, d=0):
    try:
        if isinstance(v, str):
            v = v.strip()
            if v.lower() in ['yes', 'true', 'complete']:
                return 1
            if v.lower() in ['no', 'false', 'failed']:
                return 0
        return int(float(v))
    except:
        return d

def safe_float(v, d=0.0):
    try:
        return float(v)
    except:
        return d

def get_ftc_api_headers():
    if not FTC_API_USERNAME or not FTC_API_KEY:
        return None
    auth_string = f"{FTC_API_USERNAME}:{FTC_API_KEY}"
    b64 = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
    return {'Authorization': f'Basic {b64}', 'Accept': 'application/json'}

# ============================================================================
# FETCH MATCHES
# ============================================================================
def fetch_event_matches(event_code, season=2025):
    """Fetch matches from the FTC API. Returns list or None."""
    headers = get_ftc_api_headers()
    try:
        url = f"{FTC_API_BASE}/{season}/matches/{event_code}"
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get('matches', [])
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return None

# ============================================================================
# PROCESS MATCH DATA (uses API field names where possible)
# ============================================================================
def process_ftc_match_data(matches):
    """
    Convert FTC API match data into per-team entry records.
    Each entry includes both 'friendly' fields and the API's field names:
      - autoPoints, dcPoints, endgamePoints, penaltyPointsCommitted, totalPoints
    'Lift up over base' removed entirely.
    """
    teams = {}
    for match_idx, match in enumerate(matches):
        match_num = match.get('matchNumber', match_idx + 1)
        actual_start = match.get('actualStartTime')
        post_result_time = match.get('postResultTime')
        if not actual_start and not post_result_time:
            # skip unplayed
            continue

        # Priority 1: match-level teams array (modern API)
        if 'teams' in match and isinstance(match['teams'], list):
            # try to get common score fields at match level (fallback to various names)
            score_red = match.get('scoreRedFinal') or match.get('scoreRed') or 0
            score_blue = match.get('scoreBlueFinal') or match.get('scoreBlue') or 0
            auto_red = match.get('scoreRedAuto') or 0
            auto_blue = match.get('scoreBlueAuto') or 0
            foul_red = match.get('scoreRedFoul') or 0
            foul_blue = match.get('scoreBlueFoul') or 0

            for team_entry in match['teams']:
                if not isinstance(team_entry, dict):
                    continue
                team_num = str(team_entry.get('teamNumber', '')).strip()
                station = team_entry.get('station', '') or ''
                if not team_num or team_num == '0':
                    continue

                # determine alliance by station string
                if 'Red' in station or 'red' in station.lower():
                    total_points = score_red or 0
                    auto_points = auto_red or 0
                    penalty_points = foul_red or 0
                else:
                    total_points = score_blue or 0
                    auto_points = auto_blue or 0
                    penalty_points = foul_blue or 0

                remaining = total_points - auto_points
                teleop_points = int(remaining * 0.70)
                endgame_points = remaining - teleop_points

                # Build entry with API-style field names plus friendly keys
                entry = {
                    # API-style fields
                    'autoPoints': auto_points,
                    'dcPoints': teleop_points,
                    'endgamePoints': endgame_points,
                    'penaltyPointsCommitted': penalty_points,
                    'totalPoints': total_points,

                    # Friendly / backward-compatible keys used elsewhere in code
                    'Auto Score': auto_points,
                    'Teleop Score': teleop_points,
                    'Endgame Score': endgame_points,
                    'Penalties': penalty_points // 2,
                    'Total Score': total_points,

                    'Team Number': team_num,
                    'Match Number': match_num,
                    'Cycle Time': 15,
                    'Hanging Success': 75 if endgame_points > 8 else 0,
                    'Reliability': 4 if auto_points > 15 else 3
                }

                teams.setdefault(team_num, []).append(entry)
            continue

        # Priority 2: alliance-based structure (older variation)
        for color in ['red', 'blue']:
            alliance = match.get(f'{color}Alliance') or match.get(color)
            if not alliance or not isinstance(alliance, dict):
                continue

            # Extract teams
            team_list = []
            if 'teams' in alliance and isinstance(alliance['teams'], list):
                for t in alliance['teams']:
                    if isinstance(t, dict):
                        tn = t.get('teamNumber')
                        if tn:
                            team_list.append(str(tn))
                    elif t:
                        team_list.append(str(t))
            else:
                for i in range(1, 4):
                    tn = alliance.get(f'team{i}')
                    if tn and tn != 0:
                        team_list.append(str(tn))

            if not team_list:
                continue

            # Prefer API field names where present
            total_points = alliance.get('totalPoints') or alliance.get('score') or 0
            auto_total = alliance.get('autoPoints', 0) or 0
            dc_points = alliance.get('dcPoints', 0) or 0
            endgame_points = alliance.get('endgamePoints', 0) or 0
            penalties = alliance.get('penaltyPointsCommitted', 0) or 0

            # If detailed breakdown missing, estimate
            if auto_total == 0 and dc_points == 0 and endgame_points == 0 and total_points > 0:
                auto_total = int(total_points * 0.25)
                dc_points = int(total_points * 0.55)
                endgame_points = int(total_points * 0.20)

            for tn in team_list:
                entry = {
                    # API-style
                    'autoPoints': auto_total,
                    'dcPoints': dc_points,
                    'endgamePoints': endgame_points,
                    'penaltyPointsCommitted': penalties // max(1, len(team_list)),
                    'totalPoints': total_points,

                    # Friendly
                    'Auto Score': auto_total,
                    'Teleop Score': dc_points,
                    'Endgame Score': endgame_points,
                    'Penalties': penalties // max(1, len(team_list)),
                    'Total Score': total_points,

                    'Team Number': str(tn),
                    'Match Number': match_num,
                    'Cycle Time': 15,
                    'Hanging Success': 75 if endgame_points > 8 else 0,
                    'Reliability': 4 if auto_total > 15 else 3
                }
                teams.setdefault(str(tn), []).append(entry)
    return teams

# ============================================================================
# AGGREGATION & ANALYSIS (no 'lift' fields)
# ============================================================================
def analyze_patterns(entries):
    """Analyze entries list and return pattern metrics."""
    if not entries:
        return {
            'consistency': 0, 'trend': '→', 'peak': 0,
            'strength_areas': [], 'weakness_areas': [],
            'behavioral': 'Unknown', 'reliability': 'Unknown'
        }

    scores = [safe_int(e.get('Total Score', 0)) for e in entries]
    cycles = [safe_float(e.get('Cycle Time', 20)) for e in entries]
    hanging = [safe_int(e.get('Hanging Success', 0)) for e in entries]
    reliabilities = [safe_int(e.get('Reliability', 3)) for e in entries]

    if len(scores) > 1:
        variance = np.var(scores)
        consistency = max(0, 100 - (variance * 10))
    else:
        consistency = 50

    if len(scores) >= 3:
        early_avg = np.mean(scores[:len(scores)//2])
        late_avg = np.mean(scores[len(scores)//2:])
        trend = '↗' if late_avg > early_avg + 5 else '↘' if late_avg < early_avg - 5 else '→'
    else:
        trend = '→'

    peak = max(scores) if scores else 0
    strength_areas = []
    weakness_areas = []

    thresholds = CONFIG['thresholds']
    avg_cycle = np.mean(cycles) if cycles else 20
    if avg_cycle <= thresholds['fast_cycle']:
        strength_areas.append('Speed')
    else:
        weakness_areas.append('Speed')

    avg_hanging = np.mean(hanging) if hanging else 0
    if avg_hanging >= thresholds['high_hanging']:
        strength_areas.append('Climbing')
    else:
        weakness_areas.append('Climbing')

    avg_reliability = np.mean(reliabilities) if reliabilities else 3
    if avg_reliability >= thresholds['high_reliability']:
        strength_areas.append('Reliability')
    else:
        weakness_areas.append('Reliability')

    avg_score = np.mean(scores) if scores else 0
    if avg_score >= thresholds['high_scoring']:
        strength_areas.append('Scoring')
    else:
        weakness_areas.append('Scoring')

    penalties = sum(1 for e in entries if safe_int(e.get('Penalties', 0)) > 0)
    hanging_count = sum(1 for e in entries if safe_int(e.get('Hanging Success', 0)) > 0)

    if penalties > len(entries) * 0.3:
        behavioral = 'Aggressive'
    elif hanging_count > len(entries) * 0.5:
        behavioral = 'Climbing-Focused'
    elif consistency >= 75:
        behavioral = 'Consistent'
    else:
        behavioral = 'Variable'

    reliability = '🟢 Consistent' if consistency >= thresholds['consistency_high'] else '🟡 Moderate' if consistency >= thresholds['consistency_medium'] else '🔴 Variable'

    return {
        'consistency': round(consistency, 1),
        'trend': trend,
        'peak': peak,
        'strength_areas': strength_areas,
        'weakness_areas': weakness_areas,
        'behavioral': behavioral,
        'reliability': reliability
    }

def process_teams(raw_entries_map):
    """
    Accepts dict team->entries or list(entries). Returns aggregated stats per team.
    """
    teams = {}
    if isinstance(raw_entries_map, dict):
        items = raw_entries_map.items()
    else:
        grouped = {}
        for e in raw_entries_map:
            tn = str(e.get('Team Number', 'Unknown')).strip()
            if not tn or tn == 'Unknown':
                continue
            grouped.setdefault(tn, []).append(e)
        items = grouped.items()

    for tn, entries in items:
        if not entries:
            continue
        n = len(entries)
        auto_avg = sum(safe_int(e.get('Auto Score', 0)) for e in entries) / n
        teleop_avg = sum(safe_int(e.get('Teleop Score', 0)) for e in entries) / n
        endgame_avg = sum(safe_int(e.get('Endgame Score', 0)) for e in entries) / n
        total_avg = sum(safe_int(e.get('Total Score', 0)) for e in entries) / n
        cycle_avg = sum(safe_float(e.get('Cycle Time', 20)) for e in entries) / n
        hanging_count = sum(1 for e in entries if safe_int(e.get('Hanging Success', 0)) > 0)
        hanging_pct = (hanging_count / n) * 100
        reliability = sum(safe_int(e.get('Reliability', 3)) for e in entries) / n
        penalties = sum(safe_int(e.get('Penalties', 0)) for e in entries)

        teams[str(tn)] = {
            'team_number': str(tn),
            'entries': entries,
            'auto_score_avg': auto_avg,
            'teleop_score_avg': teleop_avg,
            'endgame_score_avg': endgame_avg,
            'total_score_avg': total_avg,
            'cycle_time': cycle_avg,
            'hanging_success': hanging_pct,
            'reliability': reliability,
            'penalties': penalties,
            'pattern_analysis': analyze_patterns(entries)
        }
    return teams

# ============================================================================
# ALLIANCE / STRATEGY / RANKING
# ============================================================================
def get_strengths(team):
    s = []
    thresholds = CONFIG['thresholds']
    if team.get('auto_score_avg', 0) >= thresholds['strong_auto']:
        s.append('Strong Auto')
    if team.get('cycle_time', 30) <= thresholds['fast_cycle']:
        s.append('Fast Cycles')
    if team.get('hanging_success', 0) >= thresholds['high_hanging']:
        s.append('Climbing')
    if team.get('reliability', 0) >= thresholds['high_reliability']:
        s.append('Reliable')
    if team.get('total_score_avg', 0) >= thresholds['high_scoring']:
        s.append('High Scoring')
    return s if s else ['Developing']

def calc_alliance_score(t1, t2):
    if not t1 or not t2:
        return 0.0
    t1_total = t1.get('total_score_avg', 0)
    t2_total = t2.get('total_score_avg', 0)
    combined_avg = (t1_total + t2_total) / 2
    scoring_component = (combined_avg / 150) * 40

    t1_auto = t1.get('auto_score_avg', 0)
    t2_auto = t2.get('auto_score_avg', 0)
    t1_teleop = t1.get('teleop_score_avg', 0)
    t2_teleop = t2.get('teleop_score_avg', 0)
    t1_endgame = t1.get('endgame_score_avg', 0)
    t2_endgame = t2.get('endgame_score_avg', 0)

    auto_synergy = min((t1_auto + t2_auto) / 40, 1) * 8
    teleop_synergy = min((t1_teleop + t2_teleop) / 80, 1) * 12
    endgame_synergy = min((t1_endgame + t2_endgame) / 30, 1) * 5
    phase_component = auto_synergy + teleop_synergy + endgame_synergy

    t1_consistency = t1.get('pattern_analysis', {}).get('consistency', 50)
    t2_consistency = t2.get('pattern_analysis', {}).get('consistency', 50)
    avg_consistency = (t1_consistency + t2_consistency) / 2
    consistency_component = (avg_consistency / 100) * 20

    t1_reliability = t1.get('reliability', 3) / 5
    t2_reliability = t2.get('reliability', 3) / 5
    avg_reliability = (t1_reliability + t2_reliability) / 2

    avg_cycle = (t1.get('cycle_time', 20) + t2.get('cycle_time', 20)) / 2
    cycle_efficiency = max(0, (20 - avg_cycle) / 15)

    reliability_component = (avg_reliability * 0.6 + cycle_efficiency * 0.4) * 15

    bonus = 0
    if (t1_auto >= 20 and t2_teleop >= 40) or (t2_auto >= 20 and t1_teleop >= 40):
        bonus += 3
    if t1.get('hanging_success', 0) >= 70 and t2.get('hanging_success', 0) >= 70:
        bonus += 4

    t1_variance = np.std([t1_auto, t1_teleop, t1_endgame]) if any([t1_auto, t1_teleop, t1_endgame]) else 0
    t2_variance = np.std([t2_auto, t2_teleop, t2_endgame]) if any([t2_auto, t2_teleop, t2_endgame]) else 0
    if (t1_variance > 15 and t2_variance < 10) or (t2_variance > 15 and t1_variance < 10):
        bonus += 3

    penalty = 0
    if t1_consistency < 50 and t2_consistency < 50:
        penalty += 5
    if t1_endgame < 10 and t2_endgame < 10:
        penalty += 3
    if t1.get('penalties', 0) > 5 and t2.get('penalties', 0) > 5:
        penalty += 4

    total_score = scoring_component + phase_component + consistency_component + reliability_component + bonus - penalty
    return round(max(0, min(100, total_score)), 2)

def find_best_alliances(your_team_num, teams_dict):
    if str(your_team_num) not in teams_dict:
        return None
    your_team = teams_dict[str(your_team_num)]
    alliances = []
    for team_num, partner_team in teams_dict.items():
        if str(team_num) == str(your_team_num):
            continue
        score = calc_alliance_score(your_team, partner_team)
        reasons = []
        if (your_team.get('auto_score_avg', 0) >= 20 or partner_team.get('auto_score_avg', 0) >= 20) and (your_team.get('teleop_score_avg', 0) >= 40 or partner_team.get('teleop_score_avg', 0) >= 40):
            reasons.append("Excellent phase distribution - strong auto and teleop coverage")
        combined_score = your_team.get('total_score_avg', 0) + partner_team.get('total_score_avg', 0)
        if combined_score >= 140:
            reasons.append(f"Outstanding combined scoring potential ({combined_score:.0f} pts)")
        elif combined_score >= 120:
            reasons.append(f"Strong combined scoring potential ({combined_score:.0f} pts)")
        alliances.append({
            'partner_team': team_num,
            'alliance_score': score,
            'compatibility_reasons': reasons[:4],
            'partner_stats': {
                'auto': partner_team.get('auto_score_avg', 0),
                'teleop': partner_team.get('teleop_score_avg', 0),
                'endgame': partner_team.get('endgame_score_avg', 0),
                'total': partner_team.get('total_score_avg', 0),
                'consistency': partner_team.get('pattern_analysis', {}).get('consistency', 50),
                'hanging': partner_team.get('hanging_success', 0)
            }
        })
    alliances.sort(key=lambda x: x['alliance_score'], reverse=True)
    return alliances[:8]

def generate_ai_strategy(your_team, partner_team, teams_dict):
    if str(your_team) not in teams_dict or str(partner_team) not in teams_dict:
        return {'error': 'One or both teams not scouted'}
    our = teams_dict[str(your_team)]
    partner = teams_dict[str(partner_team)]
    score = calc_alliance_score(our, partner)
    auto_total = int(our.get('auto_score_avg', 0) + partner.get('auto_score_avg', 0))
    teleop_total = int(our.get('teleop_score_avg', 0) + partner.get('teleop_score_avg', 0))
    endgame_total = int(our.get('endgame_score_avg', 0) + partner.get('endgame_score_avg', 0))
    total = auto_total + teleop_total + endgame_total

    risks = []
    if our.get('pattern_analysis', {}).get('consistency', 50) < 50:
        risks.append('Your team shows high performance variability')
    if partner.get('pattern_analysis', {}).get('consistency', 50) < 50:
        risks.append('Partner team shows high performance variability')
    if our.get('hanging_success', 0) < 50:
        risks.append('Your team has low endgame reliability')
    if partner.get('hanging_success', 0) < 50:
        risks.append('Partner team has low endgame reliability')

    return {
        'score': score,
        'team1_num': your_team,
        'team2_num': partner_team,
        'team1_strengths': get_strengths(our),
        'team2_strengths': get_strengths(partner),
        'team1_name': f"Team {your_team}",
        'team2_name': f"Team {partner_team}",
        'strategy_data': {
            'compatibility_score': score,
            'your_team': {
                'number': your_team,
                'auto': int(our.get('auto_score_avg', 0)),
                'teleop': int(our.get('teleop_score_avg', 0)),
                'endgame': int(our.get('endgame_score_avg', 0)),
                'total': int(our.get('total_score_avg', 0)),
                'hanging': int(our.get('hanging_success', 0)),
                'consistency': our.get('pattern_analysis', {}).get('consistency', 0),
                'penalties': int(our.get('penalties', 0))
            },
            'partner_team': {
                'number': partner_team,
                'auto': int(partner.get('auto_score_avg', 0)),
                'teleop': int(partner.get('teleop_score_avg', 0)),
                'endgame': int(partner.get('endgame_score_avg', 0)),
                'total': int(partner.get('total_score_avg', 0)),
                'hanging': int(partner.get('hanging_success', 0)),
                'consistency': partner.get('pattern_analysis', {}).get('consistency', 0),
                'penalties': int(partner.get('penalties', 0))
            },
            'auto_phase': auto_total,
            'teleop_phase': teleop_total,
            'endgame_phase': endgame_total,
            'estimated_total': total,
            'risks': risks
        }
    }

def get_ranking_list(teams_dict):
    ranked = []
    for tn, team in teams_dict.items():
        ranked.append({
            'team_number': tn,
            'score': team.get('total_score_avg', 0),
            'auto': team.get('auto_score_avg', 0),
            'teleop': team.get('teleop_score_avg', 0),
            'endgame': team.get('endgame_score_avg', 0),
            'consistency': team.get('pattern_analysis', {}).get('consistency', 0),
            'trend': team.get('pattern_analysis', {}).get('trend', '→'),
            'strengths': get_strengths(team)
        })
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked

# ============================================================================
# EMBEDDED ORIGINAL UI (kept from previous original; large)
# ============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FTC DECODE Scouting System V8.0</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; min-height: 100vh; padding: 24px; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 48px 40px; border-radius: 16px; text-align: center; margin-bottom: 32px; box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3); }
        .header h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 12px; letter-spacing: -0.02em; }
        .header p { font-size: 1.125rem; opacity: 0.95; font-weight: 400; }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.8); z-index: 1000; align-items: center; justify-content: center; }
        .modal-overlay.show { display: flex; }
        .modal { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 40px; border-radius: 16px; max-width: 600px; width: 90%; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5); border: 1px solid rgba(59, 130, 246, 0.3); }
        .modal h2 { color: #60a5fa; margin-bottom: 24px; font-size: 1.75rem; font-weight: 700; }
        .modal p { color: #cbd5e1; margin-bottom: 24px; line-height: 1.6; }
        .config-section { background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); padding: 28px; border-radius: 12px; margin-bottom: 32px; border: 1px solid rgba(148, 163, 184, 0.1); }
        .config-section h3 { color: #60a5fa; margin-bottom: 20px; font-size: 1.125rem; font-weight: 600; }
        .config-group { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .input-group { margin-bottom: 20px; }
        .input-group label { display: block; margin-bottom: 8px; color: #cbd5e1; font-weight: 500; font-size: 0.875rem; }
        .input-group input { width: 100%; padding: 12px 16px; border: 1px solid rgba(148, 163, 184, 0.2); background: rgba(15, 23, 42, 0.6); color: #e2e8f0; border-radius: 8px; font-size: 0.9375rem; transition: all 0.2s; font-family: inherit; }
        .input-group input:focus { outline: none; border-color: #3b82f6; background: rgba(15, 23, 42, 0.8); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
        .menu-tabs { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 32px; }
        .menu-btn { padding: 16px 24px; border: 1px solid rgba(148, 163, 184, 0.2); background: rgba(30, 41, 59, 0.6); color: #94a3b8; border-radius: 10px; cursor: pointer; font-weight: 600; font-size: 0.9375rem; transition: all 0.2s; font-family: inherit; }
        .menu-btn:hover { background: rgba(30, 41, 59, 0.9); border-color: rgba(148, 163, 184, 0.3); transform: translateY(-2px); }
        .menu-btn.active { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-color: #3b82f6; color: #ffffff; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4); }
        .section { display: none; }
        .section.active { display: block; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .input-section { background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(10px); padding: 32px; border-radius: 12px; margin-bottom: 24px; border: 1px solid rgba(148, 163, 184, 0.1); }
        .input-section h2 { margin-bottom: 24px; color: #60a5fa; font-size: 1.5rem; font-weight: 700; }
        .button-group { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 24px; }
        .btn { padding: 14px 28px; border: none; border-radius: 8px; font-size: 0.9375rem; font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: inherit; }
        .btn-primary { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4); }
        .btn-secondary { background: rgba(30, 41, 59, 0.8); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }
        .btn-secondary:hover { background: rgba(30, 41, 59, 1); border-color: rgba(148, 163, 184, 0.3); }
        .result-section { background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(10px); padding: 32px; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.1); display: none; }
        .result-section.show { display: block; animation: slideUp 0.3s ease; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .rankings-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
        .rank-card { background: rgba(15, 23, 42, 0.6); padding: 24px; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.1); position: relative; transition: all 0.2s; }
        .rank-card:hover { transform: translateY(-4px); border-color: rgba(59, 130, 246, 0.3); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3); }
        .rank-badge { position: absolute; top: 16px; right: 16px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.875rem; }
        .rank-card h3 { color: #60a5fa; margin-bottom: 16px; font-size: 1.25rem; font-weight: 700; }
        .rank-stat { margin: 10px 0; color: #cbd5e1; font-size: 0.9375rem; }
        .rank-stat strong { color: #60a5fa; font-weight: 600; }
        .strength-tag { display: inline-block; background: rgba(59, 130, 246, 0.2); color: #93c5fd; padding: 6px 12px; border-radius: 16px; margin: 4px 4px 4px 0; font-size: 0.8125rem; font-weight: 500; border: 1px solid rgba(59, 130, 246, 0.3); }
        .error-msg { background: rgba(239, 68, 68, 0.15); color: #fca5a5; padding: 16px 20px; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 20px; display: none; font-size: 0.9375rem; }
        .error-msg.show { display: block; animation: fadeIn 0.3s ease; }
        .success-msg { background: rgba(34, 197, 94, 0.15); color: #86efac; padding: 16px 20px; border-radius: 8px; border-left: 4px solid #22c55e; margin-bottom: 20px; display: none; font-size: 0.9375rem; }
        .success-msg.show { display: block; animation: fadeIn 0.3s ease; }
        .strategy-header { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; margin-bottom: 32px; }
        .team-card { background: rgba(15, 23, 42, 0.6); padding: 24px; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.1); text-align: center; }
        .team-card h3 { color: #60a5fa; margin-bottom: 12px; font-size: 1.125rem; font-weight: 700; }
        .score-box { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 28px 24px; border-radius: 12px; text-align: center; font-weight: 700; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4); }
        .score-box h4 { font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9; margin-bottom: 12px; }
        .score-value { font-size: 3.5rem; font-weight: 800; line-height: 1; }
        .phase-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 24px 0; }
        .phase-card { background: rgba(15, 23, 42, 0.6); padding: 28px; border-radius: 12px; border-left: 4px solid #3b82f6; transition: all 0.2s; }
        .phase-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3); }
        .phase-card h4 { color: #60a5fa; margin-bottom: 16px; font-size: 1.125rem; font-weight: 700; }
        .phase-stat { display: flex; justify-content: space-between; margin: 12px 0; font-size: 0.9375rem; color: #cbd5e1; }
        .phase-stat strong { color: #34d399; font-weight: 700; }
        .risk-list { background: rgba(15, 23, 42, 0.6); padding: 24px; border-radius: 12px; margin-top: 24px; border-left: 4px solid #ef4444; }
        .risk-list h3 { margin-bottom: 16px; font-size: 1.125rem; font-weight: 700; }
        .risk-item { color: #cbd5e1; margin: 12px 0; padding: 12px 16px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border-left: 3px solid #ef4444; font-size: 0.9375rem; }
        .api-status { background: rgba(15, 23, 42, 0.6); padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #f59e0b; }
        .api-status.connected { border-left-color: #22c55e; }
        .config-card { background: rgba(15, 23, 42, 0.6); padding: 28px; border-radius: 12px; margin-bottom: 24px; border: 1px solid rgba(148, 163, 184, 0.1); }
        .config-card h4 { color: #60a5fa; margin-bottom: 16px; font-size: 1.125rem; font-weight: 700; }
        .config-item { margin-bottom: 16px; }
        .config-item label { display: block; color: #cbd5e1; margin-bottom: 8px; font-weight: 500; font-size: 0.875rem; }
        .config-item input { width: 100%; padding: 12px 16px; border: 1px solid rgba(148, 163, 184, 0.2); background: rgba(30, 41, 59, 0.6); color: #e2e8f0; border-radius: 8px; font-size: 0.9375rem; font-family: inherit; }
        @media (max-width: 768px) {
            .menu-tabs { grid-template-columns: repeat(2, 1fr); }
            .strategy-header { grid-template-columns: 1fr; }
            .config-group { grid-template-columns: 1fr; }
            .header h1 { font-size: 2rem; }
        }
    </style>
</head>
<body>
<div class="modal-overlay" id="init-modal">
    <div class="modal">
        <h2>🚀 Initialize Event Data</h2>
        <p>Enter your FTC event code to automatically load match data from the FIRST Tech Challenge API.</p>
        
        <div class="input-group">
            <label>Event Code (e.g., USAKAKM1)</label>
            <input type="text" id="init-event-code" placeholder="Enter FTC event code">
        </div>
        
        <div class="input-group">
            <label>Your Team Number</label>
            <input type="text" id="init-team-number" placeholder="Enter your team number">
        </div>
        
        <div class="input-group">
            <label>Season</label>
            <input type="number" id="init-season" value="2025" placeholder="2025">
        </div>
        
        <div class="success-msg" id="init-success"></div>
        <div class="error-msg" id="init-error"></div>
        
        <div class="button-group">
            <button class="btn btn-primary" onclick="initializeEvent()">Load Event Data</button>
            <button class="btn btn-secondary" onclick="document.getElementById('init-modal').classList.remove('show')">Cancel</button>
        </div>
    </div>
</div>

<div class="container" id="main-content">
    <div class="header">
        <h1>FTC DECODE Scouting System V8.0</h1>
        <p>Professional Analytics & Strategic Intelligence Platform • Server Edition</p>
    </div>
    
    <div class="api-status" id="api-status">
        <strong>⚠️ Not Connected:</strong> Click "Load New Event" to connect to FTC API
    </div>
    
    <div class="config-section">
        <h3>Event Configuration</h3>
        <div class="config-group">
            <div class="input-group">
                <label>Event Code</label>
                <input type="text" id="global-event-code" placeholder="Enter event code" readonly>
            </div>
            <div class="input-group">
                <label>Your Team Number</label>
                <input type="text" id="global-team-number" placeholder="Enter your team number" readonly>
            </div>
        </div>
        <button class="btn btn-primary" style="margin-top:16px;width:100%" onclick="showInitModal()">Load New Event</button>
    </div>
    
    <div class="menu-tabs">
        <button class="menu-btn active" onclick="switchSection('rankings')">Rankings</button>
        <button class="menu-btn" onclick="switchSection('analysis')">Analysis</button>
        <button class="menu-btn" onclick="switchSection('alliance')">Alliance</button>
        <button class="menu-btn" onclick="switchSection('strategy')">Strategy</button>
        <button class="menu-btn" onclick="switchSection('tuning')">Tuning</button>
    </div>
    
    <div id="rankings" class="section active">
        <div class="input-section">
            <h2>Team Rankings</h2>
            <p style="color:#94a3b8;margin-bottom:20px">View ranked performance of all teams at your event</p>
            <div class="button-group">
                <button class="btn btn-primary" onclick="generateRankings()">Load Rankings</button>
                <button class="btn btn-secondary" onclick="clearRankings()">Clear</button>
            </div>
        </div>
        <div class="error-msg" id="error-rank"></div>
        <div class="result-section" id="rankings-result">
            <div id="rankings-grid" class="rankings-grid"></div>
        </div>
    </div>
    
    <div id="analysis" class="section">
        <div class="input-section">
            <h2>Team Analysis</h2>
            <p style="color:#94a3b8;margin-bottom:20px">Deep dive into individual team performance patterns</p>
            <div class="input-group">
                <label>Team Number</label>
                <input type="text" id="analysis-team" placeholder="Leave blank for your team">
            </div>
            <div class="button-group">
                <button class="btn btn-primary" onclick="analyzeTeam()">Analyze</button>
                <button class="btn btn-secondary" onclick="clearAnalysis()">Clear</button>
            </div>
        </div>
        <div class="error-msg" id="error-analysis"></div>
        <div class="result-section" id="analysis-result">
            <div id="analysis-content"></div>
        </div>
    </div>
    
    <div id="alliance" class="section">
        <div class="input-section">
            <h2>Alliance Finder</h2>
            <p style="color:#94a3b8;margin-bottom:20px">Discover optimal alliance partners based on comprehensive compatibility analysis</p>
            <div class="input-group">
                <label>Your Team</label>
                <input type="text" id="your-team-alliance" placeholder="Leave blank for default">
            </div>
            <div class="button-group">
                <button class="btn btn-primary" onclick="findBestAlliance()">Find Best Alliance</button>
                <button class="btn btn-secondary" onclick="clearAllianceFinder()">Clear</button>
            </div>
        </div>
        <div class="error-msg" id="error-alliance"></div>
        <div class="result-section" id="alliance-result">
            <div id="alliance-content"></div>
        </div>
    </div>
    
    <div id="strategy" class="section">
        <div class="input-section">
            <h2>Strategy Generator</h2>
            <p style="color:#94a3b8;margin-bottom:20px">Generate detailed match strategy for your alliance</p>
            <div class="input-group">
                <label>Partner Team Number</label>
                <input type="text" id="partner-team" placeholder="Enter partner team number">
            </div>
            <div class="button-group">
                <button class="btn btn-primary" onclick="generateStrategy()">Generate Strategy</button>
                <button class="btn btn-secondary" onclick="clearStrategy()">Clear</button>
            </div>
        </div>
        <div class="error-msg" id="error-strat"></div>
        <div class="result-section" id="strategy-result">
            <div class="strategy-header">
                <div class="team-card">
                    <h3 id="team1-name">Team 1</h3>
                    <div id="team1-strengths"></div>
                </div>
                <div class="score-box">
                    <h4>Compatibility Score</h4>
                    <div class="score-value" id="score">0</div>
                </div>
                <div class="team-card">
                    <h3 id="team2-name">Team 2</h3>
                    <div id="team2-strengths"></div>
                </div>
            </div>
            <div class="phase-cards" id="phase-cards"></div>
            <div class="risk-list" id="risks"></div>
        </div>
    </div>
    
    <div id="tuning" class="section">
        <div class="input-section">
            <h2>Scoring Configuration Tuning</h2>
            <p style="color:#94a3b8;margin-bottom:24px">Adjust the weights and thresholds to customize how teams are evaluated.</p>
            <div class="success-msg" id="success-tuning"></div>
            <div class="error-msg" id="error-tuning"></div>
            
            <div class="config-editor" id="config-editor"></div>
            
            <div class="button-group">
                <button class="btn btn-primary" onclick="saveConfig()">Save Configuration</button>
                <button class="btn btn-secondary" onclick="loadConfig()">Reset to Saved</button>
            </div>
        </div>
    </div>
</div>

<script>
let currentConfig = null;
let eventData = {
    eventCode: null,
    teamNumber: null,
    season: 2025
};

function showInitModal() {
    document.getElementById('init-modal').classList.add('show');
}

async function initializeEvent() {
    const eventCode = document.getElementById('init-event-code').value.trim();
    const teamNumber = document.getElementById('init-team-number').value.trim();
    const season = document.getElementById('init-season').value || 2025;
    const errorEl = document.getElementById('init-error');
    const successEl = document.getElementById('init-success');
    
    errorEl.classList.remove('show');
    successEl.classList.remove('show');
    
    if (!eventCode || !teamNumber) {
        errorEl.textContent = 'Please enter both event code and team number';
        errorEl.classList.add('show');
        return;
    }
    
    try {
        const resp = await fetch('/api/ftc-event-init', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                event_code: eventCode,
                team_number: teamNumber,
                season: parseInt(season)
            })
        });
        
        const data = await resp.json();
        
        if (data.error) {
            errorEl.textContent = data.error;
            errorEl.classList.add('show');
            return;
        }
        
        successEl.textContent = `✓ ${data.message}! Loaded ${data.match_count} matches for ${data.teams_with_data} teams.`;
        successEl.classList.add('show');
        
        eventData.eventCode = eventCode;
        eventData.teamNumber = teamNumber;
        eventData.season = parseInt(season);
        
        document.getElementById('global-event-code').value = eventCode;
        document.getElementById('global-team-number').value = teamNumber;
        
        const statusEl = document.getElementById('api-status');
        statusEl.className = 'api-status connected';
        statusEl.innerHTML = `<strong>✓ Connected:</strong> ${eventCode} • ${data.teams_with_data} teams with match data`;
        
        setTimeout(() => {
            document.getElementById('init-modal').classList.remove('show');
        }, 2000);
        
    } catch (e) {
        errorEl.textContent = 'Error: ' + e.message;
        errorEl.classList.add('show');
    }
}

function switchSection(section) {
    document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
    document.querySelectorAll(".menu-btn").forEach(b => b.classList.remove("active"));
    document.getElementById(section).classList.add("active");
    event.target.classList.add("active");
    if (section === 'tuning' && !currentConfig) loadConfig();
}

async function loadConfig() {
    try {
        const resp = await fetch("/api/config");
        const config = await resp.json();
        currentConfig = config;
        displayConfig(config);
    } catch (e) {
        console.error("Error loading config:", e);
    }
}

function displayConfig(config) {
    const editor = document.getElementById("config-editor");
    editor.innerHTML = `
        <div class="config-card">
            <h4>⚖️ Scoring Weights</h4>
            <p style="color:#94a3b8;margin-bottom:16px;font-size:0.875rem">These weights determine how much each phase contributes to alliance compatibility scores.</p>
            <div class="config-item"><label>Auto Phase Weight</label><input type="number" step="0.1" min="0" max="1" id="auto_weight" value="${config.scoring_weights.auto_weight}"></div>
            <div class="config-item"><label>Teleop Phase Weight</label><input type="number" step="0.1" min="0" max="1" id="teleop_weight" value="${config.scoring_weights.teleop_weight}"></div>
            <div class="config-item"><label>Endgame Phase Weight</label><input type="number" step="0.1" min="0" max="1" id="endgame_weight" value="${config.scoring_weights.endgame_weight}"></div>
        </div>
        <div class="config-card">
            <h4>🎯 Performance Thresholds</h4>
            <p style="color:#94a3b8;margin-bottom:16px;font-size:0.875rem">These values define what constitutes "strong" performance.</p>
            <div class="config-item"><label>Strong Auto Score</label><input type="number" step="1" min="0" id="strong_auto" value="${config.thresholds.strong_auto}"></div>
            <div class="config-item"><label>Fast Cycle Time</label><input type="number" step="1" min="0" id="fast_cycle" value="${config.thresholds.fast_cycle}"></div>
            <div class="config-item"><label>High Hanging %</label><input type="number" step="5" min="0" max="100" id="high_hanging" value="${config.thresholds.high_hanging}"></div>
            <div class="config-item"><label>High Scoring</label><input type="number" step="10" min="0" id="high_scoring" value="${config.thresholds.high_scoring}"></div>
        </div>
    `;
}

async function saveConfig() {
    const errEl = document.getElementById("error-tuning");
    const succEl = document.getElementById("success-tuning");
    errEl.classList.remove("show");
    succEl.classList.remove("show");
    
    try {
        const newConfig = {
            scoring_weights: {
                auto_weight: parseFloat(document.getElementById("auto_weight").value),
                teleop_weight: parseFloat(document.getElementById("teleop_weight").value),
                endgame_weight: parseFloat(document.getElementById("endgame_weight").value),
                hanging_success_weight: 0.2,
                reliability_multiplier: 3,
                cycle_time_factor: 50
            },
            thresholds: {
                strong_auto: parseInt(document.getElementById("strong_auto").value),
                fast_cycle: parseInt(document.getElementById("fast_cycle").value),
                high_hanging: parseInt(document.getElementById("high_hanging").value),
                high_reliability: 4,
                high_scoring: parseInt(document.getElementById("high_scoring").value),
                consistency_high: 70,
                consistency_medium: 40
            },
            alliance: {
                min_combined_score: 120,
                consistency_threshold: 70
            }
        };
        
        const resp = await fetch("/api/config", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(newConfig)
        });
        
        const result = await resp.json();
        if (result.error) {
            errEl.textContent = result.error;
            errEl.classList.add("show");
        } else {
            succEl.textContent = "✓ Configuration saved successfully!";
            succEl.classList.add("show");
            currentConfig = newConfig;
            setTimeout(() => succEl.classList.remove("show"), 5000);
        }
    } catch (e) {
        errEl.textContent = "Error: " + e.message;
        errEl.classList.add("show");
    }
}

async function generateRankings() {
    const errEl = document.getElementById("error-rank");
    const resEl = document.getElementById("rankings-result");
    errEl.classList.remove("show");
    resEl.classList.remove("show");
    
    if (!eventData.eventCode) {
        errEl.textContent = "Please load an event first";
        errEl.classList.add("show");
        return;
    }
    
    try {
        const resp = await fetch('/api/ftc-rankings', {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                event_code: eventData.eventCode,
                season: eventData.season
            })
        });
        
        const data = await resp.json();
        if (data.error) {
            errEl.textContent = data.error;
            errEl.classList.add("show");
            return;
        }
        
        document.getElementById("rankings-grid").innerHTML = data.rankings.map((t, i) => `
            <div class="rank-card">
                <div class="rank-badge">#${i + 1}</div>
                <h3>Team ${t.team_number}</h3>
                <div class="rank-stat"><strong>Total:</strong> ${t.score.toFixed(1)} pts</div>
                <div class="rank-stat"><strong>Auto:</strong> ${t.auto.toFixed(1)} pts</div>
                <div class="rank-stat"><strong>Teleop:</strong> ${t.teleop.toFixed(1)} pts</div>
                <div class="rank-stat"><strong>Endgame:</strong> ${t.endgame.toFixed(1)} pts</div>
                <div class="rank-stat"><strong>Consistency:</strong> ${t.consistency.toFixed(0)}%</div>
                <div class="rank-stat"><strong>Trend:</strong> ${t.trend}</div>
                <div style="margin-top:12px">${t.strengths.map(s => `<span class="strength-tag">${s}</span>`).join("")}</div>
            </div>
        `).join("");
        resEl.classList.add("show");
    } catch (e) {
        errEl.textContent = "Error: " + e.message;
        errEl.classList.add("show");
    }
}

async function analyzeTeam() {
    const teamNum = document.getElementById("analysis-team").value || eventData.teamNumber;
    const errEl = document.getElementById("error-analysis");
    const resEl = document.getElementById("analysis-result");
    errEl.classList.remove("show");
    resEl.classList.remove("show");
    
    if (!eventData.eventCode || !teamNum) {
        errEl.textContent = "Please load an event and enter a team number";
        errEl.classList.add("show");
        return;
    }
    
    try {
        const resp = await fetch('/api/ftc-analysis', {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                event_code: eventData.eventCode,
                team_number: teamNum,
                season: eventData.season
            })
        });
        
        const data = await resp.json();
        if (data.error) {
            errEl.textContent = data.error;
            errEl.classList.add("show");
            return;
        }
        
        const a = data.analysis;
        document.getElementById("analysis-content").innerHTML = `
            <div style="background:rgba(15,23,42,0.6);padding:32px;border-radius:12px">
                <h2 style="color:#60a5fa;margin-bottom:24px;font-size:1.5rem">Team ${data.team_number}</h2>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:20px;margin-bottom:32px">
                    <div style="background:rgba(30,41,59,0.6);padding:24px;border-radius:12px;border-left:4px solid #3b82f6">
                        <p style="color:#94a3b8;font-weight:600;font-size:0.875rem">Consistency</p>
                        <p style="font-size:2.5rem;margin:12px 0;font-weight:700;color:#60a5fa">${a.consistency.toFixed(0)}%</p>
                        <p style="color:#cbd5e1">${a.reliability}</p>
                    </div>
                    <div style="background:rgba(30,41,59,0.6);padding:24px;border-radius:12px;border-left:4px solid #3b82f6">
                        <p style="color:#94a3b8;font-weight:600;font-size:0.875rem">Trend</p>
                        <p style="font-size:2.5rem;margin:12px 0;font-weight:700;color:#60a5fa">${a.trend}</p>
                    </div>
                    <div style="background:rgba(30,41,59,0.6);padding:24px;border-radius:12px;border-left:4px solid #3b82f6">
                        <p style="color:#94a3b8;font-weight:600;font-size:0.875rem">Peak Score</p>
                        <p style="font-size:2.5rem;margin:12px 0;font-weight:700;color:#60a5fa">${a.peak}</p>
                    </div>
                    <div style="background:rgba(30,41,59,0.6);padding:24px;border-radius:12px;border-left:4px solid #3b82f6">
                        <p style="color:#94a3b8;font-weight:600;font-size:0.875rem">Style</p>
                        <p style="font-size:1.25rem;margin:12px 0;font-weight:600;color:#60a5fa">${a.behavioral}</p>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
                    <div style="background:rgba(30,41,59,0.6);padding:24px;border-radius:12px;border-left:4px solid #3b82f6">
                        <h3 style="color:#60a5fa;margin-bottom:12px">Averages</h3>
                        <p style="color:#cbd5e1">Auto: <strong>${data.team_stats.auto.toFixed(1)}</strong></p>
                        <p style="color:#cbd5e1">Teleop: <strong>${data.team_stats.teleop.toFixed(1)}</strong></p>
                        <p style="color:#cbd5e1">Endgame: <strong>${data.team_stats.endgame.toFixed(1)}</strong></p>
                        <p style="color:#cbd5e1">Total: <strong>${data.team_stats.total.toFixed(1)}</strong></p>
                    </div>
                    <div style="background:rgba(30,41,59,0.6);padding:24px;border-radius:12px;border-left:4px solid #ef4444">
                        <h3 style="color:#ef4444;margin-bottom:12px">Risks & Notes</h3>
                        <p style="color:#cbd5e1">Hanging success: <strong>${data.team_stats.hanging.toFixed(0)}%</strong></p>
                        <p style="color:#cbd5e1">Reliability: <strong>${data.team_stats.reliability.toFixed(1)}</strong></p>
                        <p style="color:#cbd5e1">Penalties total: <strong>${data.team_stats.penalties}</strong></p>
                    </div>
                </div>
            </div>
        `;
        resEl.classList.add("show");
    } catch (e) {
        errEl.textContent = "Error: " + e.message;
        errEl.classList.add("show");
    }
}

async function findBestAlliance() {
    const team = document.getElementById('your-team-alliance').value || eventData.teamNumber;
    const errEl = document.getElementById("error-alliance");
    const resEl = document.getElementById("alliance-result");
    errEl.classList.remove("show");
    resEl.classList.remove("show");
    if (!eventData.eventCode || !team) {
        errEl.textContent = "Please load an event and provide your team number";
        errEl.classList.add("show");
        return;
    }
    try {
        const resp = await fetch('/api/ftc-alliance', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ team_number: team })
        });
        const data = await resp.json();
        if (data.error) {
            errEl.textContent = data.error;
            errEl.classList.add("show");
            return;
        }
        document.getElementById('alliance-content').innerHTML = data.alliances.map(a => `
            <div style="background:rgba(15,23,42,0.6);padding:16px;border-radius:8px;margin-bottom:12px">
                <strong>Partner: ${a.partner_team} — Score: ${a.alliance_score}</strong>
                <div style="margin-top:8px">${a.compatibility_reasons.map(r => `<div style="color:#cbd5e1">${r}</div>`).join('')}</div>
            </div>
        `).join('');
        resEl.classList.add('show');
    } catch (e) {
        errEl.textContent = "Error: " + e.message;
        errEl.classList.add("show");
    }
}

async function generateStrategy() {
    const partner = document.getElementById('partner-team').value;
    const our = document.getElementById('global-team-number').value || '';
    const errEl = document.getElementById('error-strat');
    const resEl = document.getElementById('strategy-result');
    errEl.classList.remove("show");
    resEl.classList.remove("show");
    if (!our || !partner) {
        errEl.textContent = "Please ensure both your team and partner team numbers are provided";
        errEl.classList.add("show");
        return;
    }
    try {
        const resp = await fetch('/api/ftc-strategy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ team1: our, team2: partner })
        });
        const data = await resp.json();
        if (data.error) {
            errEl.textContent = data.error;
            errEl.classList.add("show");
            return;
        }
        document.getElementById('team1-name').innerText = data.team1_name;
        document.getElementById('team2-name').innerText = data.team2_name;
        document.getElementById('team1-strengths').innerHTML = data.team1_strengths.map(s => `<div class="strength-tag">${s}</div>`).join('');
        document.getElementById('team2-strengths').innerHTML = data.team2_strengths.map(s => `<div class="strength-tag">${s}</div>`).join('');
        document.getElementById('score').innerText = data.score;
        document.getElementById('phase-cards').innerHTML = `
            <div class="phase-card"><h4>Auto</h4><div class="phase-stat"><strong>${data.strategy_data.auto_phase}</strong><span>points</span></div></div>
            <div class="phase-card"><h4>Teleop</h4><div class="phase-stat"><strong>${data.strategy_data.teleop_phase}</strong><span>points</span></div></div>
            <div class="phase-card"><h4>Endgame</h4><div class="phase-stat"><strong>${data.strategy_data.endgame_phase}</strong><span>points</span></div></div>
        `;
        document.getElementById('risks').innerHTML = `<h3>Risks</h3>${data.strategy_data.risks.map(r => `<div class="risk-item">${r}</div>`).join('')}`;
        resEl.classList.add('show');
    } catch (e) {
        errEl.textContent = "Error: " + e.message;
        errEl.classList.add("show");
    }
}

// Helpers for UI clearing
function clearRankings() { document.getElementById('rankings-grid').innerHTML = ''; document.getElementById('rankings-result').classList.remove('show'); }
function clearAnalysis() { document.getElementById('analysis-content').innerHTML = ''; document.getElementById('analysis-result').classList.remove('show'); }
function clearAllianceFinder() { document.getElementById('alliance-content').innerHTML = ''; document.getElementById('alliance-result').classList.remove('show'); }
function clearStrategy() { document.getElementById('phase-cards').innerHTML = ''; document.getElementById('risks').innerHTML = ''; document.getElementById('strategy-result').classList.remove('show'); }

</script>
</body>
</html>
"""

# ============================================================================
# FLASK API ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    global CONFIG
    if request.method == 'GET':
        return jsonify(CONFIG)
    data = request.get_json() or {}
    try:
        CONFIG = data
        with open('config.json', 'w') as f:
            json.dump(CONFIG, f, indent=4)
        return jsonify({'message': 'Config saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ftc-event-init', methods=['POST'])
def api_event_init():
    data = request.get_json() or {}
    event_code = data.get('event_code')
    team_number = data.get('team_number')
    season = int(data.get('season', 2025))
    if not event_code or not team_number:
        return jsonify({'error': 'event_code and team_number required'}), 400

    matches = fetch_event_matches(event_code, season=season)
    if matches is None:
        return jsonify({'error': 'Could not fetch matches for that event (check event code or API)'}), 400

    processed_entries = process_ftc_match_data(matches)
    teams_processed = process_teams(processed_entries)

    EVENT_STORE['event_code'] = event_code
    EVENT_STORE['season'] = season
    EVENT_STORE['team_number'] = str(team_number)
    EVENT_STORE['raw_matches'] = matches
    EVENT_STORE['processed_entries'] = processed_entries
    EVENT_STORE['teams_processed'] = teams_processed

    return jsonify({
        'message': 'Event loaded',
        'match_count': len(matches),
        'teams_with_data': len(teams_processed)
    })

@app.route('/api/ftc-rankings', methods=['POST'])
def api_rankings():
    teams = EVENT_STORE.get('teams_processed', {})
    if not teams:
        return jsonify({'error': 'No event loaded'}), 400
    ranked = get_ranking_list(teams)
    return jsonify({'rankings': ranked})

@app.route('/api/ftc-analysis', methods=['POST'])
def api_analysis():
    data = request.get_json() or {}
    team_number = str(data.get('team_number') or EVENT_STORE.get('team_number') or '').strip()
    teams = EVENT_STORE.get('teams_processed', {})
    if not teams:
        return jsonify({'error': 'No event loaded'}), 400
    if not team_number:
        return jsonify({'error': 'team_number required'}), 400
    team = teams.get(team_number)
    if not team:
        return jsonify({'error': f'Team {team_number} not found in loaded event'}), 404
    return jsonify({
        'team_number': team_number,
        'analysis': team.get('pattern_analysis', {}),
        'team_stats': {
            'auto': team.get('auto_score_avg', 0),
            'teleop': team.get('teleop_score_avg', 0),
            'endgame': team.get('endgame_score_avg', 0),
            'total': team.get('total_score_avg', 0),
            'hanging': team.get('hanging_success', 0),
            'reliability': team.get('reliability', 0),
            'penalties': team.get('penalties', 0)
        }
    })

@app.route('/api/ftc-alliance', methods=['POST'])
def api_alliance():
    data = request.get_json() or {}
    team_number = str(data.get('team_number') or EVENT_STORE.get('team_number') or '').strip()
    teams = EVENT_STORE.get('teams_processed', {})
    if not teams:
        return jsonify({'error': 'No event loaded'}), 400
    if not team_number:
        return jsonify({'error': 'team_number required'}), 400
    alliances = find_best_alliances(team_number, teams)
    if alliances is None:
        return jsonify({'error': f'Team {team_number} not found'}), 404
    return jsonify({'alliances': alliances})

@app.route('/api/ftc-strategy', methods=['POST'])
def api_strategy():
    data = request.get_json() or {}
    t1 = str(data.get('team1') or '')
    t2 = str(data.get('team2') or '')
    teams = EVENT_STORE.get('teams_processed', {})
    if not teams:
        return jsonify({'error': 'No event loaded'}), 400
    if not t1 or not t2:
        return jsonify({'error': 'team1 and team2 required'}), 400
    result = generate_ai_strategy(t1, t2, teams)
    return jsonify(result)

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("FTC DECODE SCOUTING SYSTEM V8.0 - CLEAN (API field names, no 'lift')")
    print("Starting server on http://localhost:5000")
    print("Ensure dependencies installed: pip install flask requests numpy")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
