"""
FTC DECODE Complete Scouting System by Team 11506
Modern UI with AI predictions, analytics, and live data
Run: python 11506_scouting.py
Access: http://127.0.0.1:5000
"""

import json
import os
import numpy as np
import requests
import base64
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.urandom(24)

# API Configuration
FTC_API_BASE = 'https://ftc-api.firstinspires.org/v2.0'
FTC_API_USERNAME = os.environ.get('FTC_API_USERNAME', 'API USERNAME HERE')
FTC_API_KEY = os.environ.get('FTC_API_KEY', 'API KEY HERE')
FTCSCOUT_API_BASE = 'https://api.ftcscout.org/rest/v1'

# ===== NEURAL NETWORK AI =====
class FTCNeuralNetwork:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_samples = 0
    
    def extract_features(self, t1, t2):
        t1_e = t1.get('entries', [])
        t2_e = t2.get('entries', [])
        return np.array([
            np.mean([e.get('autoPoints', 0) for e in t1_e]) if t1_e else 0,
            np.std([e.get('autoPoints', 0) for e in t1_e]) if len(t1_e) > 1 else 0,
            np.mean([e.get('dcPoints', 0) for e in t1_e]) if t1_e else 0,
            np.std([e.get('dcPoints', 0) for e in t1_e]) if len(t1_e) > 1 else 0,
            np.mean([e.get('endgamePoints', 0) for e in t1_e]) if t1_e else 0,
            np.std([e.get('endgamePoints', 0) for e in t1_e]) if len(t1_e) > 1 else 0,
            np.mean([e.get('totalPoints', 0) for e in t1_e]) if t1_e else 0,
            sum([e.get('penaltyPointsCommitted', 0) for e in t1_e]),
            np.mean([e.get('autoPoints', 0) for e in t2_e]) if t2_e else 0,
            np.std([e.get('autoPoints', 0) for e in t2_e]) if len(t2_e) > 1 else 0,
            np.mean([e.get('dcPoints', 0) for e in t2_e]) if t2_e else 0,
            np.std([e.get('dcPoints', 0) for e in t2_e]) if len(t2_e) > 1 else 0,
            np.mean([e.get('endgamePoints', 0) for e in t2_e]) if t2_e else 0,
            np.std([e.get('endgamePoints', 0) for e in t2_e]) if len(t2_e) > 1 else 0,
            np.mean([e.get('totalPoints', 0) for e in t2_e]) if t2_e else 0,
            sum([e.get('penaltyPointsCommitted', 0) for e in t2_e])
        ]).reshape(1, -1)
    
    def train(self, teams_dict, matches):
        if len(teams_dict) < 4:
            return False
        
        X_train, y_train = [], []
        
        for match in matches:
            if not match.get('actualStartTime') and not match.get('postResultTime'):
                continue
            
            red_teams = []
            blue_teams = []
            
            if 'teams' in match:
                for te in match.get('teams', []):
                    tn = str(te.get('teamNumber', ''))
                    st = te.get('station', '')
                    if tn in teams_dict:
                        if 'Red' in st:
                            red_teams.append(tn)
                        else:
                            blue_teams.append(tn)
            
            score_red = match.get('scoreRedFinal') or match.get('scoreRed') or 0
            score_blue = match.get('scoreBlueFinal') or match.get('scoreBlue') or 0
            
            if score_red == 0 and score_blue == 0:
                continue
            
            if len(red_teams) >= 2 and score_red > 0:
                features = self.extract_features(teams_dict[red_teams[0]], teams_dict[red_teams[1]])
                X_train.append(features[0])
                y_train.append([score_red, 1.0 if score_red > score_blue else 0.0])
            
            if len(blue_teams) >= 2 and score_blue > 0:
                features = self.extract_features(teams_dict[blue_teams[0]], teams_dict[blue_teams[1]])
                X_train.append(features[0])
                y_train.append([score_blue, 1.0 if score_blue > score_red else 0.0])
        
        if len(X_train) < 5:
            return False
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        self.training_samples = len(X_train)
        return True
    
    def predict(self, t1, t2):
        if not self.is_trained:
            return None
        features = self.extract_features(t1, t2)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return {
            'expected_score': float(max(0, prediction[0])),
            'win_probability': float(min(max(prediction[1] * 100, 0), 100))
        }

AI_MODEL = FTCNeuralNetwork()

EVENT_STORE = {
    'event_code': None,
    'season': 2025,
    'team_number': None,
    'raw_matches': [],
    'teams_processed': {},
    'match_results': {}
}

def get_ftc_api_headers():
    if not FTC_API_USERNAME or not FTC_API_KEY:
        return None
    auth = f"{FTC_API_USERNAME}:{FTC_API_KEY}"
    b64 = base64.b64encode(auth.encode('ascii')).decode('ascii')
    return {'Authorization': f'Basic {b64}', 'Accept': 'application/json'}

def fetch_event_matches(event_code, season=2025):
    headers = get_ftc_api_headers()
    
    if headers:
        try:
            url = f"{FTC_API_BASE}/{season}/matches/{event_code}"
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                matches = resp.json().get('matches', [])
                print(f"✅ Loaded {len(matches)} matches from FTC API")
                return matches
        except Exception as e:
            print(f"⚠️ FTC API failed: {e}")
    
    try:
        url = f"{FTCSCOUT_API_BASE}/events/{season}/{event_code}/matches"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            matches = resp.json().get('data', [])
            print(f"✅ Loaded {len(matches)} matches from FTCScout API")
            return matches
    except Exception as e:
        print(f"⚠️ FTCScout API failed: {e}")
    
    return None

def process_ftc_match_data(matches):
    teams = {}
    processed_matches = 0
    match_results = {}
    
    for match_idx, match in enumerate(matches):
        if not isinstance(match, dict):
            continue
        
        if not match.get('actualStartTime') and not match.get('postResultTime'):
            continue
        
        match_num = match.get('matchNumber', match_idx + 1)
        
        if 'teams' in match and isinstance(match['teams'], list):
            score_red = match.get('scoreRedFinal') or match.get('scoreRed') or 0
            score_blue = match.get('scoreBlueFinal') or match.get('scoreBlue') or 0
            
            if score_red == 0 and score_blue == 0:
                continue
            
            processed_matches += 1
            
            match_results[match_num] = {
                'red_score': score_red,
                'blue_score': score_blue,
                'red_won': score_red > score_blue,
                'red_teams': [],
                'blue_teams': []
            }
            
            auto_red = match.get('scoreRedAuto') or 0
            auto_blue = match.get('scoreBlueAuto') or 0
            
            for team_entry in match['teams']:
                if not isinstance(team_entry, dict):
                    continue
                    
                team_num = str(team_entry.get('teamNumber', '')).strip()
                station = team_entry.get('station', '') or ''
                
                if not team_num or team_num == '0':
                    continue
                
                is_red = 'Red' in station or 'red' in station.lower()
                
                if is_red:
                    total_points = score_red / 2
                    auto_points = auto_red / 2
                    match_results[match_num]['red_teams'].append(team_num)
                else:
                    total_points = score_blue / 2
                    auto_points = auto_blue / 2
                    match_results[match_num]['blue_teams'].append(team_num)
                
                remaining = total_points - auto_points
                teleop_points = int(remaining * 0.70)
                endgame_points = remaining - teleop_points
                
                entry = {
                    'autoPoints': auto_points,
                    'dcPoints': teleop_points,
                    'endgamePoints': endgame_points,
                    'penaltyPointsCommitted': 0,
                    'totalPoints': total_points,
                    'Team Number': team_num,
                    'Match Number': match_num,
                    'alliance': 'red' if is_red else 'blue',
                    'won': (is_red and score_red > score_blue) or (not is_red and score_blue > score_red)
                }
                
                teams.setdefault(team_num, []).append(entry)
    
    print(f"✅ Processed {processed_matches} completed matches")
    print(f"✅ Found data for {len(teams)} teams")
    return teams, match_results

def analyze_patterns(entries):
    if not entries:
        return {
            'consistency': 0,
            'trend': '→',
            'peak': 0,
            'strength_areas': [],
            'weakness_areas': []
        }
    
    total_scores = [e.get('totalPoints', 0) for e in entries]
    auto_scores = [e.get('autoPoints', 0) for e in entries]
    teleop_scores = [e.get('dcPoints', 0) for e in entries]
    endgame_scores = [e.get('endgamePoints', 0) for e in entries]
    
    if len(total_scores) > 1:
        std_dev = np.std(total_scores)
        mean_score = np.mean(total_scores)
        cv = (std_dev / mean_score * 100) if mean_score > 0 else 100
        consistency = max(0, 100 - cv)
    else:
        consistency = 50
    
    if len(total_scores) >= 3:
        early_avg = np.mean(total_scores[:len(total_scores)//2])
        late_avg = np.mean(total_scores[len(total_scores)//2:])
        trend = '↗' if late_avg > early_avg + 5 else '↘' if late_avg < early_avg - 5 else '→'
    else:
        trend = '→'
    
    peak = max(total_scores) if total_scores else 0
    
    strength_areas = []
    weakness_areas = []
    
    avg_auto = np.mean(auto_scores) if auto_scores else 0
    avg_teleop = np.mean(teleop_scores) if teleop_scores else 0
    avg_endgame = np.mean(endgame_scores) if endgame_scores else 0
    
    if avg_auto >= 20:
        strength_areas.append('Auto')
    else:
        weakness_areas.append('Auto')
    
    if avg_teleop >= 40:
        strength_areas.append('Teleop')
    else:
        weakness_areas.append('Teleop')
    
    if avg_endgame >= 15:
        strength_areas.append('Endgame')
    else:
        weakness_areas.append('Endgame')
    
    return {
        'consistency': round(consistency, 1),
        'trend': trend,
        'peak': peak,
        'strength_areas': strength_areas,
        'weakness_areas': weakness_areas
    }

def process_teams(raw_entries_map):
    teams = {}
    items = raw_entries_map.items() if isinstance(raw_entries_map, dict) else []
    
    for tn, entries in items:
        if not entries:
            continue
            
        auto_scores = [e.get('autoPoints', 0) for e in entries]
        teleop_scores = [e.get('dcPoints', 0) for e in entries]
        endgame_scores = [e.get('endgamePoints', 0) for e in entries]
        total_scores = [e.get('totalPoints', 0) for e in entries]
        
        teams[str(tn)] = {
            'team_number': str(tn),
            'entries': entries,
            'auto_score_avg': np.mean(auto_scores) if auto_scores else 0,
            'auto_score_std': np.std(auto_scores) if len(auto_scores) > 1 else 0,
            'teleop_score_avg': np.mean(teleop_scores) if teleop_scores else 0,
            'teleop_score_std': np.std(teleop_scores) if len(teleop_scores) > 1 else 0,
            'endgame_score_avg': np.mean(endgame_scores) if endgame_scores else 0,
            'endgame_score_std': np.std(endgame_scores) if len(endgame_scores) > 1 else 0,
            'total_score_avg': np.mean(total_scores) if total_scores else 0,
            'total_score_std': np.std(total_scores) if len(total_scores) > 1 else 0,
            'penalties': 0,
            'pattern_analysis': analyze_patterns(entries)
        }
    
    return teams

def get_strengths(team):
    s = []
    if team.get('auto_score_avg', 0) >= 20:
        s.append('Strong Auto')
    if team.get('teleop_score_avg', 0) >= 40:
        s.append('Strong Teleop')
    if team.get('endgame_score_avg', 0) >= 15:
        s.append('Strong Endgame')
    if team.get('pattern_analysis', {}).get('consistency', 0) >= 70:
        s.append('Consistent')
    if team.get('total_score_avg', 0) >= 100:
        s.append('High Scoring')
    return s if s else ['Developing']

def calc_alliance_score_ai(t1, t2):
    if not t1 or not t2:
        return 0.0
    
    ai_prediction = AI_MODEL.predict(t1, t2)
    
    if ai_prediction:
        expected_score = ai_prediction['expected_score']
        win_prob = ai_prediction['win_probability']
        score_component = min((expected_score / 150) * 60, 60)
        win_component = (win_prob / 100) * 40
        return round(max(0, min(100, score_component + win_component)), 2)
    
    combined_avg = (t1.get('total_score_avg', 0) + t2.get('total_score_avg', 0)) / 2
    return round(min((combined_avg / 150) * 100, 100), 2)

def find_best_alliances(your_team_num, teams_dict):
    if str(your_team_num) not in teams_dict:
        return None
    
    your_team = teams_dict[str(your_team_num)]
    alliances = []
    
    for team_num, partner_team in teams_dict.items():
        if str(team_num) == str(your_team_num):
            continue
        
        score = calc_alliance_score_ai(your_team, partner_team)
        ai_prediction = AI_MODEL.predict(your_team, partner_team)
        
        reasons = []
        if ai_prediction:
            reasons.append(f"AI predicts {ai_prediction['expected_score']:.0f} points")
            reasons.append(f"Win probability: {ai_prediction['win_probability']:.1f}%")
        
        combined_score = your_team.get('total_score_avg', 0) + partner_team.get('total_score_avg', 0)
        if combined_score >= 140:
            reasons.append(f"Outstanding combined avg ({combined_score:.0f} pts)")
        
        your_strengths = set(get_strengths(your_team))
        partner_strengths = set(get_strengths(partner_team))
        complementary = partner_strengths - your_strengths
        if complementary:
            reasons.append(f"Complements with: {', '.join(list(complementary)[:2])}")
        
        alliances.append({
            'partner_team': team_num,
            'alliance_score': score,
            'compatibility_reasons': reasons[:4],
            'ai_expected_score': ai_prediction['expected_score'] if ai_prediction else None,
            'ai_win_probability': ai_prediction['win_probability'] if ai_prediction else None,
            'partner_stats': {
                'auto': partner_team.get('auto_score_avg', 0),
                'teleop': partner_team.get('teleop_score_avg', 0),
                'endgame': partner_team.get('endgame_score_avg', 0),
                'total': partner_team.get('total_score_avg', 0),
                'consistency': partner_team.get('pattern_analysis', {}).get('consistency', 50)
            }
        })
    
    alliances.sort(key=lambda x: x['alliance_score'], reverse=True)
    return alliances[:8]

def generate_ai_strategy(your_team, partner_team, teams_dict):
    if str(your_team) not in teams_dict or str(partner_team) not in teams_dict:
        return {'error': 'One or both teams not scouted'}
    
    our = teams_dict[str(your_team)]
    partner = teams_dict[str(partner_team)]
    
    score = calc_alliance_score_ai(our, partner)
    ai_prediction = AI_MODEL.predict(our, partner)
    
    auto_total = int(our.get('auto_score_avg', 0) + partner.get('auto_score_avg', 0))
    teleop_total = int(our.get('teleop_score_avg', 0) + partner.get('teleop_score_avg', 0))
    endgame_total = int(our.get('endgame_score_avg', 0) + partner.get('endgame_score_avg', 0))
    total = auto_total + teleop_total + endgame_total
    
    risks = []
    if our.get('pattern_analysis', {}).get('consistency', 50) < 50:
        risks.append('Your team shows high performance variability')
    if partner.get('pattern_analysis', {}).get('consistency', 50) < 50:
        risks.append('Partner team shows high performance variability')
    
    return {
        'score': score,
        'ai_enabled': ai_prediction is not None,
        'ai_expected_score': ai_prediction['expected_score'] if ai_prediction else None,
        'ai_win_probability': ai_prediction['win_probability'] if ai_prediction else None,
        'team1_num': your_team,
        'team2_num': partner_team,
        'team1_strengths': get_strengths(our),
        'team2_strengths': get_strengths(partner),
        'team1_name': f"Team {your_team}",
        'team2_name': f"Team {partner_team}",
        'strategy_data': {
            'compatibility_score': score,
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
            'auto_std': team.get('auto_score_std', 0),
            'teleop': team.get('teleop_score_avg', 0),
            'teleop_std': team.get('teleop_score_std', 0),
            'endgame': team.get('endgame_score_avg', 0),
            'endgame_std': team.get('endgame_score_std', 0),
            'consistency': team.get('pattern_analysis', {}).get('consistency', 0),
            'trend': team.get('pattern_analysis', {}).get('trend', '→'),
            'strengths': get_strengths(team)
        })
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked

# ===== API ENDPOINTS =====

@app.route('/api/ftc-event-init', methods=['POST'])
def ftc_event_init():
    try:
        data = request.json
        event_code = data.get('event_code', '').strip().upper()
        team_number = data.get('team_number', '').strip()
        season = data.get('season', 2025)
        
        if not event_code or not team_number:
            return jsonify({'error': 'Event code and team number required'}), 400
        
        matches = fetch_event_matches(event_code, season)
        if matches is None:
            return jsonify({'error': f'Could not fetch event data for {event_code}'}), 404
        
        if len(matches) == 0:
            return jsonify({'error': f'No matches found for event {event_code}'}), 404
        
        raw_teams, match_results = process_ftc_match_data(matches)
        teams_dict = process_teams(raw_teams)
        
        ai_trained = AI_MODEL.train(teams_dict, matches)
        
        EVENT_STORE['event_code'] = event_code
        EVENT_STORE['team_number'] = team_number
        EVENT_STORE['season'] = season
        EVENT_STORE['raw_matches'] = matches
        EVENT_STORE['teams_processed'] = teams_dict
        EVENT_STORE['match_results'] = match_results
        
        return jsonify({
            'message': 'Event data loaded successfully',
            'event_code': event_code,
            'season': season,
            'match_count': len(matches),
            'teams_with_data': len(teams_dict),
            'ai_trained': ai_trained,
            'ai_training_samples': AI_MODEL.training_samples if ai_trained else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ftc-rankings', methods=['GET'])
def ftc_rankings():
    try:
        if not EVENT_STORE['teams_processed']:
            return jsonify({'error': 'No event data loaded'}), 400
        
        rankings = get_ranking_list(EVENT_STORE['teams_processed'])
        return jsonify({
            'rankings': rankings,
            'event_code': EVENT_STORE['event_code'],
            'team_count': len(rankings)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ftc-alliance', methods=['POST'])
def ftc_alliance():
    try:
        data = request.json
        team_number = str(data.get('team_number', '')).strip()
        
        if not team_number:
            return jsonify({'error': 'Team number required'}), 400
        
        teams_dict = EVENT_STORE['teams_processed']
        if not teams_dict:
            return jsonify({'error': 'No event data loaded'}), 400
        
        alliances = find_best_alliances(team_number, teams_dict)
        if alliances is None:
            return jsonify({'error': f'Team {team_number} not found'}), 404
        
        return jsonify({
            'your_team': team_number,
            'alliances': alliances,
            'ai_enabled': AI_MODEL.is_trained
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ftc-strategy', methods=['POST'])
def ftc_strategy():
    try:
        data = request.json
        team1 = str(data.get('team1', '')).strip()
        team2 = str(data.get('team2', '')).strip()
        
        if not team1 or not team2:
            return jsonify({'error': 'Both team numbers required'}), 400
        
        teams_dict = EVENT_STORE['teams_processed']
        if not teams_dict:
            return jsonify({'error': 'No event data loaded'}), 400
        
        strategy = generate_ai_strategy(team1, team2, teams_dict)
        if 'error' in strategy:
            return jsonify(strategy), 404
        
        return jsonify(strategy)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'event_loaded': EVENT_STORE['event_code'] is not None,
        'ai_trained': AI_MODEL.is_trained
    })

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FTC DECODE - Live Scouting</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; background: #0a0a0f; color: #fff; overflow-x: hidden; }
        .bg-animation { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); opacity: 0.05; }
        .bg-animation::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(102, 126, 234, 0.03) 10px, rgba(102, 126, 234, 0.03) 20px); animation: move 20s linear infinite; }
        @keyframes move { 0% { transform: translate(0, 0); } 100% { transform: translate(50px, 50px); } }
        .container { position: relative; z-index: 1; max-width: 1800px; margin: 0 auto; padding: 20px; }
        .top-bar { display: flex; justify-content: space-between; align-items: center; padding: 25px 40px; background: rgba(15, 15, 25, 0.95); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(102, 126, 234, 0.2); margin-bottom: 30px; border-radius: 20px; }
        .logo { display: flex; align-items: center; gap: 15px; }
        .logo-icon { width: 50px; height: 50px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.8em; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4); }
        .logo-text h1 { font-size: 1.6em; font-weight: 800; background: linear-gradient(135deg, #667eea, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .logo-text p { font-size: 0.85em; color: #6b7280; margin-top: 2px; }
        .quick-actions { display: flex; gap: 12px; }
        .action-btn { padding: 12px 24px; background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3); border-radius: 10px; color: #667eea; font-weight: 600; cursor: pointer; transition: all 0.3s ease; font-size: 0.95em; }
        .action-btn:hover { background: rgba(102, 126, 234, 0.2); border-color: rgba(102, 126, 234, 0.5); transform: translateY(-2px); }
        .action-btn.primary { background: linear-gradient(135deg, #667eea, #764ba2); border: none; color: #fff; }
        .action-btn.primary:hover { box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
        .setup-panel { background: rgba(15, 15, 25, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 20px; padding: 30px; margin-bottom: 30px; }
        .setup-grid { display: grid; grid-template-columns: 1fr 1fr 0.8fr auto; gap: 15px; align-items: end; }
        .input-wrapper label { display: block; font-size: 0.85em; color: #9ca3af; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .input-wrapper input { width: 100%; padding: 14px 18px; background: rgba(255, 255, 255, 0.05); border: 2px solid rgba(255, 255, 255, 0.1); border-radius: 12px; color: #fff; font-size: 1em; transition: all 0.3s ease; }
        .input-wrapper input:focus { outline: none; border-color: #667eea; background: rgba(255, 255, 255, 0.08); box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1); }
        .load-btn { padding: 14px 32px; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 12px; color: #fff; font-weight: 700; font-size: 1em; cursor: pointer; transition: all 0.3s ease; height: 52px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); }
        .load-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5); }
        .load-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: rgba(15, 15, 25, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 16px; padding: 25px; position: relative; overflow: hidden; transition: all 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); border-color: rgba(102, 126, 234, 0.5); box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3); }
        .stat-card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); }
        .stat-icon { width: 50px; height: 50px; background: rgba(102, 126, 234, 0.1); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.5em; margin-bottom: 15px; }
        .stat-icon.active { background: linear-gradient(135deg, #667eea, #764ba2); animation: pulse-glow 2s ease infinite; }
        @keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); } 50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); } }
        .stat-label { font-size: 0.85em; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .stat-value { font-size: 1.8em; font-weight: 800; background: linear-gradient(135deg, #fff, #e5e7eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; margin-bottom: 30px; }
        .section-card { background: rgba(15, 15, 25, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 20px; padding: 30px; }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }
        .section-title { font-size: 1.5em; font-weight: 700; display: flex; align-items: center; gap: 12px; }
        .live-indicator { display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 20px; font-size: 0.85em; font-weight: 600; }
        .live-dot { width: 8px; height: 8px; background: #ef4444; border-radius: 50%; animation: pulse-live 1.5s ease infinite; }
        @keyframes pulse-live { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.2); } }
        .match-card { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 25px; margin-bottom: 20px; transition: all 0.3s ease; }
        .match-card:hover { background: rgba(255, 255, 255, 0.05); border-color: rgba(102, 126, 234, 0.3); transform: scale(1.02); }
        .match-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); }
        .match-number { font-size: 1.2em; font-weight: 700; color: #667eea; }
        .match-time { font-size: 0.9em; color: #6b7280; }
        .match-content { display: grid; grid-template-columns: 1fr auto 1fr; gap: 25px; align-items: center; }
        .alliance-box { background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 20px; border: 2px solid transparent; transition: all 0.3s ease; }
        .alliance-box.red { border-color: rgba(239, 68, 68, 0.3); }
        .alliance-box.blue { border-color: rgba(59, 130, 246, 0.3); }
        .alliance-box:hover { background: rgba(255, 255, 255, 0.05); }
        .alliance-label { font-size: 0.8em; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; opacity: 0.6; }
        .alliance-box.red .alliance-label { color: #ef4444; }
        .alliance-box.blue .alliance-label { color: #3b82f6; }
        .team-item { background: rgba(255, 255, 255, 0.05); padding: 12px 15px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s ease; }
        .team-item:hover { background: rgba(255, 255, 255, 0.08); transform: translateX(5px); }
        .team-num { font-weight: 700; font-size: 1.1em; }
        .team-avg { font-size: 0.85em; color: #9ca3af; }
        .alliance-score { font-size: 3.5em; font-weight: 900; text-align: center; margin-top: 15px; text-shadow: 0 0 30px currentColor; }
        .alliance-box.red .alliance-score { color: #ef4444; }
        .alliance-box.blue .alliance-score { color: #3b82f6; }
        .vs-divider { text-align: center; font-size: 2em; font-weight: 800; color: #4b5563; }
        .rankings-table { width: 100%; }
        .rank-row { display: grid; grid-template-columns: 60px 1fr 80px 80px 60px; gap: 15px; align-items: center; padding: 15px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; margin-bottom: 10px; transition: all 0.3s ease; }
        .rank-row:hover { background: rgba(255, 255, 255, 0.05); border-color: rgba(102, 126, 234, 0.3); transform: translateX(5px); }
        .rank-badge { width: 45px; height: 45px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.1em; }
        .rank-badge.gold { background: linear-gradient(135deg, #fbbf24, #f59e0b); box-shadow: 0 0 20px rgba(251, 191, 36, 0.5); }
        .rank-badge.silver { background: linear-gradient(135deg, #e5e7eb, #9ca3af); box-shadow: 0 0 20px rgba(229, 231, 235, 0.3); }
        .rank-badge.bronze { background: linear-gradient(135deg, #fb923c, #f97316); box-shadow: 0 0 20px rgba(249, 115, 22, 0.5); }
        .team-info { display: flex; flex-direction: column; }
        .team-name { font-weight: 700; font-size: 1.1em; margin-bottom: 4px; }
        .team-strengths { display: flex; gap: 6px; flex-wrap: wrap; }
        .strength-pill { padding: 3px 10px; background: rgba(102, 126, 234, 0.2); border-radius: 12px; font-size: 0.7em; color: #667eea; font-weight: 600; }
        .score-display { font-size: 1.3em; font-weight: 800; color: #fff; }
        .consistency-bar { width: 100%; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden; }
        .consistency-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 3px; transition: width 0.5s ease; }
        .trend-icon { font-size: 1.5em; }
        .loading-state { text-align: center; padding: 60px 20px; color: #6b7280; }
        .spinner { width: 50px; height: 50px; border: 4px solid rgba(102, 126, 234, 0.2); border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .alliance-card { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 20px; margin-bottom: 15px; transition: all 0.3s ease; }
        .alliance-card:hover { background: rgba(255, 255, 255, 0.05); border-color: rgba(102, 126, 234, 0.3); }
        .alliance-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .partner-number { font-size: 1.3em; font-weight: 800; color: #667eea; }
        .alliance-score-badge { padding: 8px 16px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 12px; font-weight: 700; }
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px; }
        .stat-box { text-align: center; padding: 10px; background: rgba(255, 255, 255, 0.03); border-radius: 8px; }
        .stat-box-label { font-size: 0.7em; color: #9ca3af; text-transform: uppercase; }
        .stat-box-value { font-size: 1.2em; font-weight: 700; color: #fff; margin-top: 5px; }
        .strategy-input { display: grid; grid-template-columns: 1fr 1fr auto; gap: 15px; margin-bottom: 30px; }
        .page { display: none; }
        .page.active { display: block; }
        @media (max-width: 1400px) { .main-grid { grid-template-columns: 1fr; } }
        @media (max-width: 1024px) { .stats-row { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 768px) { .setup-grid { grid-template-columns: 1fr; } .stats-row { grid-template-columns: 1fr; } .match-content { grid-template-columns: 1fr; } .vs-divider { order: -1; } .top-bar { flex-direction: column; gap: 20px; } .quick-actions { width: 100%; justify-content: stretch; } .action-btn { flex: 1; } }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    <div class="container">
        <div class="top-bar">
            <div class="logo">
                <div class="logo-icon">🤖</div>
                <div class="logo-text"><h1>DECODE Scouting</h1><p>Neural Network Analysis System</p></div>
            </div>
            <div class="quick-actions">
                <button class="action-btn" onclick="showPage('home')">🏠 Home</button>
                <button class="action-btn" onclick="showPage('alliance')">🤝 Alliance</button>
                <button class="action-btn" onclick="showPage('strategy')">📊 Strategy</button>
                <button class="action-btn primary" onclick="refreshData()">🔄 Refresh</button>
            </div>
        </div>
        
        <div id="homePage" class="page active">
            <div class="setup-panel">
                <div class="setup-grid">
                    <div class="input-wrapper"><label>Event Code</label><input type="text" id="eventCode" placeholder="AUBRQ1"></div>
                    <div class="input-wrapper"><label>Team Number</label><input type="number" id="teamNumber" placeholder="11506"></div>
                    <div class="input-wrapper"><label>Season</label><input type="number" id="season" value="2025"></div>
                    <button class="load-btn" onclick="loadEvent()" id="loadBtn">🚀 Load Event</button>
                </div>
            </div>
            <div class="stats-row">
                <div class="stat-card"><div class="stat-icon" id="eventIcon">📡</div><div class="stat-label">Event Status</div><div class="stat-value" id="eventStatus">Not Loaded</div></div>
                <div class="stat-card"><div class="stat-icon" id="aiIcon">🤖</div><div class="stat-label">AI Model</div><div class="stat-value" id="aiStatus">Standby</div></div>
                <div class="stat-card"><div class="stat-icon" id="matchIcon">🏁</div><div class="stat-label">Matches</div><div class="stat-value" id="matchCount">0</div></div>
                <div class="stat-card"><div class="stat-icon" id="teamsIcon">👥</div><div class="stat-label">Teams</div><div class="stat-value" id="teamCount">0</div></div>
            </div>
            <div class="main-grid">
                <div class="section-card">
                    <div class="section-header">
                        <div class="section-title"><span>🔴</span>Live Match Feed</div>
                        <div class="live-indicator"><div class="live-dot"></div><span>LIVE</span></div>
                    </div>
                    <div id="matchesFeed"><div class="loading-state"><div class="spinner"></div><p>Load an event to see live match data</p></div></div>
                </div>
                <div class="section-card">
                    <div class="section-header"><div class="section-title"><span>🏆</span>Rankings</div></div>
                    <div id="rankingsContent"><div class="loading-state"><p>Rankings will appear here</p></div></div>
                </div>
            </div>
        </div>

        <div id="alliancePage" class="page">
            <div class="section-card">
                <div class="section-header">
                    <div class="section-title"><span>🤝</span>Alliance Selection</div>
                </div>
                <div class="input-wrapper" style="max-width: 300px; margin-bottom: 30px;">
                    <label>Your Team Number</label>
                    <input type="number" id="allianceTeamNumber" placeholder="11506">
                    <button class="load-btn" onclick="findAlliances()" style="margin-top: 15px; width: 100%;">🔍 Find Best Partners</button>
                </div>
                <div id="allianceResults"><div class="loading-state"><p>Enter your team number to find the best alliance partners</p></div></div>
            </div>
        </div>

        <div id="strategyPage" class="page">
            <div class="section-card">
                <div class="section-header">
                    <div class="section-title"><span>📊</span>Match Strategy</div>
                </div>
                <div class="strategy-input">
                    <div class="input-wrapper">
                        <label>Team 1</label>
                        <input type="number" id="stratTeam1" placeholder="11506">
                    </div>
                    <div class="input-wrapper">
                        <label>Team 2</label>
                        <input type="number" id="stratTeam2" placeholder="12345">
                    </div>
                    <button class="load-btn" onclick="generateStrategy()" style="height: 52px; margin-top: 24px;">🎯 Generate Strategy</button>
                </div>
                <div id="strategyResults"><div class="loading-state"><p>Enter two team numbers to generate an alliance strategy</p></div></div>
            </div>
        </div>
    </div>
    <script>
        const API = window.location.origin;
        let autoRefreshInterval = null;
        
        function showPage(pageName) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(pageName + 'Page').classList.add('active');
        }
        
        async function loadEvent() {
            const code = document.getElementById('eventCode').value.trim();
            const team = document.getElementById('teamNumber').value.trim();
            const season = document.getElementById('season').value.trim();
            
            if (!code || !team) {
                alert('Please enter event code and team number');
                return;
            }
            
            const btn = document.getElementById('loadBtn');
            btn.disabled = true;
            btn.innerHTML = '⏳ Loading...';
            
            try {
                const res = await fetch(`${API}/api/ftc-event-init`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        event_code: code,
                        team_number: team,
                        season: parseInt(season)
                    })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    updateStats(data);
                    await refreshData();
                    btn.innerHTML = '✅ Loaded';
                    
                    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
                    autoRefreshInterval = setInterval(refreshData, 30000);
                } else {
                    alert('Error: ' + data.error);
                    btn.innerHTML = '🚀 Load Event';
                }
            } catch (e) {
                alert('Error: ' + e.message);
                btn.innerHTML = '🚀 Load Event';
            } finally {
                btn.disabled = false;
            }
        }
        
        function updateStats(data) {
            document.getElementById('eventIcon').classList.add('active');
            document.getElementById('eventStatus').textContent = data.event_code;
            document.getElementById('matchIcon').classList.add('active');
            document.getElementById('matchCount').textContent = data.match_count;
            document.getElementById('teamsIcon').classList.add('active');
            document.getElementById('teamCount').textContent = data.teams_with_data;
            
            if (data.ai_trained) {
                document.getElementById('aiIcon').classList.add('active');
                document.getElementById('aiStatus').textContent = 'Trained';
            }
        }
        
        async function refreshData() {
            await Promise.all([loadMatches(), loadRankings()]);
        }
        
        async function loadMatches() {
            try {
                const res = await fetch(`${API}/api/ftc-rankings`);
                const data = await res.json();
                
                if (res.ok && data.rankings.length > 0) {
                    displayMatches(data.rankings);
                }
            } catch (e) {
                console.error('Failed to load matches:', e);
            }
        }
        
        function displayMatches(rankings) {
            const container = document.getElementById('matchesFeed');
            let html = '';
            
            for (let i = 0; i < Math.min(4, Math.floor(rankings.length / 2)); i++) {
                const red1 = rankings[i * 2] || rankings[0];
                const red2 = rankings[i * 2 + 1] || rankings[1];
                const blue1 = rankings[(i * 2 + 2) % rankings.length];
                const blue2 = rankings[(i * 2 + 3) % rankings.length];
                
                const redScore = Math.round(red1.score + red2.score);
                const blueScore = Math.round(blue1.score + blue2.score);
                
                html += `<div class="match-card">
                    <div class="match-info">
                        <div class="match-number">Qualification ${i + 1}</div>
                        <div class="match-time">Recently Completed</div>
                    </div>
                    <div class="match-content">
                        <div class="alliance-box red">
                            <div class="alliance-label">Red Alliance</div>
                            <div class="team-item">
                                <span class="team-num">${red1.team_number}</span>
                                <span class="team-avg">${red1.score.toFixed(0)} avg</span>
                            </div>
                            <div class="team-item">
                                <span class="team-num">${red2.team_number}</span>
                                <span class="team-avg">${red2.score.toFixed(0)} avg</span>
                            </div>
                            <div class="alliance-score">${redScore}</div>
                        </div>
                        <div class="vs-divider">VS</div>
                        <div class="alliance-box blue">
                            <div class="alliance-label">Blue Alliance</div>
                            <div class="team-item">
                                <span class="team-num">${blue1.team_number}</span>
                                <span class="team-avg">${blue1.score.toFixed(0)} avg</span>
                            </div>
                            <div class="team-item">
                                <span class="team-num">${blue2.team_number}</span>
                                <span class="team-avg">${blue2.score.toFixed(0)} avg</span>
                            </div>
                            <div class="alliance-score">${blueScore}</div>
                        </div>
                    </div>
                </div>`;
            }
            
            container.innerHTML = html;
        }
        
        async function loadRankings() {
            try {
                const res = await fetch(`${API}/api/ftc-rankings`);
                const data = await res.json();
                
                if (res.ok) {
                    displayRankings(data.rankings.slice(0, 8));
                }
            } catch (e) {
                console.error('Failed to load rankings:', e);
            }
        }
        
        function displayRankings(rankings) {
            const container = document.getElementById('rankingsContent');
            let html = '<div class="rankings-table">';
            
            rankings.forEach((team, i) => {
                let badgeClass = '';
                if (i === 0) badgeClass = 'gold';
                else if (i === 1) badgeClass = 'silver';
                else if (i === 2) badgeClass = 'bronze';
                
                const strengthsHTML = team.strengths.slice(0, 2).map(s => 
                    `<span class="strength-pill">${s}</span>`
                ).join('');
                
                html += `<div class="rank-row">
                    <div class="rank-badge ${badgeClass}">${i + 1}</div>
                    <div class="team-info">
                        <div class="team-name">${team.team_number}</div>
                        <div class="team-strengths">${strengthsHTML}</div>
                    </div>
                    <div class="score-display">${team.score.toFixed(1)}</div>
                    <div>
                        <div class="consistency-bar">
                            <div class="consistency-fill" style="width:${team.consistency}%"></div>
                        </div>
                    </div>
                    <div class="trend-icon">${team.trend}</div>
                </div>`;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        async function findAlliances() {
            const team = document.getElementById('allianceTeamNumber').value.trim();
            
            if (!team) {
                alert('Please enter your team number');
                return;
            }
            
            const container = document.getElementById('allianceResults');
            container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Finding best alliance partners...</p></div>';
            
            try {
                const res = await fetch(`${API}/api/ftc-alliance`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({team_number: team})
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    displayAlliances(data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        function displayAlliances(data) {
            const container = document.getElementById('allianceResults');
            let html = '';
            
            data.alliances.forEach((alliance, i) => {
                const aiInfo = alliance.ai_expected_score ? 
                    `<p style="color:#9ca3af;font-size:0.9em;margin-top:10px;">
                        AI Prediction: ${alliance.ai_expected_score.toFixed(0)} pts • 
                        Win: ${alliance.ai_win_probability.toFixed(1)}%
                    </p>` : '';
                
                html += `<div class="alliance-card">
                    <div class="alliance-header">
                        <div class="partner-number">Team ${alliance.partner_team}</div>
                        <div class="alliance-score-badge">${alliance.alliance_score.toFixed(1)}% Match</div>
                    </div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-box-label">Auto</div>
                            <div class="stat-box-value">${alliance.partner_stats.auto.toFixed(0)}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-box-label">Teleop</div>
                            <div class="stat-box-value">${alliance.partner_stats.teleop.toFixed(0)}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-box-label">Endgame</div>
                            <div class="stat-box-value">${alliance.partner_stats.endgame.toFixed(0)}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-box-label">Consistency</div>
                            <div class="stat-box-value">${alliance.partner_stats.consistency.toFixed(0)}%</div>
                        </div>
                    </div>
                    ${aiInfo}
                    <div style="margin-top:15px;">
                        ${alliance.compatibility_reasons.map(r => 
                            `<p style="color:#9ca3af;font-size:0.85em;margin-top:5px;">✓ ${r}</p>`
                        ).join('')}
                    </div>
                </div>`;
            });
            
            container.innerHTML = html || '<div class="loading-state"><p>No alliance recommendations available</p></div>';
        }
        
        async function generateStrategy() {
            const team1 = document.getElementById('stratTeam1').value.trim();
            const team2 = document.getElementById('stratTeam2').value.trim();
            
            if (!team1 || !team2) {
                alert('Please enter both team numbers');
                return;
            }
            
            const container = document.getElementById('strategyResults');
            container.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Generating strategy...</p></div>';
            
            try {
                const res = await fetch(`${API}/api/ftc-strategy`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({team1, team2})
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    displayStrategy(data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        function displayStrategy(data) {
            const container = document.getElementById('strategyResults');
            const aiInfo = data.ai_enabled ? 
                `<div style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.3);border-radius:12px;padding:20px;margin-bottom:30px;">
                    <h3 style="font-size:1.2em;margin-bottom:15px;color:#667eea;">🤖 AI Prediction</h3>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                        <div>
                            <div style="font-size:0.85em;color:#9ca3af;margin-bottom:5px;">Expected Score</div>
                            <div style="font-size:2em;font-weight:800;color:#667eea;">${data.ai_expected_score.toFixed(0)}</div>
                        </div>
                        <div>
                            <div style="font-size:0.85em;color:#9ca3af;margin-bottom:5px;">Win Probability</div>
                            <div style="font-size:2em;font-weight:800;color:#667eea;">${data.ai_win_probability.toFixed(1)}%</div>
                        </div>
                    </div>
                </div>` : '';
            
            const risksHTML = data.strategy_data.risks.length > 0 ?
                `<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:20px;margin-top:20px;">
                    <h3 style="font-size:1.1em;margin-bottom:10px;color:#ef4444;">⚠️ Risks</h3>
                    ${data.strategy_data.risks.map(r => `<p style="color:#9ca3af;font-size:0.9em;margin-top:5px;">• ${r}</p>`).join('')}
                </div>` : '';
            
            const html = `
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:30px;">
                    <div style="text-align:center;margin-bottom:30px;">
                        <h2 style="font-size:2em;font-weight:800;margin-bottom:10px;">Alliance Strategy</h2>
                        <p style="color:#9ca3af;">Team ${data.team1_num} + Team ${data.team2_num}</p>
                    </div>
                    
                    ${aiInfo}
                    
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:30px;">
                        <div style="text-align:center;background:rgba(255,255,255,0.03);padding:20px;border-radius:12px;">
                            <div style="font-size:0.85em;color:#9ca3af;margin-bottom:10px;">Auto Phase</div>
                            <div style="font-size:2.5em;font-weight:800;color:#667eea;">${data.strategy_data.auto_phase}</div>
                        </div>
                        <div style="text-align:center;background:rgba(255,255,255,0.03);padding:20px;border-radius:12px;">
                            <div style="font-size:0.85em;color:#9ca3af;margin-bottom:10px;">Teleop Phase</div>
                            <div style="font-size:2.5em;font-weight:800;color:#764ba2;">${data.strategy_data.teleop_phase}</div>
                        </div>
                        <div style="text-align:center;background:rgba(255,255,255,0.03);padding:20px;border-radius:12px;">
                            <div style="font-size:0.85em;color:#9ca3af;margin-bottom:10px;">Endgame Phase</div>
                            <div style="font-size:2.5em;font-weight:800;color:#f093fb;">${data.strategy_data.endgame_phase}</div>
                        </div>
                    </div>
                    
                    <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:30px;border-radius:16px;text-align:center;">
                        <div style="font-size:1em;margin-bottom:10px;opacity:0.9;">Estimated Total Score</div>
                        <div style="font-size:4em;font-weight:900;">${data.strategy_data.estimated_total}</div>
                    </div>
                    
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:30px;">
                        <div>
                            <h3 style="font-size:1.2em;margin-bottom:15px;">Team ${data.team1_num} Strengths</h3>
                            ${data.team1_strengths.map(s => `<div class="strength-pill" style="display:inline-block;margin:5px;">${s}</div>`).join('')}
                        </div>
                        <div>
                            <h3 style="font-size:1.2em;margin-bottom:15px;">Team ${data.team2_num} Strengths</h3>
                            ${data.team2_strengths.map(s => `<div class="strength-pill" style="display:inline-block;margin:5px;">${s}</div>`).join('')}
                        </div>
                    </div>
                    
                    ${risksHTML}
                </div>
            `;
            
            container.innerHTML = html;
        }
    </script>
</body>
</html>'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*70)
    print("🤖 FTC DECODE Scouting System by Team 11506")
    print("="*70)
    print(f"📡 Server: http://127.0.0.1:{port}")
    print(f"🔧 API Configured: {bool(FTC_API_USERNAME and FTC_API_KEY)}")
    print(f"🧠 Neural Network: Ready")
    print("="*70 + "\n")
    app.run(host='127.0.0.1', port=port, debug=True)