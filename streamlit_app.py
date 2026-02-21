# app.py
import streamlit as st
import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from lagos_engine.engine import predict_match

st.set_page_config(page_title="Lagos Betting Machine", layout="centered")

st.title("🇳🇬 Lagos Betting Machine v5.0")
st.markdown("**Pro Edition** — Live fixtures • Premium predictions • Hybrid Ensemble AI")

# Premium Token (change this to your secret)
PREMIUM_TOKEN = "sophie2026"

# API Client
class APISportsClient:
    def __init__(self):
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get_todays_fixtures(self):
        today = datetime.date.today().strftime("%Y-%m-%d")
        url = f"https://v3.football.api-sports.io/fixtures?date={today}"
        headers = {'x-apisports-key': st.secrets["API_SPORTS_KEY"]}
        r = self.session.get(url, headers=headers)
        return r.json() if r.status_code == 200 else None

# Load fixtures
@st.cache_data(ttl=300)
def load_fixtures():
    client = APISportsClient()
    data = client.get_todays_fixtures()
    if not data:
        return pd.DataFrame()
    fixtures = data.get('response', [])
    df = []
    for f in fixtures:
        df.append({
            'league': f['league']['name'],
            'league_id': f['league']['id'],
            'season': f['league']['season'],
            'home': f['teams']['home']['name'],
            'home_id': f['teams']['home']['id'],
            'away': f['teams']['away']['name'],
            'away_id': f['teams']['away']['id'],
            'fixture_id': f['fixture']['id'],
            'time': f['fixture']['date'][-5:]
        })
    return pd.DataFrame(df)

fixtures = load_fixtures()

# Premium check
token = st.sidebar.text_input("Premium Token (free for Sophie)", type="password")
is_premium = token == PREMIUM_TOKEN

if is_premium:
    st.sidebar.success("✅ Premium Unlocked")
else:
    st.sidebar.info("Enter token for full predictions")

# Main UI
if fixtures.empty:
    st.error("No matches today")
else:
    league_list = ["All"] + sorted(fixtures['league'].unique())
    selected_league = st.selectbox("League", league_list)

    if selected_league == "All":
        match_list = fixtures
    else:
        match_list = fixtures[fixtures['league'] == selected_league]

    match_options = match_list.apply(lambda x: f"{x['home']} vs {x['away']} ({x['time']})", axis=1)
    selected_match = st.selectbox("Today's Matches", match_options)

    if st.button("GET PREDICTION", type="primary"):
        # Find the match
        match_row = match_list.iloc[match_options.tolist().index(selected_match)]
        home = match_row['home']
        away = match_row['away']
        home_id = match_row['home_id']
        away_id = match_row['away_id']
        league_id = match_row['league_id']
        season = match_row['season']

        # Run engine
        result = predict_match(home_id, away_id, league_id, season)

        st.success(f"**{home} vs {away}**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{result['p_home']}%")
        col2.metric("Draw", f"{result['p_draw']}%")
        col3.metric("Away Win", f"{result['p_away']}%")

        st.metric("Under 2.5", f"{result['p_under_25']}%", help="Confidence: High")

        if is_premium:
            st.write("**Premium Features**")
            st.write(f"**Predicted Score**: {result['predicted_score']}")
            st.write(f"**FH Home Lead**: {result['fh_home']}%")
            st.write(f"**Confidence Level**: {result['confidence']}%")
            st.write("**CS Groups**:")
            for g, p in result['cs_groups'].items():
                st.write(f"  {g}: {p}%")
        else:
            st.info("🔒 Unlock Premium for full details (contact me)")
            # lagos_engine/data_pipeline.py
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = st.secrets["API_SPORTS_KEY"]

class APISportsClient:
    def __init__(self):
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get(self, endpoint, params=None):
        if params is None:
            params = {}
        url = f"https://v3.football.api-sports.io/{endpoint}"
        headers = {'x-apisports-key': API_KEY}
        r = self.session.get(url, headers=headers, params=params)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error: {r.status_code}")
            return None

client = APISportsClient()

def get_standings(league_id, season):
    data = client.get("standings", {"league": league_id, "season": season})
    if data and data['response']:
        return data['response'][0]['league']['standings'][0]
    return []

def get_last_fixtures(team_id, season, last=5):
    data = client.get("fixtures", {"team": team_id, "season": season, "last": last})
    if data:
        return data['response']
    return []

def get_h2h(home_id, away_id, last=5):
    data = client.get("fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": last})
    if data:
        return data['response']
    return []

def compute_league_avgs(standings):
    total_home_played = sum(s['home']['played'] for s in standings)
    if total_home_played == 0:
        return 1.5, 1.2
    total_home_goals = sum(s['home']['goals']['for'] for s in standings)
    total_away_goals = sum(s['away']['goals']['for'] for s in standings)
    avg_home = total_home_goals / total_home_played
    avg_away = total_away_goals / total_home_played
    return avg_home, avg_away

def get_team_stats(team_id, standings):
    for s in standings:
        if s['team']['id'] == team_id:
            return s
    return None

def compute_strengths(home_id, away_id, standings):
    avg_home, avg_away = compute_league_avgs(standings)
    home_stats = get_team_stats(home_id, standings)
    away_stats = get_team_stats(away_id, standings)
    if not home_stats or not away_stats:
        return 1.5, 1.2
    home_played = home_stats['home']['played']
    home_gf_avg = home_stats['home']['goals']['for'] / home_played if home_played else 1.0
    home_ga_avg = home_stats['home']['goals']['against'] / home_played if home_played else 1.0
    away_played = away_stats['away']['played']
    away_gf_avg = away_stats['away']['goals']['for'] / away_played if away_played else 1.0
    away_ga_avg = away_stats['away']['goals']['against'] / away_played if away_played else 1.0
    home_attack = home_gf_avg / avg_home if avg_home else 1.0
    home_defense = home_ga_avg / avg_home if avg_home else 1.0
    away_attack = away_gf_avg / avg_away if avg_away else 1.0
    away_defense = away_ga_avg / avg_away if avg_away else 1.0
    lambda_home = home_attack * away_defense * avg_home
    lambda_away = away_attack * home_defense * avg_away
    return lambda_home, lambda_away

def compute_last5(team_id, season):
    last_fixtures = get_last_fixtures(team_id, season)
    points = 0
    gd = 0
    for f in last_fixtures:
        if f['fixture']['status']['short'] != 'FT':
            continue
        if f['teams']['home']['id'] == team_id:
            gs = f['goals']['home']
            gc = f['goals']['away']
        else:
            gs = f['goals']['away']
            gc = f['goals']['home']
        gd += gs - gc
        if gs > gc:
            points += 3
        elif gs == gc:
            points += 1
    return points, gd

def compute_h2h(home_id, away_id):
    h2h_fixtures = get_h2h(home_id, away_id)
    home_wins = 0
    draws = 0
    away_wins = 0
    for f in h2h_fixtures:
        if f['fixture']['status']['short'] != 'FT':
            continue
        home_team = f['teams']['home']['id']
        gs_home = f['goals']['home']
        gs_away = f['goals']['away']
        if home_team == home_id:
            if gs_home > gs_away:
                home_wins += 1
            elif gs_home == gs_away:
                draws += 1
            else:
                away_wins += 1
        else:
            if gs_away > gs_home:
                home_wins += 1
            elif gs_home == gs_away:
                draws += 1
            else:
                away_wins += 1
    return home_wins, draws, away_wins

def get_feature_vector(home_id, away_id, league_id, season):
    standings = get_standings(league_id, season)
    lambda_home, lambda_away = compute_strengths(home_id, away_id, standings)
    points_home, gd_home = compute_last5(home_id, season)
    points_away, gd_away = compute_last5(away_id, season)
    home_stats = get_team_stats(home_id, standings)
    away_stats = get_team_stats(away_id, standings)
    position_home = home_stats['rank'] if home_stats else 10
    position_away = away_stats['rank'] if away_stats else 10
    position_diff = position_home - position_away
    h2h_home_wins, h2h_draws, h2h_away_wins = compute_h2h(home_id, away_id)
    features = {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'points_last5_home': points_home,
        'points_last5_away': points_away,
        'gd_last5_home': gd_home,
        'gd_last5_away': gd_away,
        'position_diff': position_diff,
        'h2h_home_wins': h2h_home_wins,
        'h2h_draws': h2h_draws,
        'h2h_away_wins': h2h_away_wins,
    }
    return features
    # lagos_engine/poisson_model.py
import numpy as np
from scipy.stats import poisson

def get_matrix(features):
    lambda_home = features['lambda_home']
    lambda_away = features['lambda_away']
    goals = np.arange(0, 11)
    pmf_home = poisson.pmf(goals, lambda_home)
    pmf_away = poisson.pmf(goals, lambda_away)
    matrix = np.outer(pmf_home, pmf_away)
    matrix /= matrix.sum()  # normalize
    return matrix

def get_probs(features):
    matrix = get_matrix(features)
    p_home = np.sum(np.triu(matrix, 1))
    p_draw = np.trace(matrix)
    p_away = 1 - p_home - p_draw
    return {'p_home': round(p_home * 100, 1), 'p_draw': round(p_draw * 100, 1), 'p_away': round(p_away * 100, 1)}
    # lagos_engine/dixon_coles_model.py
import numpy as np
from scipy.stats import poisson

RHO = 0.03  # Fixed rho value

def tau(i, j, lh, la, rho):
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    elif i == 0 and j == 1:
        return 1 + lh * rho
    elif i == 1 and j == 0:
        return 1 + la * rho
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1

def get_matrix(features):
    lh = features['lambda_home']
    la = features['lambda_away']
    matrix = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            matrix[i, j] = tau(i, j, lh, la, RHO) * poisson.pmf(i, lh) * poisson.pmf(j, la)
    matrix /= matrix.sum()
    return matrix

def get_probs(features):
    matrix = get_matrix(features)
    p_home = np.sum(np.triu(matrix, 1))
    p_draw = np.trace(matrix)
    p_away = 1 - p_home - p_draw
    return {'p_home': round(p_home * 100, 1), 'p_draw': round(p_draw * 100, 1), 'p_away': round(p_away * 100, 1)}
    # lagos_engine/bivariate_model.py
import numpy as np
from scipy.stats import poisson

COV_FACTOR = 0.1  # Fixed covariance factor

def get_matrix(features):
    lh_full = features['lambda_home']
    la_full = features['lambda_away']
    cov = COV_FACTOR * min(lh_full, la_full)
    lh = lh_full - cov
    la = la_full - cov
    matrix = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            p = 0
            for k in range(min(i, j) + 1):
                p += poisson.pmf(k, cov) * poisson.pmf(i - k, lh) * poisson.pmf(j - k, la)
            matrix[i, j] = p
    matrix /= matrix.sum()
    return matrix

def get_probs(features):
    matrix = get_matrix(features)
    p_home = np.sum(np.triu(matrix, 1))
    p_draw = np.trace(matrix)
    p_away = 1 - p_home - p_draw
    return {'p_home': round(p_home * 100, 1), 'p_draw': round(p_draw * 100, 1), 'p_away': round(p_away * 100, 1)}
    # lagos_engine/ml_train.py
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from .data_pipeline import client  # Assuming client is accessible; adjust API_KEY if running locally

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

def fetch_historical_fixtures(league_id, season):
    data = client.get("fixtures", {"league": league_id, "season": season})
    if data:
        return data['response']
    return []

def build_dataset(league_id, season):
    fixtures = fetch_historical_fixtures(league_id, season)
    fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']
    fixtures.sort(key=lambda x: x['fixture']['timestamp'])
    teams = set()
    for f in fixtures:
        teams.add(f['teams']['home']['id'])
        teams.add(f['teams']['away']['id'])
    team_stats = {t: {
        'home_played': 0, 'home_gf': 0, 'home_ga': 0, 'home_points': 0,
        'away_played': 0, 'away_gf': 0, 'away_ga': 0, 'away_points': 0,
        'last_matches': [], 'points': 0, 'gd': 0, 'gf': 0
    } for t in teams}
    h2h = {}
    dataset = []
    for f in fixtures:
        home_id = f['teams']['home']['id']
        away_id = f['teams']['away']['id']
        total_home_played = sum(ts['home_played'] for ts in team_stats.values())
        if total_home_played == 0:
            avg_home, avg_away = 1.5, 1.2
        else:
            total_home_gf = sum(ts['home_gf'] for ts in team_stats.values())
            total_away_gf = sum(ts['away_gf'] for ts in team_stats.values())
            avg_home = total_home_gf / total_home_played
            avg_away = total_away_gf / total_home_played
        hs = team_stats[home_id]
        a_s = team_stats[away_id]
        home_attack = (hs['home_gf'] / hs['home_played'] if hs['home_played'] else 1.0) / avg_home if avg_home else 1.0
        home_defense = (hs['home_ga'] / hs['home_played'] if hs['home_played'] else 1.0) / avg_home if avg_home else 1.0
        away_attack = (a_s['away_gf'] / a_s['away_played'] if a_s['away_played'] else 1.0) / avg_away if avg_away else 1.0
        away_defense = (a_s['away_ga'] / a_s['away_played'] if a_s['away_played'] else 1.0) / avg_away if avg_away else 1.0
        lambda_home = home_attack * away_defense * avg_home
        lambda_away = away_attack * home_defense * avg_away
        last_home = team_stats[home_id]['last_matches'][-5:]
        points_home = sum(3 if gs > gc else 1 if gs == gc else 0 for gs, gc in last_home)
        gd_home = sum(gs - gc for gs, gc in last_home)
        last_away = team_stats[away_id]['last_matches'][-5:]
        points_away = sum(3 if gs > gc else 1 if gs == gc else 0 for gs, gc in last_away)
        gd_away = sum(gs - gc for gs, gc in last_away)
        team_list = sorted(team_stats.items(), key=lambda kv: (-kv[1]['points'], -kv[1]['gd'], -kv[1]['gf']))
        rank_dict = {k: i+1 for i, (k, v) in enumerate(team_list)}
        position_diff = rank_dict[home_id] - rank_dict[away_id]
        pair = tuple(sorted([home_id, away_id]))
        if pair not in h2h:
            h2h[pair] = {'wins_first': 0, 'draws': 0, 'wins_second': 0}
        if home_id < away_id:
            home_wins = h2h[pair]['wins_first']
            away_wins = h2h[pair]['wins_second']
        else:
            home_wins = h2h[pair]['wins_second']
            away_wins = h2h[pair]['wins_first']
        draws = h2h[pair]['draws']
        feature = [lambda_home, lambda_away, points_home, points_away, gd_home, gd_away, position_diff, home_wins, draws, away_wins]
        gs = f['goals']['home']
        gc = f['goals']['away']
        label = 0 if gs > gc else 1 if gs == gc else 2
        dataset.append((feature, label))
        # Update stats
        hs['home_played'] += 1
        hs['home_gf'] += gs
        hs['home_ga'] += gc
        hs_points_add = 3 if gs > gc else 1 if gs == gc else 0
        hs['home_points'] += hs_points_add
        hs['points'] += hs_points_add
        hs['gd'] += gs - gc
        hs['gf'] += gs
        hs['last_matches'].append((gs, gc))
        a_s['away_played'] += 1
        a_s['away_gf'] += gc
        a_s['away_ga'] += gs
        as_points_add = 3 if gc > gs else 1 if gc == gs else 0
        a_s['away_points'] += as_points_add
        a_s['points'] += as_points_add
        a_s['gd'] += gc - gs
        a_s['gf'] += gc
        a_s['last_matches'].append((gc, gs))
        # Update h2h
        if gs > gc:
            if home_id < away_id:
                h2h[pair]['wins_first'] += 1
            else:
                h2h[pair]['wins_second'] += 1
        elif gs < gc:
            if home_id < away_id:
                h2h[pair]['wins_second'] += 1
            else:
                h2h[pair]['wins_first'] += 1
        else:
            h2h[pair]['draws'] += 1
    return dataset

def train_models(league_id=39, season=2025):  # Example: EPL 2025; adjust as needed
    dataset = build_dataset(league_id, season)
    if not dataset:
        print("No data available for training.")
        return
    X = np.array([d[0] for d in dataset])
    y = np.array([d[1] for d in dataset])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # XGBoost
    model_xgb = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42)
    model_xgb.fit(X_train, y_train)
    joblib.dump(model_xgb, 'xgb_model.pkl')
    # Neural Network
    input_size = X.shape[1]
    model_nn = Net(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
    train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    for epoch in range(100):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model_nn(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    torch.save(model_nn.state_dict(), 'nn_model.pkl')
    print("Models trained and saved.")

# To train, run train_models() with appropriate league/season (e.g., historical data).
# lagos_engine/ml_predict.py
import numpy as np
import joblib
import torch
from .ml_train import Net

scaler = joblib.load('scaler.pkl')
model_xgb = joblib.load('xgb_model.pkl')
input_size = 10  # Number of features
model_nn = Net(input_size)
model_nn.load_state_dict(torch.load('nn_model.pkl'))
model_nn.eval()

def prepare_features(features):
    feat_list = [
        features['lambda_home'], features['lambda_away'],
        features['points_last5_home'], features['points_last5_away'],
        features['gd_last5_home'], features['gd_last5_away'],
        features['position_diff'],
        features['h2h_home_wins'], features['h2h_draws'], features['h2h_away_wins']
    ]
    feat_array = np.array([feat_list])
    feat_scaled = scaler.transform(feat_array)
    return feat_scaled

def xgb_predict(features):
    feat = prepare_features(features)
    probs = model_xgb.predict_proba(feat)[0]
    return {'p_home': round(probs[0] * 100, 1), 'p_draw': round(probs[1] * 100, 1), 'p_away': round(probs[2] * 100, 1)}

def nn_predict(features):
    feat = prepare_features(features)
    with torch.no_grad():
        out = model_nn(torch.tensor(feat, dtype=torch.float32))
        probs = out[0].numpy()
    return {'p_home': round(probs[0] * 100, 1), 'p_draw': round(probs[1] * 100, 1), 'p_away': round(probs[2] * 100, 1)}
    # lagos_engine/ensemble.py
import numpy as np

def fuse(all_probs):
    p_home = np.mean([p['p_home'] for p in all_probs])
    p_draw = np.mean([p['p_draw'] for p in all_probs])
    p_away = np.mean([p['p_away'] for p in all_probs])
    total = p_home + p_draw + p_away
    p_home = round(p_home / total * 100, 1)
    p_draw = round(p_draw / total * 100, 1)
    p_away = round(p_away / total * 100, 1)
    return {'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away}

def get_confidence(all_probs):
    homes = [p['p_home'] for p in all_probs]
    std = np.std(homes)
    confidence = round(100 - std * 2, 1)
    return max(50, min(100, confidence))
    # lagos_engine/engine.py
import numpy as np
from scipy.stats import poisson
from .data_pipeline import get_feature_vector
from .poisson_model import get_probs as poisson_probs, get_matrix as poisson_matrix
from .dixon_coles_model import get_probs as dixon_probs, get_matrix as dixon_matrix
from .bivariate_model import get_probs as bivariate_probs, get_matrix as bivariate_matrix
from .ml_predict import xgb_predict, nn_predict
from .ensemble import fuse, get_confidence

def predict_match(home_id, away_id, league_id, season):
    features = get_feature_vector(home_id, away_id, league_id, season)
    p_poisson = poisson_probs(features)
    p_dixon = dixon_probs(features)
    p_bivariate = bivariate_probs(features)
    p_xgb = xgb_predict(features)
    p_nn = nn_predict(features)
    all_p = [p_poisson, p_dixon, p_bivariate, p_xgb, p_nn]
    final_probs = fuse(all_p)
    confidence = get_confidence(all_p)
    m_poisson = poisson_matrix(features)
    m_dixon = dixon_matrix(features)
    m_bivariate = bivariate_matrix(features)
    avg_matrix = (m_poisson + m_dixon + m_bivariate) / 3
    p_under_25 = 0
    for i in range(11):
        for j in range(11):
            if i + j <= 2:
                p_under_25 += avg_matrix[i, j]
    p_under_25 = round(p_under_25 * 100, 1)
    i, j = np.unravel_index(np.argmax(avg_matrix), avg_matrix.shape)
    predicted_score = f"{i}-{j}"
    lambda_fh_home = features['lambda_home'] / 2
    lambda_fh_away = features['lambda_away'] / 2
    goals = np.arange(0, 11)
    fh_matrix = np.outer(poisson.pmf(goals, lambda_fh_home), poisson.pmf(goals, lambda_fh_away))
    fh_matrix /= fh_matrix.sum()
    p_fh_home = round(np.sum(np.triu(fh_matrix, 1)) * 100, 1)
    cs_groups_def = {
        'Group 1': [(1, 0), (2, 0), (2, 1)],
        'Group 2': [(2, 0), (2, 1), (3, 0), (3, 1)],
        'Group 3': [(0, 1), (0, 2), (1, 2)],
        'Group 4': [(0, 2), (1, 2), (0, 3), (1, 3)],
    }
    cs_groups = {}
    for g, scores in cs_groups_def.items():
        prob = sum(avg_matrix[s[0], s[1]] for s in scores if s[0] < 11 and s[1] < 11)
        cs_groups[g] = round(prob * 100, 1)
    return {
        "p_home": final_probs['p_home'],
        "p_draw": final_probs['p_draw'],
        "p_away": final_probs['p_away'],
        "p_under_25": p_under_25,
        "predicted_score": predicted_score,
        "fh_home": p_fh_home,
        "confidence": confidence,
        "cs_groups": cs_groups
    }
​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​import re

with open("streamlit_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove zero-width spaces and similar invisibles
cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', content)

with open("streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(cleaned)

print("File cleaned! Commit and push to GitHub.")
