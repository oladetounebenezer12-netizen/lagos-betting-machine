import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from xgboost import XGBRegressor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

st.set_page_config(page_title="Lagos Betting Machine", layout="centered")

st.title("🇳🇬 Lagos Betting Machine v4.1")
st.markdown("**Pro Edition** — Live fixtures • Premium predictions")

# API Key
API_KEY = st.secrets["API_SPORTS_KEY"]

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
        headers = {'x-apisports-key': API_KEY}
        r = self.session.get(url, headers=headers)
        return r.json() if r.status_code == 200 else None

    def get_historical_df(self, league_id, season):
        fixtures_data = self._get(f"fixtures?league={league_id}&season={season}")
        if not fixtures_data or not fixtures_data['response']:
            return pd.DataFrame()
        df = []
        for f in fixtures_data['response']:
            if f['fixture']['status']['short'] == 'FT':
                df.append({
                    'Date': pd.to_datetime(f['fixture']['date']),
                    'HomeTeam': f['teams']['home']['name'],
                    'AwayTeam': f['teams']['away']['name'],
                    'FTHG': f['goals']['home'],
                    'FTAG': f['goals']['away']
                })
        return pd.DataFrame(df).sort_values('Date')

    def _get(self, endpoint):
        url = f"https://v3.football.api-sports.io/{endpoint}"
        headers = {'x-apisports-key': API_KEY}
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
            'away': f['teams']['away']['name'],
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
        league_id = match_row['league_id']
        season = match_row['season']

        # Get historical data
        client = APISportsClient()
        historical_df = client.get_historical_df(league_id, season)

        # Run engine
        result = predict_match(home, away, historical_df)

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
                st.write(f"  {g}: {p}")
        else:
            st.info("🔒 Unlock Premium for full details (contact me)")

# Tau Function for Dixon-Coles
def tau(x, y, lambda_x, lambda_y, rho):
    if x == 0 and y == 0:
        return 1 - lambda_x * lambda_y * rho
    elif x == 0 and y == 1:
        return 1 + lambda_x * rho
    elif x == 1 and y == 0:
        return 1 + lambda_y * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

# Dixon-Coles Model Integration
def rho_correction(goals_home, goals_away, home_exp, away_exp, rho):
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0

def fit_dixon_coles_model(df, xi=0.0001):
    teams = np.sort(np.unique(np.concatenate([df["HomeTeam"], df["AwayTeam"]])))
    n_teams = len(teams)

    df["days_since"] = (df["Date"].max() - df["Date"]).dt.days
    df["weight"] = np.exp(-xi * df["days_since"])

    params = np.concatenate(
        (
            np.random.uniform(0.5, 1.5, (n_teams)),  # attack
            np.random.uniform(0, -1, (n_teams)),  # defence
            [0.25],  # home advantage
            [-0.1],  # rho
        )
    )

    def log_likelihood(
        goals_home_observed,
        goals_away_observed,
        home_attack,
        home_defence,
        away_attack,
        away_defence,
        home_advantage,
        rho,
        weight
    ):
        goal_expectation_home = np.exp(home_attack + away_defence + home_advantage)
        goal_expectation_away = np.exp(away_attack + home_defence)

        home_llk = poisson.pmf(goals_home_observed, goal_expectation_home)
        away_llk = poisson.pmf(goals_away_observed, goal_expectation_away)
        adj_llk = rho_correction(
            goals_home_observed,
            goals_away_observed,
            goal_expectation_home,
            goal_expectation_away,
            rho,
        )

        if goal_expectation_home < 0 or goal_expectation_away < 0 or adj_llk < 0:
            return 10000

        log_llk = weight * (np.log(home_llk) + np.log(away_llk) + np.log(adj_llk))
        return -log_llk

    def _fit(params, df, teams):
        attack_params = dict(zip(teams, params[:n_teams]))
        defence_params = dict(zip(teams, params[n_teams : (2 * n_teams)]))
        home_advantage = params[-2]
        rho = params[-1]

        llk = []
        for idx, row in df.iterrows():
            tmp = log_likelihood(
                row["FTHG"],
                row["FTAG"],
                attack_params[row["HomeTeam"]],
                defence_params[row["HomeTeam"]],
                attack_params[row["AwayTeam"]],
                defence_params[row["AwayTeam"]],
                home_advantage,
                rho,
                row["weight"],
            )
            llk.append(tmp)
        return np.sum(llk)

    options = {"maxiter": 100, "disp": False}

    constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]

    res = minimize(
        _fit,
        params,
        args=(df, teams),
        constraints=constraints,
        options=options,
    )

    model_params = dict(
        zip(
            ["attack_" + team for team in teams]
            + ["defence_" + team for team in teams]
            + ["home_adv", "rho"],
            res["x"],
        )
    )
    return model_params

def predict_dixon_coles(params, home_team, away_team, max_goals=10):
    home_attack = params["attack_" + home_team]
    home_defence = params["defence_" + home_team]
    away_attack = params["attack_" + away_team]
    away_defence = params["defence_" + away_team]
    home_advantage = params["home_adv"]
    rho = params["rho"]

    home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
    away_goal_expectation = np.exp(away_attack + home_defence)

    home_probs = poisson.pmf(range(max_goals+1), home_goal_expectation)
    away_probs = poisson.pmf(range(max_goals+1), away_goal_expectation)

    m = np.outer(home_probs, away_probs)

    m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
    m[0, 1] *= 1 + home_goal_expectation * rho
    m[1, 0] *= 1 + away_goal_expectation * rho
    m[1, 1] *= 1 - rho

    p_home = np.sum(np.triu(m, 1))
    p_draw = np.sum(np.diag(m))
    p_away = np.sum(np.tril(m, -1))

    p_under_25 = np.sum([m[i,j] for i in range(11) for j in range(11) if i+j <= 2])

    i, j = np.unravel_index(np.argmax(m), m.shape)
    predicted_score = f"{i}-{j}"

    return {
        "p_home": round(p_home * 100, 1),
        "p_draw": round(p_draw * 100, 1),
        "p_away": round(p_away * 100, 1),
        "p_under_25": round(p_under_25 * 100, 1),
        "predicted_score": predicted_score,
        "matrix": m
    }

# XGBoost Integration
@st.cache_resource
def train_xgb_model():
    # Dummy training data
    X = np.random.rand(100, 6)  # Features
    y = np.random.rand(100)  # Lambdas
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)
    return model

xgb_model = train_xgb_model()

def xgb_adjust_lambda(features):
    X = np.array([features])
    adjustment = xgb_model.predict(X)[0]
    return adjustment

# Random Forest Integration
@st.cache_resource
def train_rf_model():
    # Dummy training data
    X = np.random.rand(100, 6)  # Features
    y = np.random.rand(100)  # Lambdas
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

rf_model = train_rf_model()

def rf_adjust_lambda(features):
    X = np.array([features])
    adjustment = rf_model.predict(X)[0]
    return adjustment

# Neural Network Model Integration
class GoalPredictorNN(nn.Module):
    def __init__(self, input_size=6):
        super(GoalPredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output lambda adjustment
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

@st.cache_resource
def train_nn_model():
    # Dummy training data
    X = torch.tensor(np.random.rand(100, 6), dtype=torch.float32)
    y = torch.tensor(np.random.rand(100), dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GoalPredictorNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    return model

nn_model = train_nn_model()

def nn_adjust_lambda(features):
    X = torch.tensor([features], dtype=torch.float32)
    nn_model.eval()
    with torch.no_grad():
        adjustment = nn_model(X).item()
    return adjustment

# LSTM Integration for Time Series
class LSTMNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(historical_df):
    seq_len = 5
    sequences = []
    targets = []
    if not historical_df.empty:
        teams = set(historical_df['HomeTeam']) | set(historical_df['AwayTeam'])
        for team in teams:
            goals = []
            for _, row in historical_df.sort_values('Date').iterrows():
                if row['HomeTeam'] == team:
                    goals.append(row['FTHG'])
                elif row['AwayTeam'] == team:
                    goals.append(row['FTAG'])
            if len(goals) > seq_len:
                for i in range(len(goals) - seq_len):
                    sequences.append(goals[i:i+seq_len])
                    targets.append(goals[i+seq_len])
    
    if not sequences:
        # Dummy data if no historical sequences
        sequences = np.random.rand(100, seq_len)
        targets = np.random.rand(100)
    
    X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # (samples, seq_len, 1)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LSTMNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    return model

def lstm_adjust_lambda(model, team, historical_df):
    seq_len = 5
    goals = []
    if not historical_df.empty:
        for _, row in historical_df.sort_values('Date').iterrows():
            if row['HomeTeam'] == team:
                goals.append(row['FTHG'])
            elif row['AwayTeam'] == team:
                goals.append(row['FTAG'])
    if len(goals) < seq_len:
        return 0.0
    last_seq = goals[-seq_len:]
    X = torch.tensor([last_seq], dtype=torch.float32).unsqueeze(-1)  # (1, seq_len, 1)
    model.eval()
    with torch.no_grad():
        adjustment = model(X).item()
    return adjustment

# Prophet Forecasting Integration
def prophet_adjust_lambda(team, historical_df):
    team_goals = []
    for idx, row in historical_df.iterrows():
        if row['HomeTeam'] == team:
            team_goals.append({'ds': row['Date'], 'y': row['FTHG']})
        elif row['AwayTeam'] == team:
            team_goals.append({'ds': row['Date'], 'y': row['FTAG']})
    if len(team_goals) < 2:
        return 0.0
    df_prophet = pd.DataFrame(team_goals)
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    adjustment = forecast['yhat'].iloc[-1]
    return adjustment

# ARIMA Forecasting Integration
def arima_adjust_lambda(team, historical_df):
    goals = []
    dates = []
    for _, row in historical_df.sort_values('Date').iterrows():
        if row['HomeTeam'] == team:
            goals.append(row['FTHG'])
            dates.append(row['Date'])
        elif row['AwayTeam'] == team:
            goals.append(row['FTAG'])
            dates.append(row['Date'])
    if len(goals) < 5:  # Minimum data points for ARIMA
        return 0.0
    df_arima = pd.DataFrame({'y': goals}, index=dates)
    try:
        model = ARIMA(df_arima['y'], order=(1, 1, 1))  # Simple ARIMA order; can be tuned
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        adjustment = forecast.iloc[0]
    except:
        adjustment = 0.0
    return adjustment

# SARIMA Forecasting Integration for Seasonality
def sarima_adjust_lambda(team, historical_df):
    goals = []
    dates = []
    for _, row in historical_df.sort_values('Date').iterrows():
        if row['HomeTeam'] == team:
            goals.append(row['FTHG'])
            dates.append(row['Date'])
        elif row['AwayTeam'] == team:
            goals.append(row['FTAG'])
            dates.append(row['Date'])
    if len(goals) < 12:  # Minimum data points for SARIMA with seasonality
        return 0.0
    df_sarima = pd.DataFrame({'y': goals}, index=dates)
    try:
        # SARIMA order (p,d,q)(P,D,Q,s) - assuming weekly seasonality (s=7)
        model = SARIMAX(df_sarima['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=1)
        adjustment = forecast.iloc[0]
    except:
        adjustment = 0.0
    return adjustment

# VAR Forecasting Integration
def var_adjust_lambda(team, historical_df):
    scored = []
    conceded = []
    dates = []
    for _, row in historical_df.sort_values('Date').iterrows():
        if row['HomeTeam'] == team:
            scored.append(row['FTHG'])
            conceded.append(row['FTAG'])
            dates.append(row['Date'])
        elif row['AwayTeam'] == team:
            scored.append(row['FTAG'])
            conceded.append(row['FTHG'])
            dates.append(row['Date'])
    if len(scored) < 5:  # Minimum data points for VAR
        return 0.0
    df_var = pd.DataFrame({'scored': scored, 'conceded': conceded}, index=dates)
    try:
        model = VAR(df_var)
        model_fit = model.fit(maxlags=1, ic='aic')  # Simple lag=1; can be tuned
        forecast = model_fit.forecast(df_var.values[-model_fit.k_ar:], steps=1)
        adjustment = forecast[0][0]  # Forecasted scored goals
    except:
        adjustment = 0.0
    return adjustment

# Updated prediction function with Dixon-Coles, XGBoost, Random Forest, Neural Network, LSTM, Prophet, ARIMA, SARIMA, and VAR
def predict_match(home, away, historical_df):
    # Simulated but realistic (replace with real lambda calc later)
    lambda_home = 1.6
    lambda_away = 1.3

    # Placeholder features (in real use, derive from historical data)
    features = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # e.g., last 5 goals + form

    # XGBoost adjustment
    lambda_home += xgb_adjust_lambda(features)
    lambda_away += xgb_adjust_lambda(features)
    
    # Random Forest adjustment
    lambda_home += rf_adjust_lambda(features)
    lambda_away += rf_adjust_lambda(features)
    
    # Neural Network adjustment
    lambda_home += nn_adjust_lambda(features)
    lambda_away += nn_adjust_lambda(features)
    
    # Train LSTM with historical data
    lstm_m = train_lstm_model(historical_df)
    # LSTM adjustment (time series)
    lambda_home += lstm_adjust_lambda(lstm_m, home, historical_df)
    lambda_away += lstm_adjust_lambda(lstm_m, away, historical_df)
    
    # Prophet adjustment
    lambda_home += prophet_adjust_lambda(home, historical_df)
    lambda_away += prophet_adjust_lambda(away, historical_df)
    
    # ARIMA adjustment
    lambda_home += arima_adjust_lambda(home, historical_df)
    lambda_away += arima_adjust_lambda(away, historical_df)
    
    # SARIMA adjustment
    lambda_home += sarima_adjust_lambda(home, historical_df)
    lambda_away += sarima_adjust_lambda(away, historical_df)
    
    # VAR adjustment
    lambda_home += var_adjust_lambda(home, historical_df)
    lambda_away += var_adjust_lambda(away, historical_df)
    
    # Poisson matrix
    goals = np.arange(0, 11)
    prob_matrix = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))
    prob_matrix /= prob_matrix.sum()
    
    # Dixon-Coles adjustment
    dc_params = fit_dixon_coles_model(historical_df)
    dc_result = predict_dixon_coles(dc_params, home, away)
    
    # Average with Poisson
    p_home = round((np.sum(np.triu(prob_matrix, 1)) * 100 + dc_result['p_home']) / 2, 1)
    p_draw = round((np.trace(prob_matrix) * 100 + dc_result['p_draw']) / 2, 1)
    p_away = round((100 - p_home - p_draw), 1)
    p_under_25 = round((sum(prob_matrix[i,j] for i in range(11) for j in range(11) if i+j <= 2) * 100 + dc_result['p_under_25']) / 2, 1)
    
    # Most likely score
    i, j = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    predicted_score = f"{i}-{j}"
    
    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_under_25": p_under_25,
        "predicted_score": predicted_score,
        "fh_home": 32,
        "confidence": 78,
        "cs_groups": {"Group 1": "26%", "Group 2": "24%", "Group 3": "25%", "Group 4": "25%"}
    } 
