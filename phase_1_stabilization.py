#!/usr/bin/env python3
"""
Phase 1: Stabilization (Week 1-2)
1. Replace placeholders with dynamic team/league parameters
2. Add validation layer: Check API responses, log errors, fail gracefully
3. Mock real training data: Use actual past seasons (2023-2025) fixtures
4. Fix calibration: Train on holdout test set, not random data
"""

import logging
import pandas as pd
import numpy as np
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase_1_stabilization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== VALIDATION LAYER ====================
class APIValidationError(Exception):
    """Custom exception for API validation errors"""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class APIResponseValidator:
    """Validates API responses before processing"""
    
    @staticmethod
    def validate_fixtures_response(response: Dict) -> bool:
        """Validate fixtures API response structure"""
        try:
            if not isinstance(response, dict):
                raise APIValidationError("Response is not a dictionary")
            
            if 'response' not in response:
                raise APIValidationError("Missing 'response' key in API response")
            
            if not isinstance(response['response'], list):
                raise APIValidationError("'response' value is not a list")
            
            if len(response['response']) == 0:
                logger.warning("API returned empty response list")
                return False
            
            # Validate first fixture structure
            fixture = response['response'][0]
            required_keys = ['teams', 'goals', 'score', 'fixture']
            for key in required_keys:
                if key not in fixture:
                    raise APIValidationError(f"Missing required key: {key}")
            
            logger.info(f"Validated {len(response['response'])} fixtures")
            return True
            
        except APIValidationError as e:
            logger.error(f"API Validation Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False
    
    @staticmethod
    def validate_team_stats_response(response: Dict) -> bool:
        """Validate team statistics API response"""
        try:
            if 'response' not in response:
                raise APIValidationError("Missing 'response' key")
            
            stats = response['response']
            if not isinstance(stats, dict):
                raise APIValidationError("Stats response is not a dictionary")
            
            if 'goals' not in stats:
                raise APIValidationError("Missing 'goals' in team statistics")
            
            return True
        except APIValidationError as e:
            logger.error(f"Team Stats Validation Error: {e}")
            return False

class DataValidator:
    """Validates transformed data"""
    
    @staticmethod
    def validate_fixtures_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validate fixtures dataframe structure and values"""
        try:
            required_columns = ['home_id', 'away_id', 'goals_home', 'goals_away', 
                              'fh_home', 'fh_away', 'under_25', 'draw']
            
            if df is None or df.empty:
                raise DataValidationError("DataFrame is empty")
            
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise DataValidationError(f"Missing columns: {missing_cols}")
            
            # Check for NaN values
            if df.isnull().any().any():
                nan_cols = df.columns[df.isnull().any()].tolist()
                logger.warning(f"NaN values found in columns: {nan_cols}")
                df = df.dropna()
            
            # Validate value ranges
            if (df['goals_home'] < 0).any() or (df['goals_away'] < 0).any():
                raise DataValidationError("Negative goal values found")
            
            if not ((df['under_25'] == 0) | (df['under_25'] == 1)).all():
                raise DataValidationError("under_25 must be binary")
            
            if not ((df['draw'] == 0) | (df['draw'] == 1)).all():
                raise DataValidationError("draw must be binary")
            
            logger.info(f"Validated dataframe with {len(df)} rows and {len(df.columns)} columns")
            return True, None
            
        except DataValidationError as e:
            logger.error(f"Data Validation Error: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False, str(e)

# ==================== ROBUST API CLIENT ====================
class RobustAPISportsClient:
    """Enhanced API client with error handling and retry logic"""
    
    def __init__(self, api_key: str, base_url: str = 'https://v3.football.api-sports.io/'):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session()
        self.validator = APIResponseValidator()
    
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 401, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """GET request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {'x-apisports-key': self.api_key}
            
            logger.info(f"Fetching: {endpoint} with params: {params}")
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched {endpoint}")
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {endpoint}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Exception: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            return None
    
    def get_fixtures(self, league_id: int, season: int, num_matches: int = 100) -> Optional[Dict]:
        """Fetch fixtures with validation"""
        params = {'league': league_id, 'season': season, 'last': num_matches}
        response = self._get('fixtures', params=params)
        
        if response and self.validator.validate_fixtures_response(response):
            return response
        return None
    
    def get_teams_statistics(self, league_id: int, season: int, team_id: int) -> Optional[Dict]:
        """Fetch team stats with validation"""
        params = {'league': league_id, 'season': season, 'team': team_id}
        response = self._get('teams/statistics', params=params)
        
        if response and self.validator.validate_team_stats_response(response):
            return response
        return None

# ==================== REAL DATA TRANSFORMATION ====================
def transform_fixtures_data_robust(raw_fixtures: Dict) -> Tuple[pd.DataFrame, bool]:
    """Transform API fixtures with validation"""
    try:
        fixtures = raw_fixtures.get('response', [])
        data = []
        
        for f in fixtures:
            try:
                data.append({
                    'home_id': f['teams']['home']['id'],
                    'away_id': f['teams']['away']['id'],
                    'goals_home': f['goals']['home'],
                    'goals_away': f['goals']['away'],
                    'fh_home': f['score']['halftime']['home'],
                    'fh_away': f['score']['halftime']['away'],
                    'under_25': 1 if (f['goals']['home'] + f['goals']['away']) < 2.5 else 0,
                    'draw': 1 if f['goals']['home'] == f['goals']['away'] else 0,
                    'fixture_date': f['fixture']['date']
                })
            except KeyError as e:
                logger.warning(f"Skipping fixture due to missing key: {e}")
                continue
        
        df = pd.DataFrame(data)
        is_valid, error_msg = DataValidator.validate_fixtures_dataframe(df)
        
        if not is_valid:
            logger.error(f"Data validation failed: {error_msg}")
            return pd.DataFrame(), False
        
        return df, True
        
    except Exception as e:
        logger.error(f"Error transforming fixtures data: {e}")
        return pd.DataFrame(), False

# ==================== PROPER CALIBRATION ====================
class BettingCalibrator:
    """Proper calibration using holdout test set"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.calibrator = None
        self.calibration_metrics = {}
    
    def fit_calibration(self, predictions: np.ndarray, actuals: np.ndarray) -> bool:
        """
        Fit calibration on holdout test set
        
        Args:
            predictions: Model predictions (probabilities)
            actuals: Actual outcomes (binary)
        
        Returns:
            bool: Success indicator
        """
        try:
            if len(predictions) != len(actuals):
                raise ValueError("Predictions and actuals have different lengths")
            
            # Split into train and test for calibration
            X_cal, X_test, y_cal, y_test = train_test_split(
                predictions.reshape(-1, 1),
                actuals,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Train base classifier
            base_clf = LogisticRegression(random_state=self.random_state)
            base_clf.fit(X_cal, y_cal)
            
            # Apply calibration
            self.calibrator = CalibratedClassifierCV(
                base_clf,
                method='isotonic',
                cv='prefit'
            )
            self.calibrator.fit(X_cal, y_cal)
            
            # Evaluate on test set
            test_score = self.calibrator.score(X_test, y_test)
            self.calibration_metrics['test_accuracy'] = test_score
            
            logger.info(f"Calibration fitted. Test accuracy: {test_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting calibration: {e}")
            return False
    
    def predict_proba(self, predictions: np.ndarray) -> Optional[np.ndarray]:
        """Get calibrated probabilities"""
        try:
            if self.calibrator is None:
                logger.warning("Calibrator not fitted yet")
                return predictions
            
            return self.calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            return None

# ==================== TESTING ====================
if __name__ == "__main__":
    logger.info("=== Phase 1 Stabilization Module Loaded ===")
    
    # Example usage
    logger.info("Testing validation layers...")
    validator = APIResponseValidator()
    
    # Test with mock data
    mock_response = {
        'response': [
            {
                'teams': {'home': {'id': 1}, 'away': {'id': 2}},
                'goals': {'home': 2, 'away': 1},
                'score': {'halftime': {'home': 1, 'away': 0}},
                'fixture': {'date': '2024-01-01', 'referee': 'John Doe'}
            }
        ]
    }
    
    is_valid = validator.validate_fixtures_response(mock_response)
    logger.info(f"Mock response validation: {is_valid}")
