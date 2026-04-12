import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

N_COMPONENTS = 7
RANDOM_STATE = 42

# Module-level scaler so predict() uses the same scale as fit()
_scaler = StandardScaler()


def train_hmm(features: np.ndarray) -> GaussianHMM:
    """
    Train a 7-component Gaussian HMM on the feature matrix.

    Features are z-score normalised before fitting.  We try 'full'
    covariance first; on numerical failure fall back to 'diag'.
    """
    global _scaler
    _scaler = StandardScaler()
    X = _scaler.fit_transform(features)

    for cov_type in ("full", "diag"):
        model = GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type=cov_type,
            n_iter=500,
            random_state=RANDOM_STATE,
            tol=1e-5,
        )
        try:
            model.fit(X)
            return model
        except Exception:
            continue

    raise RuntimeError("HMM training failed with both 'full' and 'diag' covariance types.")


def _scale(features: np.ndarray) -> np.ndarray:
    """Apply the same scaler used during training."""
    return _scaler.transform(features)


def identify_regimes(model: GaussianHMM, feature_df: pd.DataFrame):
    """
    Decode hidden states and map them to semantic regime labels.

    Returns
    -------
    states : np.ndarray  – raw integer state per bar
    regime_map : dict    – {state_id: label}
    bull_state : int     – state with highest mean return
    bear_state : int     – state with lowest mean return
    """
    feature_cols = ["Returns", "Range", "Vol_vol"]
    X = _scale(feature_df[feature_cols].values)
    states = model.predict(X)

    # Mean return per state (first feature = Returns)
    mean_returns = {s: model.means_[s][0] for s in range(N_COMPONENTS)}

    bull_state = max(mean_returns, key=mean_returns.get)
    bear_state = min(mean_returns, key=mean_returns.get)

    # Build descriptive labels
    sorted_states = sorted(mean_returns, key=mean_returns.get)
    regime_map = {}
    for rank, s in enumerate(sorted_states):
        if s == bull_state:
            regime_map[s] = "Bull Run"
        elif s == bear_state:
            regime_map[s] = "Bear/Crash"
        elif rank >= N_COMPONENTS - 2:
            regime_map[s] = "Rally"
        elif rank <= 1:
            regime_map[s] = "Correction"
        else:
            regime_map[s] = "Neutral"

    return states, regime_map, bull_state, bear_state


def add_regimes_to_df(df: pd.DataFrame, model: GaussianHMM):
    """
    Attach decoded regime state and label columns to the feature DataFrame.
    The DataFrame must already contain Returns, Range, Vol_vol columns.
    """
    states, regime_map, bull_state, bear_state = identify_regimes(model, df)
    df = df.copy()
    df["State"] = states
    df["Regime"] = df["State"].map(regime_map)
    df["Bull_state"] = bull_state
    df["Bear_state"] = bear_state
    return df, regime_map, bull_state, bear_state
