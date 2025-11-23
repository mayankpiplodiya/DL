import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import math
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Utilities

def create_sequences(array_2d: np.ndarray, time_step: int):
    X, y = [], []
    n = len(array_2d)
    if n <= time_step:
        return np.empty((0, time_step, 1)), np.empty((0,))
    for i in range(time_step, n):
        X.append(array_2d[i - time_step:i, 0])
        y.append(array_2d[i, 0])
    X = np.array(X)
    y = np.array(y)
    if X.size:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def evaluate_metrics(true_vals: np.ndarray, pred_vals: np.ndarray):
    true = np.array(true_vals).reshape(-1)
    pred = np.array(pred_vals).reshape(-1)
    mse = mean_squared_error(true, pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
# PyTorch LSTM model

class TorchLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.0):
        super(TorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_lstm_pytorch(
    series: pd.Series,
    time_step: int = 60,
    test_size: int = 30,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare values
    values = series.values.reshape(-1, 1).astype("float32")
    n_total = len(values)
    min_required = time_step + test_size + 1
    if n_total < min_required:
        raise ValueError(f"Not enough data: need >= {min_required} rows, got {n_total}.")

    # train / test split index on raw values
    train_end_idx = n_total - test_size

    # Fit scaler on training portion only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:train_end_idx])
    scaled_all = scaler.transform(values)

    # Create sequences
    X_all, y_all = create_sequences(scaled_all, time_step)
    if X_all.shape[0] < test_size:
        raise ValueError("Not enough sequences to form the requested test set. Reduce time_step or test_size.")

    # Split train/test
    X_train = X_all[:-test_size]
    y_train = y_all[:-test_size]
    X_test = X_all[-test_size:]
    y_test = y_all[-test_size:]

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build model and optimizer
    model = TorchLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.0).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with simple early stopping
    best_loss = float("inf")
    patience = 6
    wait = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)                      # (batch,1)
            loss = criterion(out.squeeze(), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = float(np.mean(losses)) if losses else 0.0

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best state if available
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy()  

    lstm_pred_inv = scaler.inverse_transform(preds_scaled.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    metrics = evaluate_metrics(y_test_inv, lstm_pred_inv)

    return {
        "model": model.cpu(),
        "scaler": scaler,
        "X_test_scaled": X_test,
        "y_test_inv": y_test_inv,
        "lstm_pred_inv": lstm_pred_inv,
        "metrics": metrics
    }
# ARIMA training & forecasting

def train_arima_forecast(series: pd.Series, test_size: int = 30, arima_order: tuple = (5, 1, 0), alpha: float = 0.05):
    values = series.copy()
    n = len(values)
    if n < test_size + 10:
        raise ValueError("Not enough data for ARIMA forecasting with requested test_size.")

    train = values[: -test_size]
    test = values[-test_size:]

    arima_model = ARIMA(train, order=arima_order)
    arima_fit = arima_model.fit()

    forecast_res = arima_fit.get_forecast(steps=test_size)
    pred_mean = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=alpha)
    # Align CI index to test index
    try:
        ci.index = test.index
    except Exception:
        # fallback: set simple RangeIndex with same length
        ci.index = test.index if hasattr(test, "index") else pd.RangeIndex(start=0, stop=len(pred_mean))

    arima_pred = np.array(pred_mean).reshape(-1, 1)
    arima_metrics = evaluate_metrics(test.values.reshape(-1, 1), arima_pred)

    return {
        "arima_pred": arima_pred,
        "ci": ci,
        "metrics": arima_metrics,
        "train_fit": arima_fit
    }
# predict_stock

def predict_stock(
    ticker: str,
    start: str = "2015-01-01",
    end: str = None,
    time_step: int = 60,
    test_size: int = 30,
    lstm_epochs: int = 30,
    lstm_batch_size: int = 32,
    lstm_lr: float = 1e-3,
    arima_order: tuple = (5, 1, 0),
    arima_alpha: float = 0.05,
    device: torch.device = None
):
    # Fetch data
    if end is None:
        df = yf.download(ticker, start=start, progress=False)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check ticker symbol and date range.")

    # Keep only Close and drop NaNs
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df[['Close']].dropna()

    # basic index sanity
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    if len(df) < (time_step + test_size + 1):
        raise ValueError("Not enough data for given time_step/test_size. Reduce them or increase date range.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train LSTM (PyTorch)
    lstm_res = train_lstm_pytorch(
        df['Close'],
        time_step=time_step,
        test_size=test_size,
        epochs=lstm_epochs,
        batch_size=lstm_batch_size,
        lr=lstm_lr,
        device=device
    )
    # Train ARIMA
    arima_res = train_arima_forecast(
        df['Close'],
        test_size=test_size,
        arima_order=arima_order,
        alpha=arima_alpha
    )
    # Assemble results DataFrame for test period
    test_dates = df.index[-test_size:]
    results_df = pd.DataFrame({
        "actual": lstm_res["y_test_inv"].flatten(),
        "lstm_pred": lstm_res["lstm_pred_inv"].flatten(),
        "arima_pred": arima_res["arima_pred"].flatten()
    }, index=test_dates)
    # Add confidence intervals to results
    ci_df = arima_res["ci"].copy()

    # try to name columns sensibly
    if ci_df.shape[1] >= 2:

        # take first two columns as lower/upper if explicit names missing
        try:
            cols = list(ci_df.columns)
            ci_df.columns = ["arima_lower", "arima_upper"]
        except Exception:
            ci_df = ci_df.iloc[:, :2].copy()
            ci_df.columns = ["arima_lower", "arima_upper"]
    else:
        # create NaNs
        ci_df = pd.DataFrame(index=test_dates, data={"arima_lower": np.nan, "arima_upper": np.nan})

    # ensure index and join
    ci_df.index = test_dates
    results_df = results_df.join(ci_df, how="left")

    return {
        "ticker": ticker,
        "data": df,
        "results_df": results_df,
        "lstm_metrics": lstm_res["metrics"],
        "arima_metrics": arima_res["metrics"],
        "lstm_model": lstm_res["model"],
        "lstm_scaler": lstm_res["scaler"],
        "arima_fit": arima_res["train_fit"]
    }
