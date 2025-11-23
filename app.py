from flask import Flask, render_template, request, redirect, session, flash
from model import predict_stock
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import traceback
import torch

app = Flask(__name__)
app.secret_key = "secret123"
USERNAME = "admin"
PASSWORD = "admin"
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pwd = request.form.get("password", "").strip()
        if user == USERNAME and pwd == PASSWORD:
            session["user"] = user
            return redirect("/dashboard")
        else:
            flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/")

    # Pre-populated suggestions
    suggested_stocks = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "RELIANCE.NS", "INFY.NS"]
    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        if not ticker:
            flash("Please enter a ticker symbol.", "warning")
            return redirect("/dashboard")
        return redirect(f"/predict/{ticker}")
    return render_template("dashboard.html", stocks=suggested_stocks)


@app.route("/predict/<ticker>")
def predict_stock_route(ticker):
    if "user" not in session:
        return redirect("/")

    # decide device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        out = predict_stock(
            ticker=ticker,
            start="2015-01-01",
            time_step=60,
            test_size=30,
            lstm_epochs=20,
            lstm_batch_size=32,
            arima_order=(5, 1, 0),
            arima_alpha=0.05,
            device=device
        )
    except Exception as e:
        traceback.print_exc()
        flash(f"Error fetching or processing data for '{ticker}': {str(e)}", "danger")
        return redirect("/dashboard")

    # Extract results
    results_df: pd.DataFrame = out["results_df"].copy()
    lstm_metrics = out.get("lstm_metrics", {})
    arima_metrics = out.get("arima_metrics", {})

    # Defensive checks
    if results_df.empty:
        flash("No results to display for this ticker.", "warning")
        return redirect("/dashboard")

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.plot(results_df.index, results_df["actual"], label="Actual", linewidth=2)
        if "lstm_pred" in results_df.columns:
            ax.plot(results_df.index, results_df["lstm_pred"], label="LSTM Prediction", linestyle="--")
        if "arima_pred" in results_df.columns:
            ax.plot(results_df.index, results_df["arima_pred"], label="ARIMA Prediction", linestyle=":")

        # Confidence interval
        if "arima_lower" in results_df.columns and "arima_upper" in results_df.columns:
            ax.fill_between(
                results_df.index,
                results_df["arima_lower"],
                results_df["arima_upper"],
                color="pink",
                alpha=0.3,
                label="ARIMA 95% CI"
            )
    except Exception:
        ax.plot(results_df.index, results_df.iloc[:, 0], label="Actual", linewidth=2)

    ax.set_title(f"{ticker} — Actual vs Predictions (Last {len(results_df)} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    # Convert plot to PNG image encoded as base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # Prepare recent table for display (last 10 rows)
    recent_table = results_df.tail(10).reset_index()
    for col in ["actual", "lstm_pred", "arima_pred", "arima_lower", "arima_upper"]:
        if col in recent_table.columns:
            recent_table[col] = recent_table[col].round(4)

    # Convert metrics to human-friendly rounded dicts
    def round_metrics(metrics):
        return {k: (round(v, 6) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    return render_template(
        "plot.html",
        ticker=ticker,
        plot_url=img_b64,
        lstm_metrics=round_metrics(lstm_metrics),
        arima_metrics=round_metrics(arima_metrics),
        recent_table=recent_table.to_dict(orient="records")
    )
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")
if __name__ == "__main__":

    # Do not use debug=True in production
    app.run(debug=True)
