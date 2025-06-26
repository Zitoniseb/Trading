import os
import math
import datetime
import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText

# ── CONFIGURATION ──
BUDGET_EUR        = 1000.0
send_email_alerts = True
save_results      = True

# ── EMAIL CREDENTIALS ──
FROM_EMAIL   = "twagirayezusebastien@gmail.com"
TO_EMAIL     = FROM_EMAIL
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
if send_email_alerts and not APP_PASSWORD:
    raise RuntimeError("Please set GMAIL_APP_PASSWORD to your 16-char App Password")

# ── INTRADAY PARAMETERS ──
PERIOD        = "2d"      # last 2 calendar days
INTERVAL      = "60m"     # hourly bars
MA_WINDOW     = 20
VOL_WINDOW    = 5
RSI_WINDOW    = 14
VOLUME_SPIKE  = 1.2
RSI_THRESHOLD = 40

def send_email(subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(FROM_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

def fetch_sp500():
    url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, flavor="lxml")[0]
    return table["Symbol"].str.replace(r"\.", "-", regex=True).tolist()

def scan_intraday_breakouts(tickers):
    signals = []
    data = yf.download(
        tickers,
        period=PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        progress=False,
        auto_adjust=False
    )
    for sym in tickers:
        df = data.get(sym)
        if df is None or len(df) < MA_WINDOW + 2:
            continue

        # drop last bar if incomplete
        df = df.iloc[:-1]

        df["MA"]    = df["Close"].rolling(MA_WINDOW).mean()
        df["VolMA"] = df["Volume"].rolling(VOL_WINDOW).mean()
        df["RSI"]   = 100 - (
            100 / (1 + df["Close"]
                      .pct_change()
                      .rolling(RSI_WINDOW)
                      .apply(lambda x: (x[x>0].sum()/abs(x[x<0].sum()))
                                      if abs(x[x<0].sum())>0 else 0))
        )

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        o      = float(latest["Open"])
        c      = float(latest["Close"])
        ma     = float(latest["MA"])
        vol    = int(latest["Volume"])
        avgvol = float(latest["VolMA"])
        rsi    = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 0.0

        if c > ma and c > float(prev["Close"]) and vol > avgvol * VOLUME_SPIKE and rsi > RSI_THRESHOLD:
            signals.append({
                "Ticker":  sym,
                "BarTime": latest.name,       # pandas Timestamp
                "Open":    round(o, 2),
                "Close":   round(c, 2),
                "MA":      round(ma, 2),
                "Volume":  vol,
                "AvgVol":  int(avgvol),
                "RSI":     round(rsi, 2),
                "ROI%":    round((c - o) / o * 100, 2)
            })
    return pd.DataFrame(signals)

def allocate_budget(signals, budget):
    signals["profit_per_euro"] = signals["ROI%"] / 100
    signals = signals.sort_values("profit_per_euro", ascending=False).reset_index(drop=True)

    cash   = budget
    orders = []
    for _, row in signals.iterrows():
        if cash < row["Open"]:
            break
        qty    = math.floor(cash / row["Open"])
        profit = qty * (row["Close"] - row["Open"])
        orders.append({
            "Ticker":    row["Ticker"],
            "Qty":       qty,
            "BuyPrice":  row["Open"],
            "SellPrice": row["Close"],
            "Profit":    round(profit, 2)
        })
        cash -= qty * row["Open"]

    total_profit = round(sum(o["Profit"] for o in orders), 2)
    return total_profit, pd.DataFrame(orders)

# ── MAIN EXECUTION ──
tickers = fetch_sp500()
print(f"Scanning {len(tickers)} S&P 500 tickers intraday ({INTERVAL})...")

signals = scan_intraday_breakouts(tickers)

# Derive scan timestamp dynamically
if signals.empty:
    scan_label = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
else:
    # use the BarTime of the first signal (all share same bar)
    scan_label = signals["BarTime"].iloc[0].strftime("%Y-%m-%d %H:%M")

print(f"\n📊 Intraday Breakouts at {scan_label}:\n")
print(signals if not signals.empty else "None")

if not signals.empty:
    profit, alloc_df = allocate_budget(signals, BUDGET_EUR)
    print(f"\n💰 Expected intraday profit with €{BUDGET_EUR:.2f} budget: €{profit:.2f}")
    print(alloc_df)

    if save_results:
        date_part = scan_label.replace(":", "-").replace(" ", "_")
        signals.to_csv(f"intraday_signals_{date_part}.csv", index=False)
        alloc_df.to_csv(f"intraday_alloc_{date_part}.csv", index=False)
        print("✅ Saved CSVs")

    if send_email_alerts:
        body = signals.to_string(index=False) + f"\n\nExpected profit: €{profit:.2f}"
        send_email(f"📈 Intraday Breakouts {scan_label}", body)
        print(f"✅ Email sent to {TO_EMAIL}")
