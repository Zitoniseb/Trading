import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import math

# ✅ Optional: hide warning about auto_adjust
warnings.simplefilter("ignore", category=FutureWarning)

# ✅ Tickers to analyze
tickers = ['AAPL', 'TSLA', 'AMD', 'MSFT', 'NVDA']

# ✅ Date range: last 14 weeks (~70 trading days)
end_date = datetime.now()
start_date = end_date - timedelta(weeks=14)

results = []

for ticker in tickers:
    try:
        # 🔄 Download data with auto_adjust=False
        df = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=False
        )

        if df is not None and not df.empty and len(df) >= 2:
            avg_volume = float(df['Volume'].mean())
            last_two = df['Volume'].astype(float).tail(2).values
            today_volume = last_two[-1].item() if hasattr(last_two[-1], "item") else float(last_two[-1])
            yesterday_volume = last_two[-2].item() if hasattr(last_two[-2], "item") else float(last_two[-2])
        else:
            avg_volume = 0.0
            today_volume = 0.0
            yesterday_volume = 0.0

        if any(math.isnan(x) for x in [avg_volume, today_volume, yesterday_volume]):
            print(f"❌ Skipped {ticker}: NaN detected in volume data")
            continue

        length = len(df) if df is not None else 0  # or handle the None case as needed

        # Note: We require at least 60 US market opening (trading) days, not just calendar days.
        if length < 60:
            print(f"\n⏭ Skipping {ticker}: Not enough data ({length} rows)")
            continue

        # ✅ Volume analysis (NO reassignment here!)
        explosion = bool(today_volume >= 4 * avg_volume or yesterday_volume >= 4 * avg_volume)
        strong_surge = bool(today_volume >= 2 * avg_volume or yesterday_volume >= 2 * avg_volume)
        recent_surge = bool(today_volume >= 1.5 * avg_volume or yesterday_volume >= 1.5 * avg_volume)
        stable_liquidity = bool(avg_volume >= 1_000_000)

        # 🧠 Debug info
        print(f"\n📊 {ticker}")
        print(f"12W Avg Volume: {avg_volume:.0f}")
        print(f"Yesterday: {yesterday_volume}, Today: {today_volume}")
        print(f"✔ Explosion (4x): {explosion}")
        print(f"✔ Strong Surge (2x): {strong_surge}")
        print(f"✔ Recent Surge (1.5x): {recent_surge}")
        print(f"✔ Stable Liquidity (1M+): {stable_liquidity}")

        if not stable_liquidity:
            print(f"❌ Skipped: Avg volume < 1M")
            continue

        if explosion:
            priority = 1
        elif strong_surge:
            priority = 2
        elif recent_surge:
            priority = 3
        else:
            print(f"❌ Skipped: No volume surge today/yesterday")
            continue

        # ✅ Store results
        results.append({
            'Ticker': ticker,
            'Avg_Volume_12W': int(avg_volume),
            'Yesterday_Volume': int(yesterday_volume),
            'Today_Volume': int(today_volume),
            'Explosion (4x)': explosion,
            'Recent Surge (1.5x)': recent_surge,
            'Strong Surge (2x)': strong_surge,
            'Priority': priority,
            'Bold': strong_surge
        })

    except Exception as e:
        print(f"⚠️ Error with {ticker}: {e}")
        continue

# ✅ Handle empty results
if not results:
    print("\n❌ No stocks passed all conditions.")
    exit()

# ✅ Final DataFrame
df = pd.DataFrame(results)
df_sorted = df.sort_values(by='Priority')

# ✅ Display results
print("\n✅ Final Sorted Results:\n")
print(df_sorted)
