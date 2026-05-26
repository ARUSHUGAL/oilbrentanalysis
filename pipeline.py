# ============================================================
# OIL PRICE TIME SERIES DASHBOARD
# pipeline.py
# ============================================================


# --- IMPORTS ---
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --- DATA LOADING ---
print("Downloading Brent Crude data...")

raw = yf.download("BZ=F", start="2014-01-01", end="2024-01-01")

# Keep only the closing price, rename for clarity
df = raw[["Close"]].copy()
df.columns = ["Price"]
df.index.name = "Date"

print(f"Downloaded {len(df)} rows")
print(df.head())


# --- CLEANING & VALIDATION ---
print("\nCleaning data...")

# Drop missing values
before = len(df)
df = df.dropna()
after = len(df)
print(f"Dropped {before - after} rows with missing values")

# Sort by date
df = df.sort_index()

# Anomaly detection: flag prices more than 3 std devs from the mean
mean = df["Price"].mean()
std  = df["Price"].std()
df["anomaly"] = (df["Price"] - mean).abs() > 3 * std

anomalies = df[df["anomaly"]]
print(f"Found {len(anomalies)} anomalous price points:")
print(anomalies[["Price"]])


# --- ROLLING STATS (risk & volatility) ---
df["rolling_avg_90"] = df["Price"].rolling(90).mean()   # 90-day moving average
df["volatility_30"]  = df["Price"].rolling(30).std()    # 30-day rolling volatility


# --- ANNOTATIONS: major global events ---
events = {
    "2014-11-28": "OPEC No-Cut Decision",
    "2016-02-11": "Price Hits 12-Year Low",
    "2020-03-09": "COVID + Saudi-Russia Price War",
    "2022-03-08": "Ukraine War Spike",
}


# --- VISUALIZATION ---
print("\nBuilding dashboard...")

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 9),
    gridspec_kw={"height_ratios": [3, 1]},
    facecolor="#0f0f0f"
)
fig.subplots_adjust(hspace=0.08)

# --- TOP CHART: Price + Rolling Avg ---
ax1.set_facecolor("#0f0f0f")
ax1.plot(df.index, df["Price"], color="#c8a951", linewidth=1.2, label="Brent Crude (USD/bbl)", zorder=3)
ax1.fill_between(df.index, df["Price"], alpha=0.12, color="#c8a951")
ax1.plot(df.index, df["rolling_avg_90"], color="#ffffff", linewidth=1, linestyle="--", alpha=0.6, label="90-day Moving Avg")

# Plot anomalies
ax1.scatter(
    anomalies.index, anomalies["Price"],
    color="red", zorder=5, s=30, label="Anomaly"
)

# Annotate events
for date_str, label in events.items():
    date = pd.to_datetime(date_str)
    if date in df.index:
        price = df.loc[date, "Price"]
    else:
        # find nearest date
        nearest = df.index[df.index.get_indexer([date], method="nearest")[0]]
        price = df.loc[nearest, "Price"]
    ax1.axvline(date, color="#ff4444", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.text(date, price + 3, label, fontsize=7, rotation=40,
             color="#ff8888", ha="left", va="bottom")

ax1.set_title("Brent Crude Oil Price  |  2014–2024", fontsize=15, color="white",
              fontfamily="monospace", pad=14)
ax1.set_ylabel("USD per Barrel", color="#aaaaaa", fontsize=10)
ax1.tick_params(colors="#aaaaaa", labelbottom=False)
ax1.spines[["top","right","bottom","left"]].set_color("#333333")
ax1.yaxis.grid(True, color="#222222", linewidth=0.6)
ax1.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=8)

# --- BOTTOM CHART: Volatility ---
ax2.set_facecolor("#0f0f0f")
ax2.fill_between(df.index, df["volatility_30"], color="#5588ff", alpha=0.5, label="30-day Volatility")
ax2.plot(df.index, df["volatility_30"], color="#5588ff", linewidth=0.8)

ax2.set_ylabel("Volatility (σ)", color="#aaaaaa", fontsize=10)
ax2.set_xlabel("Year", color="#aaaaaa", fontsize=10)
ax2.tick_params(colors="#aaaaaa")
ax2.spines[["top","right","bottom","left"]].set_color("#333333")
ax2.yaxis.grid(True, color="#222222", linewidth=0.6)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=8)

plt.savefig("oil_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("\nSaved: oil_dashboard.png")
plt.show()
