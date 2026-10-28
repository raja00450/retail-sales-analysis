"""
Real-world Data Project: Retail Sales Analysis (end-to-end)
-----------------------------------------------------------
Steps: Load -> Clean -> EDA -> Visualize -> Predict Profit -> Conclusions

Data:
  * If a real Superstore CSV is placed at data/superstore.csv (Kaggle
    "Superstore Sales"), it is used automatically.
  * Otherwise a sample retail dataset is generated so the script always runs.

Run:  python retail_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ 1. LOAD
def make_sample_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    cats = {
        "Technology": ["Phones", "Accessories", "Copiers", "Machines"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Paper", "Storage", "Art", "Labels"],
    }
    base_price = {"Technology": 300, "Furniture": 250, "Office Supplies": 40}
    base_margin = {"Technology": 0.22, "Furniture": 0.08, "Office Supplies": 0.18}
    rows = []
    dates = pd.date_range("2022-01-01", "2025-12-31")
    for _ in range(n):
        cat = rng.choice(list(cats), p=[0.2, 0.25, 0.55])
        sub = rng.choice(cats[cat])
        d = rng.choice(dates)
        month = pd.Timestamp(d).month
        season = 1.35 if month in (11, 12) else 1.0
        qty = int(rng.integers(1, 10))
        discount = float(rng.choice([0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        price = base_price[cat] * rng.lognormal(0, 0.5)
        sales = price * qty * (1 - discount) * season
        margin = base_margin[cat] - 0.55 * discount + rng.normal(0, 0.05)
        rows.append([
            d, cat, sub, rng.choice(["West", "East", "Central", "South"]),
            rng.choice(["Consumer", "Corporate", "Home Office"], p=[0.5, 0.3, 0.2]),
            round(sales, 2), qty, discount, round(sales * margin, 2),
        ])
    df = pd.DataFrame(rows, columns=[
        "Order Date", "Category", "Sub-Category", "Region",
        "Segment", "Sales", "Quantity", "Discount", "Profit"])
    # Add realistic dirt: missing values and duplicates
    df.loc[rng.choice(n, 60, replace=False), "Sales"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "Region"] = np.nan
    df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)
    return df


real_path = os.path.join(DATA_DIR, "superstore.csv")
if os.path.exists(real_path):
    df = pd.read_csv(real_path, encoding="latin-1")
    source = "Real dataset: data/superstore.csv"
else:
    df = make_sample_data()
    df.to_csv(os.path.join(DATA_DIR, "sales_data.csv"), index=False)
    source = "Sample retail dataset (generated)"
print("DATA SOURCE:", source)
print("Raw shape:", df.shape)

# ----------------------------------------------------------------- 2. CLEAN
df.columns = [c.strip().replace("_", " ").title().replace("Sub Category", "Sub-Category")
              for c in df.columns]
df = df.rename(columns={"Sub-Category": "Sub-Category"})

raw_rows = len(df)
dups = int(df.duplicated().sum())
df = df.drop_duplicates()

missing_before = df.isna().sum()
missing_total = int(missing_before.sum())
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df = df.dropna(subset=["Order Date"])

# Outliers: IQR rule, values are capped (winsorized), not deleted
outlier_counts = {}
for col in ["Sales", "Profit"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    df[col] = df[col].clip(lo, hi)

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Profit Margin %"] = (df["Profit"] / df["Sales"] * 100).replace([np.inf, -np.inf], 0)

print(f"Duplicates removed: {dups} | Missing values filled: {missing_total}")
print("Outliers capped:", outlier_counts)
print("Clean shape:", df.shape)

# ------------------------------------------------------------------ 3. EDA
print("\nSTATISTICAL SUMMARY\n", df[["Sales", "Quantity", "Discount", "Profit"]].describe().round(2))

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
cat_profit = df.groupby("Category")[["Sales", "Profit"]].sum().sort_values("Profit", ascending=False)
sub_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
monthly = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
monthly["Date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(Day=1))
disc_profit = df.groupby("Discount")["Profit Margin %"].mean()
corr = df[["Sales", "Quantity", "Discount", "Profit"]].corr()

# ------------------------------------------------------------ 4. VISUALIZE
def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=130)
    plt.close()

plt.figure(figsize=(9, 4))
plt.plot(monthly["Date"], monthly["Sales"], marker="o", color="#2563eb")
plt.title("Monthly Sales Trend"); plt.ylabel("Sales"); plt.xlabel("")
save("1_monthly_sales_trend.png")

plt.figure(figsize=(7, 4))
cat_profit[["Sales", "Profit"]].plot(kind="bar", ax=plt.gca(), color=["#60a5fa", "#16a34a"])
plt.title("Sales vs Profit by Category"); plt.xticks(rotation=0)
save("2_category_sales_profit.png")

plt.figure(figsize=(8, 5))
colors = ["#dc2626" if v < 0 else "#16a34a" for v in sub_profit.values]
sub_profit.plot(kind="barh", color=colors)
plt.title("Profit by Sub-Category"); plt.xlabel("Total Profit")
save("3_subcategory_profit.png")

plt.figure(figsize=(6, 4))
sns.barplot(x=region_sales.index, y=region_sales.values, color="#2563eb")
plt.title("Sales by Region"); plt.ylabel("Sales")
save("4_region_sales.png")

plt.figure(figsize=(7, 4))
sns.lineplot(x=disc_profit.index * 100, y=disc_profit.values, marker="o", color="#dc2626")
plt.axhline(0, color="gray", ls="--")
plt.title("Discount vs Average Profit Margin"); plt.xlabel("Discount %"); plt.ylabel("Profit Margin %")
save("5_discount_vs_margin.png")

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
save("6_correlation_heatmap.png")

# ------------------------------------------------------------ 5. PREDICTION
features = ["Sales", "Quantity", "Discount", "Category", "Region", "Segment", "Month"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Profit"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_tr, y_tr)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
res = {}
for name, m in [("Linear Regression", lr), ("Random Forest", rf)]:
    p = m.predict(X_te)
    res[name] = (r2_score(y_te, p), mean_absolute_error(y_te, p))
    print(f"{name}: R2={res[name][0]:.3f}  MAE={res[name][1]:.2f}")

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(8)
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="#7c3aed")
plt.title("Key Drivers of Profit (Random Forest)")
save("7_feature_importance.png")

plt.figure(figsize=(5.5, 5))
plt.scatter(y_te, rf.predict(X_te), alpha=0.3, color="#2563eb")
lim = [y_te.min(), y_te.max()]
plt.plot(lim, lim, "r--"); plt.xlabel("Actual Profit"); plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
save("8_actual_vs_predicted.png")

# ------------------------------------------------------------ 6. CONCLUSIONS
best_cat = cat_profit.index[0]
worst_sub = sub_profit.index[0]
top_region = region_sales.index[0]
peak_month = df.groupby("Month")["Sales"].sum().idxmax()
top_driver = imp.index[-1]
corr_dp = corr.loc["Discount", "Profit"]
best_model = max(res, key=lambda k: res[k][0])

report = f"""# Retail Sales Analysis

**Data source:** {source}

## Data Cleaning
- Rows: {raw_rows} -> {len(df)} after cleaning
- Duplicates removed: {dups}
- Missing values filled (median for numbers, mode for text): {missing_total}
- Outliers capped using IQR rule: {outlier_counts}

## Key Numbers
- Total Sales: {total_sales:,.0f}
- Total Profit: {total_profit:,.0f} (margin {total_profit / total_sales * 100:.1f}%)

## Insights
1. **{best_cat}** is the most profitable category.
2. **{worst_sub}** has the lowest profit among sub-categories.
3. **{top_region}** is the top region by sales.
4. Sales peak in month **{peak_month}** (seasonal demand).
5. Discount and Profit correlation is **{corr_dp:.2f}**: higher discounts reduce profit.
6. Most important profit driver in the model: **{top_driver}**.

## Prediction Model (predicting Profit)
| Model | R2 | MAE |
|---|---|---|
""" + "\n".join(f"| {k} | {v[0]:.3f} | {v[1]:.2f} |" for k, v in res.items()) + f"""

Best model: **{best_model}**.

## Conclusion
- Limit heavy discounts (above 30%), because they push margins negative.
- Focus marketing and stock on the most profitable category and peak months.
- Review or reprice the weakest sub-category.

## Charts
![](outputs/1_monthly_sales_trend.png)
![](outputs/2_category_sales_profit.png)
![](outputs/3_subcategory_profit.png)
![](outputs/4_region_sales.png)
![](outputs/5_discount_vs_margin.png)
![](outputs/6_correlation_heatmap.png)
![](outputs/7_feature_importance.png)
![](outputs/8_actual_vs_predicted.png)

## How to run
```
pip install pandas matplotlib seaborn scikit-learn
python retail_analysis.py
```
"""
with open(os.path.join(BASE, "README.md"), "w", encoding="utf-8") as f:
    f.write(report)
print("\nDone. README.md and charts saved in outputs/")
