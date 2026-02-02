# %%
from bs4 import BeautifulSoup, Tag
import requests
import numpy as np
import pandas as pd
from matplotlib import lines, pyplot as plt
import darkmode_orange
from scipy.stats import linregress
from matplotlib.markers import MarkerStyle
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from pathlib import Path
from datetime import datetime
import json
pd.options.display.float_format = '{:,.0f}'.format

# Output directory for GitHub Pages
OUTPUT_DIR = Path("docs")
OUTPUT_DIR.mkdir(exist_ok=True)

def display(df):
    """Print DataFrame nicely."""
    print(df.to_string())

# %%
"""
extract cpu data from each row in the table
example HTML:

<div class="chart_subheader">
    <div class="chart_tabletitle1">CPU</div> <div class="chart_tabletitle2">Avarage CPU Mark</div> <div class="chart_tabletitle3">Price (USD)</div> <div class="chart_tabletitle4">First Seen</div>
</div>
<div class="chart_body">
    <ul class="chartlist">
        <li id="rk5573">
            <span class="more_details" onclick="x( event, 87, 7, 6, 2, 'NA');"></span> <a href="cpu.php?cpu=AMD+Ryzen+5+5600X3D&amp;id=5573"> <span class="prdname">AMD Ryzen 5 5600X3D</span>
            <div><span class="index orange" style="width: 19%">(19%)</span></div> <span class="count">22,035</span>
            <span class="price-neww">NA</span> <span class="first-seen">Q2 2023</span> </a> </li>
        <li id="rk5533">
            <span class="more_details" onclick="x( event, 17, 17, 16, 1, 'NA');"></span> <a href="cpu.php?cpu=Apple+M2+Ultra+24+Core&amp;id=5533"> <span class="prdname">Apple M2 Ultra 24 Core</span>
            <div><span class="index orange" style="width: 42%">(42%)</span></div> <span class="count">48,279</span>
            <span class="price-neww">NA</span> <span class="first-seen">Q2 2023</span> </a> </li>
        <li id="rk5490">
    </ul>
</div>
"""
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
response = requests.get("https://www.cpubenchmark.net/desktop.html#cpumark", headers=headers)
page = BeautifulSoup(response.content, "html.parser")
chart_body = page.find("div", class_="chart_body")
assert isinstance(chart_body, Tag)
cpu_list = []
for li in chart_body.find_all("li"):
    spans = li.find_all("span")
    name = li.find("span", class_="prdname").text
    splits = name.split(" ")
    brand = splits[0]
    model = ' '.join(splits[1:])
    cpu_list.append({
        "brand": brand,
        "model": model,
        "score": li.find("span", class_="count").text.replace(",", ""),
        "price": li.find("span", class_="price-neww").text
    })

# %% make df
cpu = pd.DataFrame(cpu_list)
cpu.score = cpu.score.astype(int)
# convert from currency format: $5,699.99
cpu.price = cpu.price.str.replace(",", "")
cpu.price = cpu.price.str.replace("$", "")
cpu.price = cpu.price.str.replace("*", "")
# replace "NA"
cpu.price = cpu.price.str.replace("NA", "")
# ValueError: could not convert string to float: ''
# set empty string to NaN
cpu.price = cpu.price.replace("", np.nan)
cpu.price = cpu.price.astype(float)
cpu = cpu.dropna(subset=['price', 'score'])
cpu["value"] = cpu.score / cpu.price

# %% based on a histogram of price, remove the outliers
nbins = 20
hist_values, bins = np.histogram(cpu["price"], bins=nbins)
i = np.argmin(hist_values)
print(f"first empty bin {bins[i]:,.0f}-{bins[i+1]:,.0f}")
print(f"keeping values below {bins[i]:,.0f}")
plt.hist(cpu["price"], bins=nbins)
plt.savefig(OUTPUT_DIR / "price_histogram.png", dpi=150, bbox_inches='tight')
plt.close()

# %%
# Remove outliers based on the index i
cpu = cpu.where(cpu["price"] <= bins[i])
# remove NaN values
cpu = cpu.dropna(subset=['price', 'score'])
print(f"new price range is {cpu['price'].min():,.0f}-{cpu['price'].max():,.0f}")

# %%
def prettyplot(df):
    scatter = plt.scatter(df["price"], df["score"], c=df["value"], cmap="RdYlGn")
    plt.colorbar(scatter)
    plt.draw()  # Draw the plot to force color assignment
    scatter.set_edgecolors(scatter.get_facecolor())  # Set edgecolors to be the same as facecolors
    scatter.set_facecolors('none')

def addexp(df):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df['price'], df['score'])
    # plot the regression line
    df["exp"] = slope * df["price"] + intercept

# %%
n=len(cpu)
iteration = 0
while n > 50:
    fig = plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    prettyplot(cpu)
    # drop everything below the regression line
    addexp(cpu)
    below_the_line = cpu["score"] < cpu["exp"]
    cpu = cpu.loc[~below_the_line,:]
    plt.plot(cpu["price"], cpu["exp"], "r-", label="Regression Line")
    plt.title(f"cutoff regression n={n}")

    plt.subplot(1, 2, 2)
    addexp(cpu)
    prettyplot(cpu)
    plt.plot(cpu["price"], cpu["exp"], "r-", label="Regression Line");
    plt.title(f"new regression n={len(cpu)}")

    plt.savefig(OUTPUT_DIR / f"regression_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close()
    iteration += 1
    n = len(cpu)

# %% show df by value
cpu = cpu.sort_values("value", ascending=False)
top10 = cpu.head(10)
display(top10)

# %%
print("total")
cpug = cpu.groupby("brand").agg({"brand": "count", "score": "mean", "price": "mean", "value": "mean"})
display(cpug)

print("top10")
cpug = top10.groupby("brand").agg({"brand": "count", "score": "mean", "price": "mean", "value": "mean"})
display(cpug)

# %%
fig = plt.figure(figsize=(8, 4))
amd = cpu.brand == "AMD"
intel = cpu.brand == "Intel"
plt.scatter(cpu[amd]["price"], cpu[amd]["score"], facecolor="none", edgecolor="red")
plt.scatter(cpu[intel]["price"], cpu[intel]["score"], facecolor="none", edgecolor="blue")

# KNN model
knn = NearestNeighbors(n_neighbors=5)
data = cpu[['price', 'score']].values
knn.fit(data)
# KMeans clustering to determine groups
# Assuming the number of clusters to be 5 as we are choosing 5 nearest neighbors in KNN
kmeans = KMeans(n_clusters=5, random_state=0).fit(data)

# Assigning different colors to different clusters
colors = ['red', 'blue', 'green', 'yellow', 'purple']

for i, center in enumerate(kmeans.cluster_centers_):
    plt.scatter(*center, s=300, marker=MarkerStyle('*'), c=colors[i], alpha=0.5)

# Adding the clusters to the scatter plot
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], color=colors[kmeans.labels_[i]])

# Calculate and plot the efficient frontier (Pareto frontier)
# A CPU is Pareto-optimal if no other CPU has both higher score AND lower price
cpu_sorted = cpu.sort_values('price')
frontier_prices = []
frontier_scores = []
max_score = 0
for _, row in cpu_sorted.iterrows():
    if row['score'] > max_score:
        frontier_prices.append(row['price'])
        frontier_scores.append(row['score'])
        max_score = row['score']

plt.plot(frontier_prices, frontier_scores, 'w-', linewidth=2, label='Efficient Frontier')
plt.scatter(frontier_prices, frontier_scores, c='white', s=50, zorder=5, edgecolors='black')
plt.xlabel('Price ($)')
plt.ylabel('Score')
plt.legend()

plt.savefig(OUTPUT_DIR / "clusters.png", dpi=150, bbox_inches='tight')
plt.close()

# pick the highest value item from each cluster
# highest_value = [np.argmax(cpu[cpu.brand == brand].value) for brand in cpu.brand.unique()]

# %% Calculate Value Above Replacement (VAR) for each brand
def get_frontier(df):
    """Calculate efficient frontier for a dataframe of CPUs."""
    sorted_df = df.sort_values('price')
    prices, scores = [], []
    max_score = 0
    for _, row in sorted_df.iterrows():
        if row['score'] > max_score:
            prices.append(row['price'])
            scores.append(row['score'])
            max_score = row['score']
    return prices, scores

def calc_auc_above_regression(frontier_prices, frontier_scores, slope, intercept, price_min, price_max):
    """Calculate area between frontier and regression line using trapezoidal integration."""
    if len(frontier_prices) < 2:
        return 0.0

    # Filter frontier points to price range
    points = [(p, s) for p, s in zip(frontier_prices, frontier_scores) if price_min <= p <= price_max]
    if len(points) < 2:
        return 0.0

    prices = [p for p, s in points]
    scores = [s for p, s in points]

    # Calculate area using trapezoidal rule
    # Area = sum of (width * avg_height_above_regression)
    total_area = 0.0
    for i in range(len(prices) - 1):
        p1, p2 = prices[i], prices[i + 1]
        s1, s2 = scores[i], scores[i + 1]
        # Regression values at these prices
        r1 = slope * p1 + intercept
        r2 = slope * p2 + intercept
        # Height above regression at each point
        h1 = s1 - r1
        h2 = s2 - r2
        # Trapezoidal area
        width = p2 - p1
        avg_height = (h1 + h2) / 2
        total_area += width * avg_height

    return total_area

# Fit regression to all CPUs (this is the "replacement level")
slope, intercept, _, _, _ = linregress(cpu['price'], cpu['score'])
print(f"Regression: score = {slope:.2f} * price + {intercept:.2f}")

# Get price range
price_min, price_max = cpu['price'].min(), cpu['price'].max()

# Calculate brand-specific frontiers and VAR
var_results = {}
brand_frontiers = {}
for brand in ['AMD', 'Intel']:
    brand_df = cpu[cpu.brand == brand]
    if len(brand_df) > 0:
        prices, scores = get_frontier(brand_df)
        brand_frontiers[brand] = {'prices': prices, 'scores': scores}
        var = calc_auc_above_regression(prices, scores, slope, intercept, price_min, price_max)
        var_results[brand] = var
        print(f"{brand} VAR: {var:,.0f}")

# %% Plot brand frontiers with regression line
fig = plt.figure(figsize=(10, 6))

# Plot all CPUs by brand
amd_df = cpu[cpu.brand == 'AMD']
intel_df = cpu[cpu.brand == 'Intel']
plt.scatter(amd_df['price'], amd_df['score'], alpha=0.3, c='red', label='AMD CPUs')
plt.scatter(intel_df['price'], intel_df['score'], alpha=0.3, c='blue', label='Intel CPUs')

# Plot regression line
reg_x = np.array([price_min, price_max])
reg_y = slope * reg_x + intercept
plt.plot(reg_x, reg_y, 'w--', linewidth=2, label=f'Regression (baseline)')

# Plot brand frontiers
if 'AMD' in brand_frontiers:
    plt.plot(brand_frontiers['AMD']['prices'], brand_frontiers['AMD']['scores'], 'r-', linewidth=2, label=f'AMD Frontier (VAR: {var_results["AMD"]:,.0f})')
    plt.scatter(brand_frontiers['AMD']['prices'], brand_frontiers['AMD']['scores'], c='red', s=60, zorder=5, edgecolors='white')
if 'Intel' in brand_frontiers:
    plt.plot(brand_frontiers['Intel']['prices'], brand_frontiers['Intel']['scores'], 'b-', linewidth=2, label=f'Intel Frontier (VAR: {var_results["Intel"]:,.0f})')
    plt.scatter(brand_frontiers['Intel']['prices'], brand_frontiers['Intel']['scores'], c='blue', s=60, zorder=5, edgecolors='white')

plt.xlabel('Price ($)')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.savefig(OUTPUT_DIR / "var_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# %% Update history timeseries
today = datetime.now().strftime("%Y-%m-%d")
history_file = OUTPUT_DIR / "history.json"
if history_file.exists():
    history = json.loads(history_file.read_text())
else:
    history = []

# Only add if today's date isn't already in history
if not any(h['date'] == today for h in history):
    history.append({
        'date': today,
        'amd_var': var_results.get('AMD', 0),
        'intel_var': var_results.get('Intel', 0),
    })
    history_file.write_text(json.dumps(history, indent=2))
    print(f"History updated: {len(history)} entries")
else:
    # Update today's entry
    for h in history:
        if h['date'] == today:
            h['amd_var'] = var_results.get('AMD', 0)
            h['intel_var'] = var_results.get('Intel', 0)
    history_file.write_text(json.dumps(history, indent=2))
    print(f"History entry for {today} updated")

# %% Generate HTML report
today = datetime.now().strftime("%Y-%m-%d")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CPU Price/Performance Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a1a; color: #e0e0e0; }}
        h1, h2 {{ color: #ff9800; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
        th {{ background: #333; color: #ff9800; }}
        tr:nth-child(even) {{ background: #252525; }}
        tr:hover {{ background: #333; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; }}
        .updated {{ color: #888; font-size: 0.9em; }}
        a {{ color: #ff9800; }}
    </style>
</head>
<body>
    <h1>CPU Price/Performance Analysis</h1>
    <p class="updated">Last updated: {today}</p>

    <h2>Top 10 Best Value CPUs</h2>
    <table>
        <tr><th>Brand</th><th>Model</th><th>Score</th><th>Price</th><th>Value</th></tr>
"""

for _, row in top10.iterrows():
    html += f"        <tr><td>{row['brand']}</td><td>{row['model']}</td><td>{row['score']:,.0f}</td><td>${row['price']:,.0f}</td><td>{row['value']:.2f}</td></tr>\n"

html += """    </table>

    <h2>Brand Summary (Best Value CPUs)</h2>
    <table>
        <tr><th>Brand</th><th>Count</th><th>Avg Score</th><th>Avg Price</th><th>Avg Value</th></tr>
"""

cpug = cpu.groupby("brand").agg({"brand": "count", "score": "mean", "price": "mean", "value": "mean"})
for brand, row in cpug.iterrows():
    html += f"        <tr><td>{brand}</td><td>{row['brand']:.0f}</td><td>{row['score']:,.0f}</td><td>${row['price']:,.0f}</td><td>{row['value']:.2f}</td></tr>\n"

html += """    </table>

    <h2>Price Distribution</h2>
    <img src="price_histogram.png" alt="Price Histogram">

    <h2>Price vs Performance</h2>
    <p>CPUs clustered by price/performance. The white line shows the efficient frontier — Pareto-optimal CPUs where you can't get better performance without paying more.</p>
    <img src="clusters.png" alt="CPU Clusters">

    <p>Data source: <a href="https://www.cpubenchmark.net/desktop.html">PassMark CPU Benchmark</a></p>
</body>
</html>
"""

(OUTPUT_DIR / "index.html").write_text(html)
print(f"Report generated: {OUTPUT_DIR / 'index.html'}")

# %% Generate JSON for client-side rendering
brands_data = [{"brand": brand, "count": int(row["brand"]), "score": row["score"], "price": row["price"], "value": row["value"]} for brand, row in cpug.iterrows()]
data = {
    "updated": today,
    "top10": top10[["brand", "model", "score", "price", "value"]].to_dict(orient="records"),
    "brands": brands_data,
    "frontier": [{"price": p, "score": s} for p, s in zip(frontier_prices, frontier_scores)],
    "var": var_results,
    "brand_frontiers": {brand: [{"price": p, "score": s} for p, s in zip(f['prices'], f['scores'])] for brand, f in brand_frontiers.items()},
    "regression": {"slope": slope, "intercept": intercept},
    "history": history,
}
(OUTPUT_DIR / "data.json").write_text(json.dumps(data, indent=2))
print(f"JSON generated: {OUTPUT_DIR / 'data.json'}")

# %%