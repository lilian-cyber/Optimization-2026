# %%
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# %%
tax_path = "/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Problem Statement, Dataset, and Rubric/1. DatathonMasterStateTaxData_2004_2025Q2.csv"
econ_path = "/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Problem Statement, Dataset, and Rubric/2. DatathonMasterEconomicDataset_2004_2025Q2.csv"
tax = pd.read_csv(tax_path)
econ = pd.read_csv(econ_path)
print(list(tax.columns))
print(list(econ.columns))
print(tax["Tax_Category"].unique())
# %%
# data cleaning
rename_map = {
    "Individual income taxes":                "Individual income",
    "Corporation net income taxes":           "Corporation net income",
    "Other taxes, NEC":                       "Other selective sales and gross receipts",
    "Property tax":                           "Property taxes",
    "Insurance premiums":                     "Insurance",
    "Motor fuels":                            "Motor fuel sales taxes",
    "Motor fuel sales":                       "Motor fuel sales taxes",
    "Death & gift taxes":                     "Death and gift taxes",
    "Death and gift":                         "Death and gift taxes",
    "Alcoholic beverage license":             "Alcoholic beverages",
    "Hunting and fishing":                    "Hunting and fishing licenses",
    "Hunting & fishing licenses":             "Hunting and fishing licenses",
    "Occupation and businesses":              "Occupation and business licenses",
    "Occupation & business licenses":         "Occupation and business licenses",
    "Other license taxes":                    "Other licenses taxes",
    "Documentary and stock transfer":         "Documentary and stock transfer taxes",
    "Documentary & stock transfer taxes":     "Documentary and stock transfer taxes",
    "Severance":                              "Severance taxes",
    "Motor vehicle operators":                "Motor vehicle operator",
}
tax["Tax_Category"] = tax["Tax_Category"].replace(rename_map)
mi_ind = tax[(tax["State"] == "Michigan") &
             (tax["Tax_Category"] == "Individual income")]
print(sorted(mi_ind['Year'].unique()))
# %%
def make_period(year, quarter):
    return year + (quarter - 1) / 4
tax["period"]  = tax.apply(lambda r: make_period(r["Year"], r["Quarter"]), axis=1)
econ["period"] = econ.apply(lambda r: make_period(r["Year"], r["Quarter"]), axis=1)
# the definition for recession period
RECESSION_PERIODS = [
    (2001, 1, 2001, 4),
    (2007, 4, 2009, 2),
    (2020, 1, 2020, 2),
]
def is_recession(year, quarter):
    p = make_period(year, quarter)
    for (sy, sq, ey, eq) in RECESSION_PERIODS:
        if make_period(sy, sq) <= p <= make_period(ey, eq):
            return True
    return False
tax["is_recession"] = tax.apply(lambda r: is_recession(r["Year"], r["Quarter"]), axis=1)
econ["is_recession"] = econ.apply(lambda r: is_recession(r["Year"], r["Quarter"]), axis=1)
tax.head()
# %%
# Many states suffered great losses from the 2007-2009 recession, but Michigan was hit particularly hard.
# We find a way to quanrify the severity - recession severity index. 1) score GDP 2)Score unemployment rate change 3) Scrore unemployment rate peak level
# severe recession (2007Q4 - 2009Q2) real change
SEVERE_GDP_DROP   = -4.2
SEVERE_UR_CHANGE  =  5.1
SEVERE_UR_LEVEL   =  9.9

# mild recession (1990-91 & 2001 these two events) average
MILD_GDP_DROP     = -0.6
MILD_UR_CHANGE    =  1.9
MILD_UR_LEVEL     =  6.9

# build the recession severity index
def score_gdp(gdp_pct_change):
    return -26.1 * gdp_pct_change - 10.8
def score_ur_change(delta_ur):
    return 31.8 * delta_ur - 61.9
def score_ur_level(max_ur):
    return 100 * (max_ur - 6.9) / (9.9 - 6.9)
def overall_real_activity_score(gdp_drop_pct, delta_ur, max_ur):
    s1 = score_gdp(gdp_drop_pct)
    s2 = score_ur_change(delta_ur)
    s3 = score_ur_level(max_ur)
    return (s1 + s2 + s3) / 3, s1, s2, s3
# %%
# for unemployment rate, we also include the post-recession period (hysteresis effect) to capture the peak level after the recession trough.
pre  = econ[econ["Year"].between(2005, 2007)]
dur  = econ[(econ["Year"].isin([2008,2009])) &
            (econ["period"] <= make_period(2009, 2))]
post = econ[econ["Year"].between(2008, 2011)]
gdp_pre    = pre.groupby("State")["GDP_Total"].mean()
gdp_trough = dur.groupby("State")["GDP_Total"].min()
gdp_change = ((gdp_trough - gdp_pre) / gdp_pre * 100)
ur_pre    = pre.groupby("State")["Unemployment_Rate"].mean()
ur_peak   = post.groupby("State")["Unemployment_Rate"].max()
delta_ur  = (ur_peak - ur_pre).clip(lower=0)
max_ur    = ur_peak
# %%
# calculate the recession severity index for each state from 2008.00 to 2009.25
states = gdp_change.index.intersection(delta_ur.index).intersection(max_ur.index)
results = []
for state in states:
    g  = gdp_change.get(state, np.nan)
    du = delta_ur.get(state, np.nan)
    mu = max_ur.get(state, np.nan)
    if any(pd.isna([g, du, mu])): continue
    overall, s1, s2, s3 = overall_real_activity_score(g, du, mu)
    results.append({
        "State": state,
        "GDP_change_pct": round(g, 2),
        "Delta_UR_pp": round(du, 2),
        "Max_UR_pct": round(mu, 2),
        "Score_GDP": round(s1, 1),
        "Score_UR_change": round(s2, 1),
        "Score_UR_level": round(s3, 1),
        "Overall_Score": round(overall, 1),
    })
df = pd.DataFrame(results).set_index("State").sort_values("Overall_Score", ascending=False)
# %%
# visualization for each state
BLUE = "#1B2A5E"; GOLD = "#FFCB05"; RED = "#C0392B"; GRAY = "#555555"; GREEN = "#27AE60"
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ax = axes[0]
colors = []
for state in df.index:
    if state == "Michigan":
        colors.append(RED)
    elif df.loc[state, "Overall_Score"] >= 100:
        colors.append(GOLD)
    elif df.loc[state, "Overall_Score"] >= 0:
        colors.append(BLUE)
    else:
        colors.append(GRAY)

ax.bar(range(len(df)), df["Overall_Score"].values, color=colors, width=0.8)
ax.axhline(100, color=RED, lw=1.5, ls="--", label="Great Recession = 100")
ax.axhline(0,   color=GRAY, lw=1,   ls=":",  label="Mild Recession avg = 0")

if "Michigan" in df.index:
    mi_i = list(df.index).index("Michigan")
    mi_v = df.loc["Michigan", "Overall_Score"]
    ax.annotate(f"Michigan\n{mi_v:.0f}",
                xy=(mi_i, mi_v), xytext=(mi_i+4, mi_v+15),
                fontsize=10, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

ax.set_xticks(range(len(df)))
ax.set_xticklabels(df.index, rotation=90, fontsize=6.5)
ax.set_ylabel("Severity Score\n(Great Recession = 100,  Mild Recession avg = 0)")
ax.set_title("Great Recession Severity by State\n(Durdu-Edge-Schwindt Method)",
             fontweight="bold", color=BLUE, fontsize=12)
ax.legend(fontsize=9)
ax = axes[1]
mi_row = df.loc["Michigan"]
components = ["Score_GDP", "Score_UR_change", "Score_UR_level"]
labels     = ["GDP Drop\n(Score_GDP)", "ΔUnemployment\n(Score_UR_change)", "Peak Unemp Level\n(Score_UR_level)"]
values     = [mi_row[c] for c in components]
bar_colors = [BLUE, BLUE, BLUE]

bars = ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white", linewidth=2)
ax.axhline(100, color=RED,  lw=1.5, ls="--", label="Great Recession = 100")
ax.axhline(0,   color=GRAY, lw=1,   ls=":",  label="Mild Recession avg = 0")
ax.axhline(mi_row["Overall_Score"], color=GOLD, lw=2.5, ls="-",
           label=f"Michigan Overall = {mi_row['Overall_Score']:.0f}")

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 3,
            f"{val:.1f}", ha="center", fontsize=12, fontweight="bold", color=BLUE)

ax.text(0, values[0]/2, f"GDP: {mi_row['GDP_change_pct']:.1f}%",
        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(1, values[1]/2, f"ΔUR: +{mi_row['Delta_UR_pp']:.1f}pp",
        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(2, values[2]/2, f"MaxUR: {mi_row['Max_UR_pct']:.1f}%",
        ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.set_ylim(0, max(values) * 1.3)
ax.set_title("Michigan: Score Decomposition\n(3 components averaged → Overall Score)",
             fontweight="bold", color=BLUE, fontsize=12)
ax.legend(fontsize=9)

plt.suptitle("Recession Severity Index — Durdu, Edge & Schwindt (2017) Fed Notes",
             fontsize=13, fontweight="bold", color=BLUE, y=1.01)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code' + "step3_severity_index_corrected.png", dpi=150, bbox_inches="tight")
plt.close()
# %%
# from this part, we mainly focus on the sensitivity of Michigan State tax revenue.
mi = tax[tax["State"] == "Michigan"]
mi_q = (mi.groupby(["Year", "Quarter", "period", "is_recession", "Tax_Category"],
                    as_index=False)["Amount"].sum())
# calculate the two indicators
pre_annual = (mi_q[mi_q['Year'].between(2005, 2007)]).groupby(["Year","Tax_Category"])["Amount"].sum().reset_index()
rec_annual = (mi_q[mi_q["Year"].between(2008, 2011)]
              .groupby(["Year","Tax_Category"])["Amount"].sum().reset_index())
pre_mean = pre_annual.groupby("Tax_Category")["Amount"].mean()
rec_mean = rec_annual.groupby("Tax_Category")["Amount"].mean()
drop_pct = (rec_mean - pre_mean) / pre_mean.abs() * 100
pre_size = pre_mean.abs()
impact   = (drop_pct.clip(upper=0).abs() / 100) * pre_size
sensitivity = pd.DataFrame({
    "Pre_size_$M":  pre_size / 1e3,
    "Drop_pct":     drop_pct,
    "Impact_$M":    impact / 1e3,
}).dropna()
sensitivity = sensitivity[sensitivity["Pre_size_$M"] > 0]
sensitivity = sensitivity.sort_values("Impact_$M", ascending=False)
TOP3 = sensitivity[sensitivity["Impact_$M"] > 0].head(3).index.tolist()
volatile_share = sensitivity.loc[TOP3, "Pre_size_$M"].sum() / sensitivity["Pre_size_$M"].sum() * 100
sensitivity.head(10)
print(volatile_share)
# %% Visualization for tax sensitivity, Drop% vs Pre_size scatter plot，Impact is the size of the bubble.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
for cat, row in sensitivity.iterrows():
    fell = row["Drop_pct"] < 0
    if cat in TOP3:
        color, zorder, fw = RED, 6, "bold"
    elif fell:
        color, zorder, fw = BLUE, 4, "normal"
    else:
        color, zorder, fw = GRAY, 2, "normal"

    ms = max(row["Impact_$M"] / sensitivity["Impact_$M"].max() * 2000, 30) if fell else 40
    marker = "o" if fell else "x"
    ax.scatter(row["Drop_pct"], row["Pre_size_$M"],
               color=color, s=ms, zorder=zorder, alpha=0.8,
               marker=marker,
               edgecolors="white" if fell else color, linewidth=1.2)

    if row["Pre_size_$M"] > 200 or cat in TOP3:
        dy = row["Pre_size_$M"] * 0.04
        if cat == "Individual income":                dy = -600
        if cat == "General sales and gross receipts": dy =  300
        ax.annotate(
            f"{cat}\n({row['Drop_pct']:.1f}%, ${row['Pre_size_$M']:.0f}M)",
            xy=(row["Drop_pct"], row["Pre_size_$M"]),
            xytext=(row["Drop_pct"] + 1.5, row["Pre_size_$M"] + dy),
            fontsize=8, color=color, fontweight=fw,
        )

ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.3)
ax.set_xlabel("Revenue Drop 2008–2011 (%)\n← fell more        rose →", fontsize=10)
ax.set_ylabel("Pre-Recession Annual Revenue ($M, 2005–2007 avg)", fontsize=10)
ax.set_title("Drop% vs Revenue Size\n"
             "Bubble size = Impact ($M lost)  |  ✕ = rose in recession",
             fontweight="bold", color=BLUE)
ax.legend(handles=[
    Patch(facecolor=RED,  label="Top3 (highest impact)"),
    Patch(facecolor=BLUE, label="Fell — lower impact"),
    Line2D([0],[0], marker="x", color=GRAY, linestyle="None",
           markersize=8, label="Rose in 2008-2011 (Impact = 0)"),
], fontsize=8)

# for the bar chart.
ax = axes[1]
df_fell = sensitivity[sensitivity["Impact_$M"] > 0].head(10)
colors  = [RED if c in TOP3 else BLUE for c in df_fell.index]
ax.barh(df_fell.index[::-1], df_fell["Impact_$M"][::-1],
        color=colors[::-1], height=0.65, edgecolor="white")
for i, (cat, val) in enumerate(zip(df_fell.index[::-1], df_fell["Impact_$M"][::-1])):
    ax.text(val + 3, i, f"${val:.0f}M", va="center", fontsize=8.5)

ax.set_xlabel("Impact = |Drop%/100| × Pre_size  ($M/year)")
ax.set_title("Revenue Impact Ranking\n"
             "= Annual revenue lost during Great Recession (2008–2011 vs 2005–2007)",
             fontweight="bold", color=BLUE)
ax.legend(handles=[Patch(facecolor=RED,  label="Selected Top3"),
                   Patch(facecolor=BLUE, label="Not selected")], fontsize=8)

plt.suptitle("Michigan: Identifying Recession-Sensitive Tax Categories\n"
             "Impact_i = |Drop_i%/100| × Size_i  (no normalization needed)",
             fontsize=12, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code/step4_scatter_clean.png'+"step4_impact_final.png", dpi=150, bbox_inches="tight")
plt.close()
# %% this step we gonna look at the function for social welfare.
TOP3_taxes = ["Corporation net income", "Property taxes", "Individual income"]
total = (mi_q.groupby(["Year","Quarter","period","is_recession"])["Amount"]
         .sum().reset_index().sort_values("period")
         .rename(columns={"Amount": "Total"}))
volatile = (mi_q[mi_q["Tax_Category"].isin(TOP3_taxes)]
            .groupby(["Year","Quarter","period","is_recession"])["Amount"]
            .sum().reset_index().sort_values("period")
            .rename(columns={"Amount": "Volatile"}))
rdf = total.merge(volatile[["period","Volatile"]], on="period")
# target expenditure
WINDOW = 8
buf, targets = [], []
for _, row in rdf.iterrows():
    if not row["is_recession"]:
        buf.append(row["Total"])
    t = np.mean(buf[-WINDOW:]) if len(buf) >= WINDOW else (np.mean(buf) if buf else row["Total"])
    targets.append(t)
rdf["Target"] = targets
pre_rdf = rdf[rdf["period"] < make_period(2008, 1)]
rec_rdf = rdf[rdf["is_recession"]]
print(rdf.head(10))
# %% visualization
mask      = rdf["period"].between(make_period(2006,1), make_period(2012,4))
win       = rdf[mask]
rec_start = make_period(2007, 4)
rec_end   = make_period(2009, 2)
xtick_pos = win[win["Quarter"] == 1]["period"].values
xtick_lab = win[win["Quarter"] == 1]["Year"].astype(int).values

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(win["period"], win["Total"]/1e3,  color=BLUE, lw=2,
        label="Total tax revenue (baseline, no policy)")
ax.plot(win["period"], win["Target"]/1e3, color=GOLD, lw=2.5,
        label=f"Target (rolling {WINDOW}-quarter non-recession mean)")

ax.fill_between(win["period"],
                win["Total"]/1e3, win["Target"]/1e3,
                where=win["Total"] < win["Target"],
                alpha=0.15, color=RED, label="Gap (RDF needs to fill)")
ax.axvspan(rec_start, rec_end, alpha=0.07, color=RED)
ax.axvline(rec_start, color=BLUE, lw=0.8, ls="--", alpha=0.4)
ax.axvline(rec_end,   color=BLUE, lw=0.8, ls="--", alpha=0.4)
ax.annotate("NBER recession\n2007Q4–2009Q2",
            xy=(make_period(2008,3), win["Total"].max()*0.97/1e3),
            fontsize=8, color=RED, ha="center", alpha=0.7)
ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_lab)
ax.set_ylabel("Revenue ($K)")
ax.set_title(f"Michigan: Total Revenue vs Rolling Target\n"
             f"Volatile base = {', '.join(TOP3)}",
             fontweight="bold", color=BLUE)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code/step4_scatter_clean.png' + "step5_revenue_target.png", dpi=150, bbox_inches="tight")
plt.close()
# %%
# standardize
mean_total   = rdf["Total"].mean()
rdf_norm     = rdf.copy()
rdf_norm["Total"]    = rdf["Total"]    / mean_total
rdf_norm["Volatile"] = rdf["Volatile"] / mean_total
rdf_norm["Target"]   = rdf["Target"]   / mean_total

def simulate_fund(df, s):
    fund, eff, draws, bals = 0.0, [], [], []
    for _, row in df.iterrows():
        rev, vol, tgt, rec = (row["Total"], row["Volatile"],
                              row["Target"], row["is_recession"])
        if not rec:
            dep = s * vol
            fund += dep
            eff.append(rev - dep)
            draws.append(0.0)
        else:
            gap = max(tgt - rev, 0)
            w   = min(gap, fund)
            fund -= w
            eff.append(rev + w)
            draws.append(w)
        bals.append(fund)
    out = df.copy()
    out["Effective_Spending"] = eff
    out["Withdrawal"]         = draws
    out["Fund_Balance"]       = bals
    return out

# simulate all the s options
s_grid   = np.arange(0.00, 0.51, 0.01)
eff_by_s = {s: simulate_fund(rdf_norm, s)["Effective_Spending"].values
            for s in s_grid}

for s in [0.05, 0.10, 0.11, 0.15, 0.20]:
    sim_raw = simulate_fund(rdf, s)
    rec_f   = rdf["is_recession"].values
    gap     = (rdf["Target"] - rdf["Total"]).values[rec_f].clip(min=0)
    draw    = sim_raw["Withdrawal"].values[rec_f]
    fill    = draw.sum() / gap.sum() * 100 if gap.sum() > 0 else 100
    cost    = (rdf["Volatile"] * s)[~rec_f].mean() / 1e6
# %% calibration
def social_welfare(eff, eta, alpha=1.0):
    """
    SW = α·Σlog(p_t) - η·Var(p_t)
    eff : effective spending
    eta : for penalty
    """
    p        = np.clip(np.array(eff, dtype=float), 1e-9, None)
    log_term = alpha * np.sum(np.log(p))
    var_term = eta   * np.var(p)
    return log_term - var_term
eta_grid   = np.linspace(0, 300, 600)
opt_s_list = []
for eta in eta_grid:
    scores = [social_welfare(eff_by_s[s], eta) for s in s_grid]
    opt_s_list.append(s_grid[np.argmax(scores)])
opt_s_arr = np.array(opt_s_list)

in_target = (opt_s_arr >= 0.05) & (opt_s_arr <= 0.15)
if in_target.any():
    eta_star = float(np.median(eta_grid[in_target]))
else:
    eta_star = float(eta_grid[np.argmin(np.abs(opt_s_arr - 0.10))])
# use η* to get optimal s*
scores_final = [social_welfare(eff_by_s[s], eta_star) for s in s_grid]
best_s       = s_grid[np.argmax(scores_final)]

# 福利提升
sim_0_raw    = simulate_fund(rdf, 0.0)
sim_best_raw = simulate_fund(rdf, best_s)
# s = 0和s = best_s 的福利水平, that is to look at RDF policy vs no RDF policy, how much welfare gain we can get by implementing the RDF policy with the optimal savings rate.
sw0    = social_welfare(eff_by_s[0.0],    eta_star)
swbest = social_welfare(eff_by_s[best_s], eta_star)

# RDF对衰退期公共服务的改善效果
rec_f       = rdf_norm["is_recession"].values
sw_rec_0    = social_welfare(eff_by_s[0.0][rec_f],    eta=0)
sw_rec_best = social_welfare(eff_by_s[best_s][rec_f], eta=0)
# %% visualization
# Grid search
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calibration — η vs s*
ax = axes[0]
ax.plot(eta_grid, opt_s_arr * 100, color=BLUE, lw=2)
ax.axhspan(5, 15, alpha=0.15, color=GREEN, label="Target: s* ∈ [5%, 15%]")
if in_target.any():
    ax.axvspan(eta_grid[in_target].min(), eta_grid[in_target].max(),
               alpha=0.2, color=RED,
               label=f"η ∈ [{eta_grid[in_target].min():.0f}, {eta_grid[in_target].max():.0f}]")
ax.axvline(eta_star, color=RED, ls="--", lw=2, label=f"η* = {eta_star:.1f}")
ax.set_xlabel("η  (variance penalty weight)", fontsize=10)
ax.set_ylabel("Optimal savings rate s* (%)")
ax.set_title("Step 8a: Calibration\nFind η such that s* ∈ [5%, 15%]",
             fontweight="bold", color=BLUE)
ax.legend(fontsize=9); ax.set_ylim(-1, 52)

# SW(s) curve
ax = axes[1]
sc      = np.array(scores_final)
sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
ax.plot(s_grid * 100, sc_norm, color=BLUE, lw=2.5,
        label=f"SW(s)  with η*={eta_star:.0f}")
ax.axvline(best_s * 100, color=RED, ls="--", lw=2,
           label=f"Optimal s* = {best_s*100:.0f}%")
ax.scatter([best_s * 100], [sc_norm[np.argmax(sc)]], color=RED, s=120, zorder=6)
ax.axvspan(5, 15, alpha=0.08, color=GREEN, label="Calibration target [5%, 15%]")
ax.annotate(f"  s* = {best_s*100:.0f}%\n  SW gain: {(swbest-sw0)/abs(sw0)*100:+.1f}%",
            xy=(best_s*100, sc_norm[np.argmax(sc)]),
            xytext=(best_s*100+4, sc_norm[np.argmax(sc)]*0.9),
            fontsize=10, color=RED, fontweight="bold")
ax.set_xlabel("Savings Rate s (%)")
ax.set_ylabel("Normalised Social Welfare")
ax.set_title(f"Step 8b: Grid Search\nSW = Σlog(p_t) - {eta_star:.0f}·Var(p_t),  s* = {best_s*100:.0f}%",
             fontweight="bold", color=BLUE)
ax.legend(fontsize=9)

plt.suptitle("Michigan RDF: Social Welfare Calibration & Optimal Savings Rate\n"
             "SW = α·Σlog(p_t(X_t)) - η·Var(p_t(X_t)),  α=1,  η calibrated to Michigan",
             fontsize=11, fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code/step4_scatter_clean.pngstep5_revenue_target.png' + "step8_calibration_gridsearch.png", dpi=150, bbox_inches="tight")
plt.close()
# %% 回测
sim_best = simulate_fund(rdf, best_s)

# Revenue vs Target vs Policy（2006-2012）
mask      = sim_best["period"].between(make_period(2006,1), make_period(2012,4))
win       = sim_best[mask].copy()
xtick_pos = win[win["Quarter"] == 1]["period"].values
xtick_lab = win[win["Quarter"] == 1]["Year"].astype(int).values

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(win["period"], win["Total"]/1e3,              color=BLUE,  lw=2,
        label="Baseline revenue (no policy)")
ax.plot(win["period"], win["Target"]/1e3,             color=GOLD,  lw=2.5,
        label=f"Target (rolling {WINDOW}-quarter mean)")
ax.plot(win["period"], win["Effective_Spending"]/1e3, color=GREEN, lw=2, ls="--",
        label=f"Policy spending (s*={best_s*100:.0f}%)")
ax.fill_between(win["period"],
                win["Total"]/1e3, win["Effective_Spending"]/1e3,
                where=win["is_recession"], alpha=0.12, color=GREEN,
                label="RDF drawdown")
ax.axvspan(rec_start, rec_end, alpha=0.07, color=RED)
ax.axvline(rec_start, color=BLUE, lw=0.8, ls="--", alpha=0.4)
ax.axvline(rec_end,   color=BLUE, lw=0.8, ls="--", alpha=0.4)
ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_lab)
ax.set_ylabel("Revenue / Spending ($K)")
ax.set_title(f"Michigan: Baseline vs Target vs Policy Spending\n"
             f"s* = {best_s*100:.0f}%  |  Volatile: {', '.join(TOP3)}",
             fontweight="bold", color=BLUE)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code/step4_scatter_clean.pngstep5_revenue_target.pngstep8_calibration_gridsearch.png' + "step9_revenue_policy.png", dpi=150, bbox_inches="tight")
plt.close()


# 基金余额全历史
fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(sim_best["period"], sim_best["Fund_Balance"]/1e6, alpha=0.2, color=BLUE)
ax.plot(sim_best["period"], sim_best["Fund_Balance"]/1e6, color=BLUE, lw=1.8)
for sy, sq, ey, eq in RECESSION_PERIODS:
    ax.axvspan(make_period(sy,sq), make_period(ey,eq), alpha=0.1, color=RED)
for label, yr, q in [("2001", 2001, 2), ("2008-09", 2008, 3), ("2020", 2020, 1)]:
    ax.annotate(label, xy=(make_period(yr,q), sim_best["Fund_Balance"].max()*0.55/1e6),
                color=RED, fontsize=8, ha="center", alpha=0.7)
ax.set_ylabel("Fund Balance ($M)")
ax.set_title(f"Michigan RDF Balance — Full History  (s*={best_s*100:.0f}%)\n"
             f"Red shading = recession periods",
             fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code/step4_scatter_clean.pngstep5_revenue_target.pngstep8_calibration_gridsearch.png' + "step9_fund_balance.png", dpi=150, bbox_inches="tight")
plt.close()
# %% try different eta values to see how the optimal s* changes, and how the welfare curve looks like for different eta values.
eta_sens  = [100, 150, eta_star, 250, 300]
palette   = [BLUE, GREEN, RED, GOLD, GRAY]

fig, ax = plt.subplots(figsize=(10, 5))
for eta_val, col in zip(eta_sens, palette):
    sc_s = np.array([social_welfare(eff_by_s[s], eta_val) for s in s_grid])
    sc_n = (sc_s - sc_s.min()) / (sc_s.max() - sc_s.min() + 1e-9)
    opt  = s_grid[np.argmax(sc_s)]
    lw   = 2.5 if abs(eta_val - eta_star) < 1 else 1.5
    ls   = "-" if abs(eta_val - eta_star) < 1 else "--"
    ax.plot(s_grid * 100, sc_n, color=col, lw=lw, ls=ls,
            label=f"η={eta_val:.0f}  →  s*={opt*100:.0f}%"
                  + (" (η*)" if abs(eta_val - eta_star) < 1 else ""))

ax.set_xlabel("Savings Rate s (%)")
ax.set_ylabel("Normalised Welfare")
ax.set_title(f"Sensitivity: Optimal s* across Different η Values\n"
             f"(SW = Σlog(p_t) - η·Var(p_t),  data normalised)",
             fontweight="bold", color=BLUE)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('/Users/lilianli/Library/CloudStorage/GoogleDrive-yiyuanli@umich.edu/My Drive/Datathon 2026/Michigan Code/step4_scatter_clean.pngstep5_revenue_target.pngstep8_calibration_gridsearch.pngstep9_revenue_policy.png' + "step9_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()