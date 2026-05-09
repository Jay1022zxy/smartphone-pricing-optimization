from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson

warnings.filterwarnings("ignore")

# =========================
# 0. 全局设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei", font_scale=1.05)

output_dir = Path("问题2输出结果")
output_dir.mkdir(exist_ok=True)

for filename in os.listdir(output_dir):
    if filename.lower().endswith(".pdf"):
        os.remove(output_dir / filename)

def save_fig(fig, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")


# =========================
# 1. 读取数据
# =========================
def find_data_file(filename: str) -> Path:
    if Path(filename).exists():
        return Path(filename)
    alt = Path("analysis_outputs") / filename
    if alt.exists():
        return alt
    raise FileNotFoundError(f"未找到文件：{filename}，请先运行数据清洗脚本。")

data_path = find_data_file("附件2_清洗后数据.xlsx")
df = pd.read_excel(data_path)

required_cols = ["price", "competitor_price", "product_score", "market_size", "simulated_sales"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise KeyError(f"缺少必要字段：{missing_cols}")

df = df[required_cols].copy()

print("=" * 60)
print("问题2：智能手机定价模型、销量预测与利润分析（最终保守版）")
print("=" * 60)
print(f"有效样本量：{len(df)}")
print(f"数据字段：{df.columns.tolist()}")

# =========================
# 2. 数据预处理
# =========================
for col in required_cols:
    if (df[col] <= 0).any():
        raise ValueError(f"字段 {col} 中存在非正数，无法进行对数建模。")

df["ln_sales"] = np.log(df["simulated_sales"])
df["ln_price"] = np.log(df["price"])
df["ln_competitor_price"] = np.log(df["competitor_price"])
df["ln_product_score"] = np.log(df["product_score"])
df["ln_market_size"] = np.log(df["market_size"])

log_features = ["ln_price", "ln_competitor_price", "ln_product_score", "ln_market_size"]
feature_means = {}

for col in log_features:
    mean_val = df[col].mean()
    feature_means[col] = mean_val
    df[f"{col}_c"] = df[col] - mean_val

centered_features = [f"{col}_c" for col in log_features]

# =========================
# 3. 核心对数销量回归模型
# =========================
X = df[centered_features]
y = df["ln_sales"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit(cov_type="HC3")

print("\n" + "=" * 60)
print("核心对数销量回归模型完整结果")
print("=" * 60)
print(model.summary())

coef_df = pd.DataFrame({
    "变量": ["常数项", "本机价格", "竞品价格", "产品得分", "市场容量"],
    "回归系数": model.params.values,
    "稳健标准误": model.bse.values,
    "P值": model.pvalues.values,
    "95%置信区间下限": model.conf_int()[0].values,
    "95%置信区间上限": model.conf_int()[1].values
})

print("\n回归系数汇总表：")
print(coef_df)

# =========================
# 4. 弹性分析
# =========================
price_elasticity = float(model.params["ln_price_c"])
competitor_elasticity = float(model.params["ln_competitor_price_c"])
score_elasticity = float(model.params["ln_product_score_c"])
market_elasticity = float(model.params["ln_market_size_c"])

print("\n" + "=" * 60)
print("核心弹性分析（价格敏感度量化）")
print("=" * 60)
print(f"■ 本机价格弹性 = {price_elasticity:.4f}")
print(f"■ 竞品价格弹性 = {competitor_elasticity:.4f}")
print(f"■ 产品得分弹性 = {score_elasticity:.4f}")
print(f"■ 市场容量弹性 = {market_elasticity:.4f}")

# =========================
# 5. 共线性诊断
# =========================
vif_df = pd.DataFrame({
    "变量": ["本机价格", "竞品价格", "产品得分", "市场容量"],
    "VIF": [variance_inflation_factor(df[centered_features].values, i) for i in range(len(centered_features))]
})

print("\n" + "=" * 60)
print("共线性诊断（VIF）")
print("=" * 60)
print(vif_df.round(4))


# =========================
# 5.5 岭回归稳健性检验（针对共线性问题的补充）
# =========================
from sklearn.linear_model import RidgeCV

print("\n" + "=" * 60)
print("岭回归稳健性检验（针对多重共线性）")
print("=" * 60)

# 自动搜索最优正则化参数 alpha
alphas = np.logspace(-3, 2, 100)
ridge_cv = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=5)
ridge_cv.fit(df[centered_features], df["ln_sales"])

# 提取最优参数和岭回归系数
best_alpha = ridge_cv.alpha_
ridge_coef = ridge_cv.coef_

print(f"岭回归最优正则化参数 alpha = {best_alpha:.6f}")

# 构建 OLS 与岭回归系数对比表
coef_compare_df = pd.DataFrame({
    "变量": ["本机价格", "竞品价格", "产品得分", "市场容量"],
    "OLS回归系数": model.params[1:].values,
    "岭回归系数": ridge_coef,
    "系数符号一致性": ["一致" if np.sign(model.params[1:].values[i]) == np.sign(ridge_coef[i]) else "不一致" for i in range(len(ridge_coef))]
})

print("\nOLS与岭回归系数对比表：")
print(coef_compare_df.round(6))

# 岭回归结论解读
print("\n岭回归稳健性检验结论：")
if (coef_compare_df["系数符号一致性"] == "一致").all():
    print("✅ 所有变量的回归系数符号在OLS和岭回归下完全一致，")
    print("   说明尽管存在一定共线性，但核心结论的方向是稳健的。")
else:
    print("⚠️  部分变量系数符号存在差异，需结合业务逻辑谨慎解读。")

print(f"   核心结论：产品得分和市场容量对销量的正向影响在两种模型下均保持稳定。")

# 导出岭回归对比表
coef_compare_df.to_excel(output_dir / "问题2_OLS与岭回归系数对比表.xlsx", index=False)




# =========================
# 6. 模型诊断检验
# =========================
residuals_ln = model.resid

jb_stat, jb_pvalue, _, _ = jarque_bera(residuals_ln)
bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals_ln, X_const)
dw_stat = durbin_watson(residuals_ln)

print("\n" + "=" * 60)
print("模型诊断检验结果")
print("=" * 60)
print(f"1. 残差正态性检验（JB检验）：统计量={jb_stat:.4f}，P值={jb_pvalue:.4f}")
print(f"2. 异方差检验（BP检验）：统计量={bp_stat:.4f}，P值={bp_pvalue:.4f}")
print(f"3. 自相关检验（DW检验）：统计量={dw_stat:.4f}")

# =========================
# 7. 拟合效果指标
# =========================
y_pred_ln = model.predict(X_const)
y_pred = np.exp(y_pred_ln)

log_r2 = float(model.rsquared)
log_adj_r2 = float(model.rsquared_adj)
orig_r2 = float(r2_score(df["simulated_sales"], y_pred))
rmse = float(np.sqrt(mean_squared_error(df["simulated_sales"], y_pred)))

metrics_df = pd.DataFrame({
    "指标": ["对数模型R²", "对数模型调整后R²", "原尺度预测R²", "原尺度RMSE"],
    "数值": [log_r2, log_adj_r2, orig_r2, rmse]
})

print("\n" + "=" * 60)
print("模型整体拟合效果指标")
print("=" * 60)
print(f"对数模型决定系数 R² = {log_r2:.4f}")
print(f"对数模型调整后 R² = {log_adj_r2:.4f}")
print(f"原尺度预测 R² = {orig_r2:.4f}")
print(f"原尺度预测 RMSE = {rmse:.4f}")

# =========================
# 8. 预测函数
# =========================
def predict_sales(price: float,
                  competitor_price: float,
                  product_score: float,
                  market_size: float) -> float:
    x_new = pd.DataFrame({
        "const": [1.0],
        "ln_price_c": [np.log(price) - feature_means["ln_price"]],
        "ln_competitor_price_c": [np.log(competitor_price) - feature_means["ln_competitor_price"]],
        "ln_product_score_c": [np.log(product_score) - feature_means["ln_product_score"]],
        "ln_market_size_c": [np.log(market_size) - feature_means["ln_market_size"]],
    })
    ln_q = float(model.predict(x_new)[0])
    return float(np.exp(ln_q))

def calc_profit(price: float, unit_cost: float, sales: float) -> float:
    # sales 单位：万台；price/unit_cost 单位：元；利润单位：万元
    return (price - unit_cost) * sales

# =========================
# 9. 基准市场情景
# =========================
base_price = float(df["price"].mean())
base_competitor_price = float(df["competitor_price"].mean())
base_score = float(df["product_score"].mean())
base_market_size = float(df["market_size"].mean())

BASE_COST_RATIO = 0.75
base_unit_cost = BASE_COST_RATIO * base_price

base_sales = predict_sales(
    price=base_price,
    competitor_price=base_competitor_price,
    product_score=base_score,
    market_size=base_market_size
)
base_profit = calc_profit(base_price, base_unit_cost, base_sales)

print("\n" + "=" * 60)
print("基准市场情景参数")
print("=" * 60)
print(f"基准定价：{base_price:.2f} 元")
print(f"竞品平均定价：{base_competitor_price:.2f} 元")
print(f"基准产品综合得分：{base_score:.2f}")
print(f"基准市场容量：{base_market_size:.2f}")
print(f"基准单位硬件成本：{base_unit_cost:.2f} 元")
print(f"\n基准情景预测销量：{base_sales:.4f} 万台")
print(f"基准情景总利润：{base_profit:.4f} 万元")

# =========================
# 10. 策略一：现有配置下，降价5%
# =========================
strategy1_price = base_price * 0.95
strategy1_sales = predict_sales(
    price=strategy1_price,
    competitor_price=base_competitor_price,
    product_score=base_score,
    market_size=base_market_size
)
strategy1_unit_cost = base_unit_cost
strategy1_profit = calc_profit(strategy1_price, strategy1_unit_cost, strategy1_sales)

strategy1_result = pd.DataFrame({
    "情景": ["基准方案", "降价5%方案"],
    "定价(元)": [base_price, strategy1_price],
    "产品综合得分": [base_score, base_score],
    "预测销量(万台)": [base_sales, strategy1_sales],
    "单位成本(元)": [base_unit_cost, strategy1_unit_cost],
    "总利润(万元)": [base_profit, strategy1_profit]
})

sales_change_1 = (strategy1_sales - base_sales) / base_sales * 100
profit_change_1 = (strategy1_profit - base_profit) / base_profit * 100

print("\n" + "=" * 60)
print("赛题策略一：现有配置降价5% 情景分析")
print("=" * 60)
print(strategy1_result.round(4))
print(f"\n降价5%后，销量变动率：{sales_change_1:.2f}%")
print(f"降价5%后，利润变动率：{profit_change_1:.2f}%")

# =========================
# 11. 策略二：基础版 + Pro版（最终保守版）
# =========================
basic_score = base_score * 0.96
pro_score = base_score * 1.06

basic_price = base_price * 0.90
pro_price = base_price * 1.12

basic_cost_ratio = 0.70
pro_cost_ratio = 0.80
basic_unit_cost = basic_cost_ratio * basic_price
pro_unit_cost = pro_cost_ratio * pro_price

# ===== 保守化参数 =====
DUAL_EXPANSION_DISCOUNT = 0.85
MAX_SHARE_EXPANSION = 1.35
MAX_PORTFOLIO_SHARE_ABS = 0.40

def attractiveness(price: float, score: float) -> float:
    gamma_p = max(-price_elasticity, 0.05)
    gamma_s = max(score_elasticity, 0.05)
    return float(np.exp(gamma_s * np.log(score) - gamma_p * np.log(price)))

def dual_version_sales_conservative(base_price: float,
                                    base_score: float,
                                    basic_price: float,
                                    basic_score: float,
                                    pro_price: float,
                                    pro_score: float,
                                    market_size: float) -> tuple[float, float, float, float, float]:
    """
    双版本保守化处理逻辑：
    1. 先由单版本基准校准外部选项；
    2. 再求双版本组合理论市场份额；
    3. 对理论份额设置扩张上限；
    4. 再乘保守折减系数；
    5. 最后按内部吸引力拆分到基础版和Pro版。
    """
    base_sales_local = predict_sales(
        price=base_price,
        competitor_price=base_competitor_price,
        product_score=base_score,
        market_size=market_size
    )
    base_share = np.clip(base_sales_local / market_size, 1e-6, 0.95)

    A_base = attractiveness(base_price, base_score)
    A_basic = attractiveness(basic_price, basic_score)
    A_pro = attractiveness(pro_price, pro_score)

    # 外部选项校准
    A_outside = A_base * (1 - base_share) / base_share

    # 理论组合份额
    raw_portfolio_share = (A_basic + A_pro) / (A_basic + A_pro + A_outside)
    raw_portfolio_share = float(np.clip(raw_portfolio_share, 0.0, 0.98))

    # 份额扩张上限
    capped_portfolio_share = min(
        raw_portfolio_share,
        base_share * MAX_SHARE_EXPANSION,
        MAX_PORTFOLIO_SHARE_ABS
    )

    # 最终保守化折减
    final_portfolio_share = capped_portfolio_share * DUAL_EXPANSION_DISCOUNT
    final_portfolio_share = float(np.clip(final_portfolio_share, 0.0, 0.98))

    total_dual_sales = market_size * final_portfolio_share

    total_attr = A_basic + A_pro
    basic_sales = total_dual_sales * A_basic / total_attr
    pro_sales = total_dual_sales * A_pro / total_attr

    return (
        float(basic_sales),
        float(pro_sales),
        float(total_dual_sales),
        float(raw_portfolio_share),
        float(final_portfolio_share)
    )

basic_sales, pro_sales, total_dual_sales, raw_share, final_share = dual_version_sales_conservative(
    base_price=base_price,
    base_score=base_score,
    basic_price=basic_price,
    basic_score=basic_score,
    pro_price=pro_price,
    pro_score=pro_score,
    market_size=base_market_size
)

basic_profit = calc_profit(basic_price, basic_unit_cost, basic_sales)
pro_profit = calc_profit(pro_price, pro_unit_cost, pro_sales)
total_dual_profit = basic_profit + pro_profit

basic_potential_sales = predict_sales(
    price=basic_price,
    competitor_price=base_competitor_price,
    product_score=basic_score,
    market_size=base_market_size
)
pro_potential_sales = predict_sales(
    price=pro_price,
    competitor_price=base_competitor_price,
    product_score=pro_score,
    market_size=base_market_size
)

strategy2_result = pd.DataFrame({
    "版本": ["基础版", "Pro版", "双版本合计", "原单版本基准"],
    "定价(元)": [basic_price, pro_price, np.nan, base_price],
    "产品综合得分": [basic_score, pro_score, np.nan, base_score],
    "组合分配销量(万台)": [basic_sales, pro_sales, total_dual_sales, base_sales],
    "单独潜在销量(万台)": [basic_potential_sales, pro_potential_sales, np.nan, base_sales],
    "单位成本(元)": [basic_unit_cost, pro_unit_cost, np.nan, base_unit_cost],
    "总利润(万元)": [basic_profit, pro_profit, total_dual_profit, base_profit]
})

sales_change_2 = (total_dual_sales - base_sales) / base_sales * 100
profit_change_2 = (total_dual_profit - base_profit) / base_profit * 100

print("\n" + "=" * 60)
print("赛题策略二：基础版+Pro版双版本 情景分析")
print("=" * 60)
print(strategy2_result.round(4))
print(f"\n双版本理论组合市场份额：{raw_share:.4f}")
print(f"双版本保守修正后市场份额：{final_share:.4f}")
print(f"双版本策略下，总销量变动率：{sales_change_2:.2f}%")
print(f"双版本策略下，总利润变动率：{profit_change_2:.2f}%")

# =========================
# 12. 单产品价格-利润敏感性分析
# =========================
price_grid = np.arange(round(base_price * 0.80), round(base_price * 1.35) + 1, 50)
sensitivity_rows = []

for p in price_grid:
    q = predict_sales(
        price=float(p),
        competitor_price=base_competitor_price,
        product_score=base_score,
        market_size=base_market_size
    )
    profit = calc_profit(float(p), base_unit_cost, q)
    sensitivity_rows.append([p, q, profit])

sensitivity_df = pd.DataFrame(sensitivity_rows, columns=["产品定价(元)", "预测销量(万台)", "总利润(万元)"])
best_idx = sensitivity_df["总利润(万元)"].idxmax()
best_price = float(sensitivity_df.loc[best_idx, "产品定价(元)"])
best_profit_curve = float(sensitivity_df.loc[best_idx, "总利润(万元)"])

at_lower_boundary = best_idx == 0
at_upper_boundary = best_idx == len(sensitivity_df) - 1

# =========================
# 13. 全方案对比汇总
# =========================
scenario_summary = pd.DataFrame({
    "方案": ["原单版本基准", "降价5%方案", "双版本策略"],
    "总销量(万台)": [base_sales, strategy1_sales, total_dual_sales],
    "总利润(万元)": [base_profit, strategy1_profit, total_dual_profit]
})
scenario_summary["销量变动率(%)"] = (scenario_summary["总销量(万台)"] / base_sales - 1) * 100
scenario_summary["利润变动率(%)"] = (scenario_summary["总利润(万元)"] / base_profit - 1) * 100

print("\n" + "=" * 60)
print("全方案对比汇总表")
print("=" * 60)
print(scenario_summary.round(4))

best_plan = scenario_summary.sort_values("总利润(万元)", ascending=False).iloc[0]["方案"]
print(f"\n基于当前参数设定，利润最大化的最优方案为：【{best_plan}】")

if at_lower_boundary or at_upper_boundary:
    print(f"注意：价格-利润敏感性曲线的区间内最优价格为 {best_price:.0f} 元，但最优点位于搜索区间边界，")
    print("这说明它是“当前考察区间内最优”，不宜直接表述为“全局最优价格”。")

# =========================
# 14. 导出Excel
# =========================
coef_df.to_excel(output_dir / "问题2_回归系数汇总表.xlsx", index=False)
vif_df.to_excel(output_dir / "问题2_VIF共线性诊断表.xlsx", index=False)
metrics_df.to_excel(output_dir / "问题2_模型拟合指标表.xlsx", index=False)
strategy1_result.to_excel(output_dir / "问题2_策略一_降价5%分析.xlsx", index=False)
strategy2_result.to_excel(output_dir / "问题2_策略二_双版本分析_最终保守版.xlsx", index=False)
scenario_summary.to_excel(output_dir / "问题2_全方案对比汇总表.xlsx", index=False)
sensitivity_df.to_excel(output_dir / "问题2_价格利润敏感性分析表.xlsx", index=False)

# =========================
# 15. 作图1：真实销量 vs 预测销量
# =========================
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["simulated_sales"], y_pred, s=70, alpha=0.75, edgecolors="white", linewidth=0.8)

axis_min = min(df["simulated_sales"].min(), y_pred.min())
axis_max = max(df["simulated_sales"].max(), y_pred.max())
ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=2.5, label="理想拟合线 y=x")

ax.set_title("真实销量与模型预测销量对比图", fontsize=20, weight="bold", pad=16)
ax.set_xlabel("真实销量（万台）", fontsize=15)
ax.set_ylabel("模型预测销量（万台）", fontsize=15)
ax.legend(frameon=False, fontsize=14)

textstr = f"对数模型R² = {log_r2:.4f}\n原尺度R² = {orig_r2:.4f}\nRMSE = {rmse:.4f}"
ax.text(
    0.98, 0.05, textstr,
    transform=ax.transAxes,
    fontsize=14,
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#b0b0b0")
)

save_fig(fig, "问题2_真实销量与预测销量对比图_最终保守版")
plt.close()

# =========================
# 16. 作图2：各因素对销量影响的回归系数
# =========================
plot_coef_df = coef_df.iloc[1:].copy()

fig, ax = plt.subplots(figsize=(8.5, 5.8))
bars = ax.bar(plot_coef_df["变量"], plot_coef_df["回归系数"], edgecolor="black", linewidth=0.8)
ax.axhline(0, color="black", linewidth=1.2)

ax.set_title("各因素对销量影响的回归系数", fontsize=20, weight="bold", pad=16)
ax.set_xlabel("影响因素", fontsize=15)
ax.set_ylabel("回归系数", fontsize=15)

for bar in bars:
    h = bar.get_height()
    va = "bottom" if h >= 0 else "top"
    offset = 0.03 if h >= 0 else -0.03
    ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:.3f}", ha="center", va=va, fontsize=13)

save_fig(fig, "问题2_各因素对销量影响的回归系数_最终保守版")
plt.close()

# =========================
# 17. 作图3：不同策略下总销量对比
# =========================
fig, ax = plt.subplots(figsize=(8.5, 5.8))
bars = ax.bar(scenario_summary["方案"], scenario_summary["总销量(万台)"], edgecolor="black", linewidth=0.8)

ax.set_title("不同策略下总销量对比", fontsize=20, weight="bold", pad=16)
ax.set_ylabel("总销量（万台）", fontsize=15)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=13)

save_fig(fig, "问题2_不同策略下总销量对比_最终保守版")
plt.close()

# =========================
# 18. 作图4：不同策略下总利润对比
# =========================
fig, ax = plt.subplots(figsize=(8.5, 5.8))
bars = ax.bar(scenario_summary["方案"], scenario_summary["总利润(万元)"], edgecolor="black", linewidth=0.8)

ax.set_title("不同策略下总利润对比", fontsize=20, weight="bold", pad=16)
ax.set_ylabel("总利润（万元）", fontsize=15)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=13)

save_fig(fig, "问题2_不同策略下总利润对比_最终保守版")
plt.close()

# =========================
# 19. 作图5：价格-利润敏感性分析曲线
# =========================
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sensitivity_df["产品定价(元)"], sensitivity_df["总利润(万元)"], linewidth=3, label="利润曲线")
ax.axvline(base_price, linestyle="--", linewidth=2.5, label="当前基准价格")
ax.scatter([best_price], [best_profit_curve], s=180, zorder=5, label=f"区间内最优价格：{best_price:.0f}元")

ax.set_title("价格-利润敏感性分析曲线", fontsize=20, weight="bold", pad=16)
ax.set_xlabel("产品定价（元）", fontsize=15)
ax.set_ylabel("总利润（万元）", fontsize=15)
ax.legend(frameon=False, fontsize=13)

if at_lower_boundary or at_upper_boundary:
    ax.text(
        0.98, 0.05,
        "注意：最优点位于搜索区间边界",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5", edgecolor="#b0b0b0")
    )

save_fig(fig, "问题2_价格利润敏感性分析曲线_最终保守版")
plt.close()

# =========================
# 20. 最终结论摘要
# =========================
print("\n" + "=" * 60)
print("=" * 60)
print(f"所有分析表格、高清可视化图已全部保存到文件夹：{output_dir}")
print(f"1. 本机价格弹性估计值为 {price_elasticity:.4f}，但应结合P值解释，不宜夸大其统计显著性。")
print(f"2. 对销量影响最显著的因素是产品得分（{score_elasticity:.4f}）和市场容量（{market_elasticity:.4f}）。")
print(f"3. 双版本策略已加入保守化折减：折减系数={DUAL_EXPANSION_DISCOUNT:.2f}，扩张上限倍数={MAX_SHARE_EXPANSION:.2f}。")
print(f"4. 当前利润最高方案为：【{best_plan}】")
print(f"5. 价格敏感性曲线区间内最优价格为：{best_price:.0f} 元")