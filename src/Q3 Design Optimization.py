from __future__ import annotations

import os
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =========================
# 0. 全局设置
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
sns.set_theme(style="whitegrid", font="Microsoft YaHei", font_scale=1.05)

MAIN_COLOR = "#4C78A8"
SECOND_COLOR = "#59A14F"
THIRD_COLOR = "#E15759"
ACCENT_COLOR = "#B279A2"
GRID_COLOR = "#D9D9D9"

output_dir = Path("问题3输出结果")
output_dir.mkdir(exist_ok=True)

for filename in os.listdir(output_dir):
    if filename.lower().endswith((".pdf", ".tmp", ".log")):
        os.remove(output_dir / filename)


def save_fig(fig, name: str) -> None:
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )


# =========================
# 1. 数据读取
# =========================
def find_data_file(filename: str) -> Path:
    candidates = [
        Path(filename),
        Path("analysis_outputs") / filename,
        Path("../") / filename,
        Path("问题1输出结果") / filename,
        Path("问题2输出结果") / filename,
        Path("问题2输出结果_最终保守版") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到文件：{filename}")


review_path = find_data_file("附件1_清洗后数据.xlsx")
sales_path = find_data_file("附件2_清洗后数据.xlsx")

df_review = pd.read_excel(review_path)
df_sales = pd.read_excel(sales_path)

review_required_cols = ["总体评分", "外观", "屏幕", "摄像", "续航", "性能", "发热控制"]
sales_required_cols = ["price", "competitor_price", "product_score", "market_size", "simulated_sales"]

missing_review = [c for c in review_required_cols if c not in df_review.columns]
missing_sales = [c for c in sales_required_cols if c not in df_sales.columns]

if missing_review:
    raise KeyError(f"附件1缺少必要字段：{missing_review}")
if missing_sales:
    raise KeyError(f"附件2缺少必要字段：{missing_sales}")

print("=" * 70)
print("问题3：智能手机设计参数化建模与利润最大化优化（最终版）")
print("=" * 70)
print(f"附件1用户评价有效样本量：{len(df_review)}")
print(f"附件2市场销量有效样本量：{len(df_sales)}")


# =========================
# 2. 问题1子模型：维度满意度 -> 总体评分
# =========================
X_review = df_review[["外观", "屏幕", "摄像", "续航", "性能", "发热控制"]]
y_review = df_review["总体评分"]

X_review_const = sm.add_constant(X_review)
review_model = sm.OLS(y_review, X_review_const).fit(cov_type="HC3")

print("\n" + "=" * 70)
print("问题1子模型：总体评分回归结果")
print("=" * 70)
print(f"模型R² = {review_model.rsquared:.4f}，调整后R² = {review_model.rsquared_adj:.4f}")

review_coef_df = pd.DataFrame({
    "变量": ["常数项", "外观", "屏幕", "摄像", "续航", "性能", "发热控制"],
    "回归系数": review_model.params.values,
    "稳健标准误": review_model.bse.values,
    "P值": review_model.pvalues.values,
    "95%置信区间下限": review_model.conf_int().iloc[:, 0].values,
    "95%置信区间上限": review_model.conf_int().iloc[:, 1].values
})
print("\n总体评分回归系数汇总表：")
print(review_coef_df.round(6))


# =========================
# 3. 问题2子模型：价格/得分/市场容量 -> 销量
# 与问题二最终版一致：中心化对数模型
# =========================
for col in sales_required_cols:
    if (df_sales[col] <= 0).any():
        invalid_count = int((df_sales[col] <= 0).sum())
        raise ValueError(f"字段【{col}】中存在{invalid_count}个非正数，无法进行对数建模。")

df_sales = df_sales.copy()
df_sales["ln_sales"] = np.log(df_sales["simulated_sales"])
df_sales["ln_price"] = np.log(df_sales["price"])
df_sales["ln_competitor_price"] = np.log(df_sales["competitor_price"])
df_sales["ln_product_score"] = np.log(df_sales["product_score"])
df_sales["ln_market_size"] = np.log(df_sales["market_size"])

log_features = ["ln_price", "ln_competitor_price", "ln_product_score", "ln_market_size"]
feature_means = {col: float(df_sales[col].mean()) for col in log_features}

for col in log_features:
    df_sales[f"{col}_c"] = df_sales[col] - feature_means[col]

X_sales = df_sales[[f"{col}_c" for col in log_features]]
y_sales = df_sales["ln_sales"]

X_sales_const = sm.add_constant(X_sales)
sales_model = sm.OLS(y_sales, X_sales_const).fit(cov_type="HC3")

print("\n" + "=" * 70)
print("问题2子模型：对数销量回归结果")
print("=" * 70)
print(f"模型R² = {sales_model.rsquared:.4f}，调整后R² = {sales_model.rsquared_adj:.4f}")
print(f"核心价格弹性系数 = {sales_model.params['ln_price_c']:.4f}")

sales_coef_df = pd.DataFrame({
    "变量": ["常数项", "本机价格", "竞品价格", "产品得分", "市场容量"],
    "回归系数": sales_model.params.values,
    "稳健标准误": sales_model.bse.values,
    "P值": sales_model.pvalues.values,
    "95%置信区间下限": sales_model.conf_int().iloc[:, 0].values,
    "95%置信区间上限": sales_model.conf_int().iloc[:, 1].values
})
print("\n销量回归系数汇总表：")
print(sales_coef_df.round(6))


# =========================
# 4. 基准参数与校准目标
# =========================
BASE_BATTERY = 5000
BASE_CAMERA = 50
BASE_CPU = 85
BASE_REFRESH = 120

BASE_PRICE = float(df_sales["price"].mean())
BASE_COMPETITOR_PRICE = float(df_sales["competitor_price"].mean())
BASE_MARKET_SIZE = float(df_sales["market_size"].mean())
BASE_PRODUCT_SCORE = float(df_sales["product_score"].mean())

BASE_COST_RATIO = 0.75
BASE_UNIT_COST = BASE_COST_RATIO * BASE_PRICE

TARGET_BASE_DIMS = {
    "外观": float(df_review["外观"].mean()),
    "屏幕": float(df_review["屏幕"].mean()),
    "摄像": float(df_review["摄像"].mean()),
    "续航": float(df_review["续航"].mean()),
    "性能": float(df_review["性能"].mean()),
    "发热控制": float(df_review["发热控制"].mean()),
}
TARGET_BASE_RATING = float(df_review["总体评分"].mean())
TARGET_BASE_SCORE = float(df_sales["product_score"].mean())

SCORE_PER_STAR = float(
    np.clip(
        df_sales["product_score"].std() / df_review["总体评分"].std(),
        4.0,
        8.0
    )
)

print("\n" + "=" * 70)
print("当前产品基准参数设定（校准版）")
print("=" * 70)
print(f"硬件配置：{BASE_BATTERY}mAh电池 | {BASE_CAMERA}MP主摄 | 处理器性能指数{BASE_CPU} | {BASE_REFRESH}Hz刷新率")
print(f"市场基准：定价{BASE_PRICE:.2f}元 | 竞品均价{BASE_COMPETITOR_PRICE:.2f}元 | 基准市场容量{BASE_MARKET_SIZE:.2f}")
print(f"成本基准：单位硬件成本{BASE_UNIT_COST:.2f}元")
print(f"评分校准：基准总体评分={TARGET_BASE_RATING:.4f} | 基准产品得分={TARGET_BASE_SCORE:.4f} | 映射斜率={SCORE_PER_STAR:.4f}")


# =========================
# 5. 辅助函数
# =========================
def sigmoid_scaled(x: float, x0: float, k: float, low: float = -1.0, high: float = 1.0) -> float:
    sigmoid_val = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    return low + (high - low) * sigmoid_val


def clip_score(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(np.clip(x, low, high))


def predict_overall_rating(
    appearance: float,
    screen: float,
    camera: float,
    battery: float,
    performance: float,
    thermal: float
) -> float:
    x_new = pd.DataFrame({
        "const": [1.0],
        "外观": [appearance],
        "屏幕": [screen],
        "摄像": [camera],
        "续航": [battery],
        "性能": [performance],
        "发热控制": [thermal],
    })
    rating_pred = float(review_model.predict(x_new)[0])
    return float(np.clip(rating_pred, 1.0, 5.0))


def rating_to_product_score(rating: float) -> float:
    score = TARGET_BASE_SCORE + SCORE_PER_STAR * (rating - TARGET_BASE_RATING)
    return float(np.clip(score, 60.0, 100.0))


def predict_sales(
    price: float,
    competitor_price: float,
    product_score: float,
    market_size: float
) -> float:
    x_new = pd.DataFrame({
        "const": [1.0],
        "ln_price_c": [np.log(price) - feature_means["ln_price"]],
        "ln_competitor_price_c": [np.log(competitor_price) - feature_means["ln_competitor_price"]],
        "ln_product_score_c": [np.log(product_score) - feature_means["ln_product_score"]],
        "ln_market_size_c": [np.log(market_size) - feature_means["ln_market_size"]],
    })
    ln_sales_pred = float(sales_model.predict(x_new)[0])
    return float(np.exp(ln_sales_pred))


def design_to_dimensions(
    battery: float,
    camera_mp: float,
    cpu_index: float,
    refresh_rate: float
) -> dict[str, float]:
    appearance = TARGET_BASE_DIMS["外观"]

    screen = TARGET_BASE_DIMS["屏幕"] + sigmoid_scaled(refresh_rate, x0=BASE_REFRESH, k=0.06)
    camera = TARGET_BASE_DIMS["摄像"] + sigmoid_scaled(camera_mp, x0=BASE_CAMERA, k=0.05)
    battery_score = TARGET_BASE_DIMS["续航"] + sigmoid_scaled(battery, x0=BASE_BATTERY, k=0.002)
    performance = TARGET_BASE_DIMS["性能"] + sigmoid_scaled(cpu_index, x0=BASE_CPU, k=0.08)

    thermal = (
        TARGET_BASE_DIMS["发热控制"]
        - 0.010 * (cpu_index - BASE_CPU)
        - 0.00005 * (battery - BASE_BATTERY)
    )

    return {
        "外观": clip_score(appearance),
        "屏幕": clip_score(screen),
        "摄像": clip_score(camera),
        "续航": clip_score(battery_score),
        "性能": clip_score(performance),
        "发热控制": clip_score(thermal),
    }


def calc_unit_cost(
    battery: float,
    camera_mp: float,
    cpu_index: float,
    refresh_rate: float
) -> float:
    battery_cost = 0.18 * (battery - BASE_BATTERY)
    camera_cost = 6.5 * (camera_mp - BASE_CAMERA)
    cpu_cost = 14.0 * (cpu_index - BASE_CPU)
    refresh_cost = 7.5 * (refresh_rate - BASE_REFRESH)

    total_cost = BASE_UNIT_COST + battery_cost + camera_cost + cpu_cost + refresh_cost
    return float(max(total_cost, 2500.0))


def evaluate_solution(
    battery: float,
    camera_mp: float,
    cpu_index: float,
    refresh_rate: float,
    price: float,
    competitor_price: float = BASE_COMPETITOR_PRICE,
    market_size: float = BASE_MARKET_SIZE
) -> dict:
    dims = design_to_dimensions(battery, camera_mp, cpu_index, refresh_rate)

    overall_rating = predict_overall_rating(
        appearance=dims["外观"],
        screen=dims["屏幕"],
        camera=dims["摄像"],
        battery=dims["续航"],
        performance=dims["性能"],
        thermal=dims["发热控制"]
    )

    product_score = rating_to_product_score(overall_rating)
    unit_cost = calc_unit_cost(battery, camera_mp, cpu_index, refresh_rate)

    sales = predict_sales(
        price=price,
        competitor_price=competitor_price,
        product_score=product_score,
        market_size=market_size
    )

    profit = (price - unit_cost) * sales

    return {
        "电池容量(mAh)": battery,
        "主摄像素(MP)": camera_mp,
        "处理器性能指数": cpu_index,
        "屏幕刷新率(Hz)": refresh_rate,
        "定价(元)": price,
        "单位成本(元)": unit_cost,
        "外观满意度": dims["外观"],
        "屏幕满意度": dims["屏幕"],
        "摄像满意度": dims["摄像"],
        "续航满意度": dims["续航"],
        "性能满意度": dims["性能"],
        "发热控制满意度": dims["发热控制"],
        "预测总体评分(1-5星)": overall_rating,
        "产品综合得分(0-100)": product_score,
        "预测销量(万台)": sales,
        "总利润(万元)": profit
    }


# =========================
# 6. 基准方案评估
# =========================
base_result = evaluate_solution(
    battery=BASE_BATTERY,
    camera_mp=BASE_CAMERA,
    cpu_index=BASE_CPU,
    refresh_rate=BASE_REFRESH,
    price=BASE_PRICE
)

print("\n" + "=" * 70)
print("原产品基准方案评估结果（校准后）")
print("=" * 70)
for k, v in base_result.items():
    if isinstance(v, (int, float, np.floating)):
        print(f"{k}: {v:.4f}")


# =========================
# 7. 网格搜索最优方案
# =========================
battery_grid = [4500, 4800, 5000, 5200, 5500, 5800]
camera_grid = [50, 64, 80, 100]
cpu_grid = [85, 88, 92, 96]
refresh_grid = [120, 144]
price_grid = np.arange(3499, 6201, 100)

total_search_count = len(battery_grid) * len(camera_grid) * len(cpu_grid) * len(refresh_grid) * len(price_grid)
print(f"\n开始枚举搜索最优方案，总搜索量：{total_search_count} 个参数组合")
print("正在搜索中，请稍候...")

results = []
for battery, camera_mp, cpu_index, refresh_rate, price in product(
    battery_grid, camera_grid, cpu_grid, refresh_grid, price_grid
):
    unit_cost = calc_unit_cost(battery, camera_mp, cpu_index, refresh_rate)

    # 约束1：售价必须高于成本并保留最低毛利
    if price <= unit_cost + 250:
        continue

    # 约束2：旗舰机型不允许处理器降档
    if cpu_index < BASE_CPU:
        continue

    # 约束3：不允许核心配置低于原产品
    if camera_mp < BASE_CAMERA:
        continue
    if refresh_rate < BASE_REFRESH:
        continue

    # 约束4：高端配置不匹配过低价格
    if cpu_index >= 92 and price < 4400:
        continue
    if camera_mp >= 80 and price < 4300:
        continue
    if refresh_rate == 144 and price < 4200:
        continue

    result = evaluate_solution(
        battery=battery,
        camera_mp=camera_mp,
        cpu_index=cpu_index,
        refresh_rate=refresh_rate,
        price=price
    )

    # 约束5：优化方案的用户总体评分不得低于原产品
    if result["预测总体评分(1-5星)"] < base_result["预测总体评分(1-5星)"]:
        continue

    results.append(result)

if len(results) == 0:
    raise RuntimeError("未找到可行解，请放宽约束或调整参数网格。")

results_df = pd.DataFrame(results).sort_values("总利润(万元)", ascending=False).reset_index(drop=True)
best_result = results_df.iloc[0].to_dict()

print("\n" + "=" * 70)
print("✅ 搜索完成！利润最大化最优方案结果")
print("=" * 70)
for k, v in best_result.items():
    if isinstance(v, (int, float, np.floating)):
        print(f"{k}: {v:.4f}")


# =========================
# 8. 基准方案 vs 最优方案
# =========================
comparison_df = pd.DataFrame({
    "指标": [
        "电池容量(mAh)", "主摄像素(MP)", "处理器性能指数", "屏幕刷新率(Hz)", "定价(元)",
        "单位成本(元)", "预测总体评分(1-5星)", "产品综合得分(0-100)", "预测销量(万台)", "总利润(万元)"
    ],
    "原产品基准方案": [
        base_result["电池容量(mAh)"],
        base_result["主摄像素(MP)"],
        base_result["处理器性能指数"],
        base_result["屏幕刷新率(Hz)"],
        base_result["定价(元)"],
        base_result["单位成本(元)"],
        base_result["预测总体评分(1-5星)"],
        base_result["产品综合得分(0-100)"],
        base_result["预测销量(万台)"],
        base_result["总利润(万元)"],
    ],
    "优化后最优方案": [
        best_result["电池容量(mAh)"],
        best_result["主摄像素(MP)"],
        best_result["处理器性能指数"],
        best_result["屏幕刷新率(Hz)"],
        best_result["定价(元)"],
        best_result["单位成本(元)"],
        best_result["预测总体评分(1-5星)"],
        best_result["产品综合得分(0-100)"],
        best_result["预测销量(万台)"],
        best_result["总利润(万元)"],
    ],
})

comparison_df["提升幅度(%)"] = (
    (comparison_df["优化后最优方案"] - comparison_df["原产品基准方案"])
    / comparison_df["原产品基准方案"] * 100
).replace([np.inf, -np.inf], np.nan)

profit_improve_pct = (
    (best_result["总利润(万元)"] - base_result["总利润(万元)"])
    / base_result["总利润(万元)"] * 100
)
rating_improve_pct = (
    (best_result["预测总体评分(1-5星)"] - base_result["预测总体评分(1-5星)"])
    / base_result["预测总体评分(1-5星)"] * 100
)

print("\n" + "=" * 70)
print("原产品基准方案 vs 优化后最优方案 核心对比")
print("=" * 70)
print(comparison_df.round(4))


# =========================
# 9. 核心参数敏感性分析
# =========================
print("\n" + "=" * 70)
print("核心参数敏感性分析（固定其他参数为最优值）")
print("=" * 70)

battery_sensitivity = []
for b in battery_grid:
    res = evaluate_solution(
        battery=b,
        camera_mp=best_result["主摄像素(MP)"],
        cpu_index=best_result["处理器性能指数"],
        refresh_rate=best_result["屏幕刷新率(Hz)"],
        price=best_result["定价(元)"]
    )
    battery_sensitivity.append(res)
battery_sensitivity_df = pd.DataFrame(battery_sensitivity)
print("\n电池容量敏感性分析结果：")
print(battery_sensitivity_df[["电池容量(mAh)", "预测总体评分(1-5星)", "预测销量(万台)", "总利润(万元)"]].round(4))

cpu_sensitivity = []
for c in cpu_grid:
    res = evaluate_solution(
        battery=best_result["电池容量(mAh)"],
        camera_mp=best_result["主摄像素(MP)"],
        cpu_index=c,
        refresh_rate=best_result["屏幕刷新率(Hz)"],
        price=best_result["定价(元)"]
    )
    cpu_sensitivity.append(res)
cpu_sensitivity_df = pd.DataFrame(cpu_sensitivity)
print("\n处理器性能敏感性分析结果：")
print(cpu_sensitivity_df[["处理器性能指数", "预测总体评分(1-5星)", "预测销量(万台)", "总利润(万元)"]].round(4))


# =========================
# 10. 导出表格
# =========================
review_coef_df.to_excel(output_dir / "问题3_总体评分回归系数表.xlsx", index=False)
sales_coef_df.to_excel(output_dir / "问题3_销量回归系数表.xlsx", index=False)
results_df.to_excel(output_dir / "问题3_全部可行方案结果汇总.xlsx", index=False)
results_df.head(30).to_excel(output_dir / "问题3_利润最高前30个最优方案.xlsx", index=False)
comparison_df.to_excel(output_dir / "问题3_基准方案与最优方案对比表.xlsx", index=False)
pd.DataFrame([best_result]).to_excel(output_dir / "问题3_利润最大化最优方案明细.xlsx", index=False)
battery_sensitivity_df.to_excel(output_dir / "问题3_电池容量敏感性分析表.xlsx", index=False)
cpu_sensitivity_df.to_excel(output_dir / "问题3_处理器性能敏感性分析表.xlsx", index=False)

# 前10方案明细表导出
top10_detail_df = results_df.head(10).copy().reset_index(drop=True)
top10_detail_df.insert(0, "方案编号", [f"方案{i+1}" for i in range(len(top10_detail_df))])
top10_detail_df.to_excel(output_dir / "问题3_利润最高前10方案明细表.xlsx", index=False)

print("\n利润最高前10个方案明细如下：")
print(
    top10_detail_df[
        [
            "方案编号",
            "电池容量(mAh)",
            "主摄像素(MP)",
            "处理器性能指数",
            "屏幕刷新率(Hz)",
            "定价(元)",
            "预测总体评分(1-5星)",
            "预测销量(万台)",
            "总利润(万元)",
        ]
    ].round(4)
)


# =========================
# 11. 作图
# =========================

# 图1：利润最高前10方案
top10_df = results_df.head(10).copy().reset_index(drop=True)
top10_df["方案编号"] = [f"方案{i+1}" for i in range(len(top10_df))]

top10_df["方案说明"] = top10_df.apply(
    lambda row: f"{row['方案编号']} | "
                f"{int(row['电池容量(mAh)'])}mAh, "
                f"{int(row['主摄像素(MP)'])}MP, "
                f"CPU{int(row['处理器性能指数'])}, "
                f"{int(row['屏幕刷新率(Hz)'])}Hz, "
                f"{row['定价(元)']:.0f}元",
    axis=1
)

plot_df = top10_df.sort_values("总利润(万元)", ascending=True).copy()

fig, ax = plt.subplots(figsize=(13, 7.2))
bars = ax.barh(
    plot_df["方案说明"],
    plot_df["总利润(万元)"],
    color=MAIN_COLOR,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.88
)

ax.set_title("利润最高前10个方案（参数组合版）", fontsize=16, weight="bold", pad=12)
ax.set_xlabel("总利润（万元）", fontsize=12)
ax.set_ylabel("候选方案", fontsize=12)

xmin = plot_df["总利润(万元)"].min()
xmax = plot_df["总利润(万元)"].max()
margin = (xmax - xmin) * 0.25 if xmax > xmin else 100
ax.set_xlim(xmin - margin, xmax + margin)

ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID_COLOR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + margin * 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.1f}",
        va="center",
        ha="left",
        fontsize=10
    )

save_fig(fig, "问题3_利润最高前10方案_参数组合横向图")
plt.close()


# 图2：前10方案相对最优利润差额图
gap_df = top10_df.copy()
best_profit_top10 = gap_df["总利润(万元)"].max()
gap_df["与最优方案利润差额(万元)"] = best_profit_top10 - gap_df["总利润(万元)"]
gap_df = gap_df.sort_values("与最优方案利润差额(万元)", ascending=True)

fig, ax = plt.subplots(figsize=(10.5, 6.0))
bars = ax.barh(
    gap_df["方案编号"],
    gap_df["与最优方案利润差额(万元)"],
    color=ACCENT_COLOR,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.88
)

ax.set_title("前10方案相对最优利润差额", fontsize=16, weight="bold", pad=12)
ax.set_xlabel("与最优方案利润差额（万元）", fontsize=12)
ax.set_ylabel("候选方案", fontsize=12)

ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID_COLOR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

max_gap = gap_df["与最优方案利润差额(万元)"].max()
offset = max_gap * 0.02 if max_gap > 0 else 1.0

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.1f}",
        va="center",
        ha="left",
        fontsize=10
    )

save_fig(fig, "问题3_前10方案相对最优利润差额图")
plt.close()


# 图3：原方案与优化方案利润对比图
fig, ax = plt.subplots(figsize=(7.8, 5.6))
profit_compare_df = pd.DataFrame({
    "方案": ["原产品基准方案", "优化后最优方案"],
    "总利润(万元)": [base_result["总利润(万元)"], best_result["总利润(万元)"]]
})

bars = ax.bar(
    profit_compare_df["方案"],
    profit_compare_df["总利润(万元)"],
    color=[SECOND_COLOR, THIRD_COLOR],
    edgecolor="black",
    linewidth=0.7,
    alpha=0.85
)
ax.set_title("原方案与优化方案利润对比", fontsize=15, weight="bold", pad=12)
ax.set_ylabel("总利润（万元）", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=10)

ax.text(
    0.98, 0.94,
    f"利润提升幅度 = {profit_improve_pct:.2f}%\n用户评分提升幅度 = {rating_improve_pct:.2f}%",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB")
)
save_fig(fig, "问题3_基准方案与最优方案利润对比图")
plt.close()


# 图4：定价与利润关系散点图
fig, ax = plt.subplots(figsize=(8.2, 5.8))
sample_df = results_df.sample(min(500, len(results_df)), random_state=42)
sns.scatterplot(
    data=sample_df,
    x="定价(元)",
    y="总利润(万元)",
    alpha=0.65,
    s=55,
    color=ACCENT_COLOR,
    edgecolor="white",
    ax=ax
)
ax.scatter(
    best_result["定价(元)"],
    best_result["总利润(万元)"],
    color=THIRD_COLOR,
    s=150,
    zorder=10,
    label="利润最大化最优定价"
)
ax.set_title("定价与利润关系散点图", fontsize=15, weight="bold", pad=12)
ax.set_xlabel("产品定价（元）", fontsize=11)
ax.set_ylabel("总利润（万元）", fontsize=11)
ax.legend(frameon=False)
ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save_fig(fig, "问题3_定价与利润关系散点图")
plt.close()


# 图5：电池容量敏感性分析图
fig, ax = plt.subplots(figsize=(8.2, 5.6))
ax.plot(
    battery_sensitivity_df["电池容量(mAh)"],
    battery_sensitivity_df["总利润(万元)"],
    color=MAIN_COLOR,
    linewidth=2.5,
    marker="o",
    markersize=7,
    label="总利润"
)
ax2 = ax.twinx()
ax2.plot(
    battery_sensitivity_df["电池容量(mAh)"],
    battery_sensitivity_df["预测总体评分(1-5星)"],
    color=SECOND_COLOR,
    linewidth=2.0,
    linestyle="--",
    marker="s",
    markersize=6,
    label="用户总体评分"
)
ax.set_title("电池容量敏感性分析", fontsize=15, weight="bold", pad=12)
ax.set_xlabel("电池容量（mAh）", fontsize=11)
ax.set_ylabel("总利润（万元）", fontsize=11, color=MAIN_COLOR)
ax2.set_ylabel("用户总体评分（1-5星）", fontsize=11, color=SECOND_COLOR)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)
ax.spines["top"].set_visible(False)
save_fig(fig, "问题3_电池容量敏感性分析图")
plt.close()


# =========================
# 12. 最终结论摘要
# =========================
print("\n" + "=" * 70)
print("=" * 70)
print("1. 已建立完整的“设计参数—维度满意度—总体评分—产品得分—销量—利润”全链路参数化优化模型。")
print(f"2. 原产品基准方案：定价{base_result['定价(元)']:.2f}元，对应总利润{base_result['总利润(万元)']:.2f}万元，用户总体评分{base_result['预测总体评分(1-5星)']:.4f}星。")
print(f"3. 优化后利润最大化最优方案：定价{best_result['定价(元)']:.2f}元，对应总利润{best_result['总利润(万元)']:.2f}万元，用户总体评分{best_result['预测总体评分(1-5星)']:.4f}星。")
print(f"4. 优化后核心提升：利润提升幅度{profit_improve_pct:.2f}%，用户总体评分提升幅度{rating_improve_pct:.2f}%。")
print(f"5. 最优硬件设计参数组合：{best_result['电池容量(mAh)']:.0f}mAh电池、{best_result['主摄像素(MP)']:.0f}MP主摄、处理器性能指数{best_result['处理器性能指数']:.0f}、{best_result['屏幕刷新率(Hz)']:.0f}Hz屏幕刷新率。")
print(f"6. 所有表格与图片已保存到：{output_dir.resolve()}")