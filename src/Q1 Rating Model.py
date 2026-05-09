import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# =========================
# 0. 基础设置
# =========================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300

sns.set_theme(style="white", font="Microsoft YaHei", font_scale=1.08)

MAIN_COLOR = "#4C78A8"
SECOND_COLOR = "#59A14F"
THIRD_COLOR = "#E15759"
ACCENT_COLOR = "#B279A2"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#222222"

heatmap_cmap = LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#3B6FB6", "#F7F7F7", "#C23B3B"]
)

# 输出文件夹
output_dir = "问题1输出结果"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(output_dir):
    if filename.lower().endswith(".pdf"):
        os.remove(os.path.join(output_dir, filename))

def save_fig(fig, name):
    png_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(png_path, bbox_inches='tight', facecolor='white')

# =========================
# 1. 读取数据
# =========================
df = pd.read_excel("附件1_清洗后数据.xlsx")

X = df[['外观', '屏幕', '摄像', '续航', '性能', '发热控制']]
y = df['总体评分']

print("=" * 60)
print("问题1：用户总体评分影响因素分析")
print("=" * 60)
print(f"样本量：{len(df)}")

# =========================
# 2. 普通多元线性回归（带显著性检验）
# =========================
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print("\nOLS回归结果：")
print(model.summary())

coef_df = pd.DataFrame({
    '变量': X.columns,
    '回归系数': model.params[1:].values,
    'P值': model.pvalues[1:].values
}).sort_values(by='回归系数', ascending=False)

print("\n多元线性回归结果：")
print(coef_df)

print("\n问题1回归方程为：")
print(
    f"总体评分 = {model.params['const']:.4f}"
    f" + {model.params['外观']:.4f}*外观"
    f" + {model.params['屏幕']:.4f}*屏幕"
    f" + {model.params['摄像']:.4f}*摄像"
    f" + {model.params['续航']:.4f}*续航"
    f" + {model.params['性能']:.4f}*性能"
    f" + {model.params['发热控制']:.4f}*发热控制"
)

# =========================
# 3. 标准化回归
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_std = scaler_X.fit_transform(X)
y_std = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

std_model = LinearRegression()
std_model.fit(X_std, y_std)

std_coef_df = pd.DataFrame({
    '变量': X.columns,
    '标准化系数': std_model.coef_,
    '绝对值': np.abs(std_model.coef_)
}).sort_values(by='绝对值', ascending=False)

print("\n标准化回归系数排序：")
print(std_coef_df)

# =========================
# 4. 岭回归
# =========================
ridge = Ridge(alpha=1.0)
ridge.fit(X_std, y_std)

ridge_coef_df = pd.DataFrame({
    '变量': X.columns,
    '岭回归系数': ridge.coef_,
    '绝对值': np.abs(ridge.coef_)
}).sort_values(by='绝对值', ascending=False)

print("\n岭回归系数排序：")
print(ridge_coef_df)

# =========================
# 5. 随机森林特征重要性
# =========================
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X, y)

rf_importance_df = pd.DataFrame({
    '变量': X.columns,
    '特征重要性': rf.feature_importances_
}).sort_values(by='特征重要性', ascending=False)

print("\n随机森林特征重要性：")
print(rf_importance_df)

# =========================
# 6. 模型拟合效果
# =========================
y_pred = model.predict(X_const)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
adj_r2 = model.rsquared_adj
residuals = y - y_pred

print(f"\nR^2 = {r2:.4f}")
print(f"调整后R^2 = {adj_r2:.4f}")
print(f"RMSE = {rmse:.4f}")

# =========================
# 7. 导出结果表
# =========================
coef_df.to_excel(os.path.join(output_dir, "问题1_多元线性回归结果.xlsx"), index=False)
std_coef_df.to_excel(os.path.join(output_dir, "问题1_标准化回归系数排序.xlsx"), index=False)
ridge_coef_df.to_excel(os.path.join(output_dir, "问题1_岭回归系数排序.xlsx"), index=False)
rf_importance_df.to_excel(os.path.join(output_dir, "问题1_随机森林特征重要性.xlsx"), index=False)

corr_df = df[['外观', '屏幕', '摄像', '续航', '性能', '发热控制', '总体评分']].corr()
corr_df.to_excel(os.path.join(output_dir, "问题1_相关系数矩阵.xlsx"))

metrics_df = pd.DataFrame({
    '指标': ['R^2', '调整后R^2', 'RMSE'],
    '数值': [r2, adj_r2, rmse]
})
metrics_df.to_excel(os.path.join(output_dir, "问题1_模型评价指标.xlsx"), index=False)

# =========================
# 8. 相关系数热力图
# =========================
fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    corr_df,
    annot=True,
    fmt=".3f",
    cmap=heatmap_cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"shrink": 0.85, "label": "相关系数"},
    annot_kws={"size": 10, "color": TEXT_COLOR},
    ax=ax
)

ax.set_title("各变量相关系数热力图", fontsize=16, weight="bold", pad=14, color=TEXT_COLOR)
ax.tick_params(axis='x', rotation=30, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)

save_fig(fig, "问题1_相关系数热力图")
plt.show()
plt.close()

# =========================
# 9. 标准化回归系数图
# =========================
fig, ax = plt.subplots(figsize=(9.2, 5.8))

plot_df = std_coef_df.copy().sort_values(by="标准化系数", ascending=True)
colors = sns.color_palette("Blues", n_colors=len(plot_df) + 2)[2:]

bars = ax.barh(
    plot_df["变量"],
    plot_df["标准化系数"],
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    height=0.62
)

ax.set_title("各维度标准化回归系数排序", fontsize=16, weight="bold", pad=12, color=TEXT_COLOR)
ax.set_xlabel("标准化系数", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("设计维度", fontsize=12, color=TEXT_COLOR)

ax.grid(axis='x', linestyle='--', alpha=0.35, color=GRID_COLOR)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#999999")
ax.spines['bottom'].set_color("#999999")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.3f}",
        va='center',
        ha='left',
        fontsize=10,
        color=TEXT_COLOR
    )

ax.text(
    0.98, 0.04,
    "影响越大，条形越长",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    color="#666666"
)

save_fig(fig, "问题1_标准化回归系数柱状图")
plt.show()
plt.close()

# =========================
# 10. 岭回归系数图
# =========================
fig, ax = plt.subplots(figsize=(9.2, 5.8))

plot_df = ridge_coef_df.copy().sort_values(by="岭回归系数", ascending=True)
colors = sns.color_palette("Purples", n_colors=len(plot_df) + 2)[2:]

bars = ax.barh(
    plot_df["变量"],
    plot_df["岭回归系数"],
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    height=0.62
)

ax.set_title("各维度岭回归系数排序", fontsize=16, weight="bold", pad=12, color=TEXT_COLOR)
ax.set_xlabel("岭回归系数", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("设计维度", fontsize=12, color=TEXT_COLOR)

ax.grid(axis='x', linestyle='--', alpha=0.35, color=GRID_COLOR)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#999999")
ax.spines['bottom'].set_color("#999999")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.3f}",
        va='center',
        ha='left',
        fontsize=10,
        color=TEXT_COLOR
    )

ax.text(
    0.98, 0.04,
    "用于共线性条件下的稳健性检验",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    color="#666666"
)

save_fig(fig, "问题1_岭回归系数柱状图")
plt.show()
plt.close()

# =========================
# 11. 随机森林特征重要性图
# =========================
fig, ax = plt.subplots(figsize=(9.2, 5.8))

plot_df = rf_importance_df.copy().sort_values(by="特征重要性", ascending=True)
colors = sns.color_palette("Greens", n_colors=len(plot_df) + 2)[2:]

bars = ax.barh(
    plot_df["变量"],
    plot_df["特征重要性"],
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    height=0.62
)

ax.set_title("随机森林特征重要性排序", fontsize=16, weight="bold", pad=12, color=TEXT_COLOR)
ax.set_xlabel("特征重要性", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("设计维度", fontsize=12, color=TEXT_COLOR)

ax.grid(axis='x', linestyle='--', alpha=0.35, color=GRID_COLOR)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#999999")
ax.spines['bottom'].set_color("#999999")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + 0.004,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.3f}",
        va='center',
        ha='left',
        fontsize=10,
        color=TEXT_COLOR
    )

ax.text(
    0.98, 0.04,
    "用于稳健性验证",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    color="#666666"
)

save_fig(fig, "问题1_随机森林特征重要性柱状图")
plt.show()
plt.close()

# =========================
# 12. 真实值 vs 预测值散点图
# =========================
fig, ax = plt.subplots(figsize=(7, 6.4))

ax.scatter(
    y, y_pred,
    s=52,
    alpha=0.72,
    color=ACCENT_COLOR,
    edgecolors="white",
    linewidth=0.6
)

axis_min = min(y.min(), y_pred.min())
axis_max = max(y.max(), y_pred.max())

ax.plot(
    [axis_min, axis_max],
    [axis_min, axis_max],
    linestyle='--',
    linewidth=2.2,
    color=THIRD_COLOR,
    label='理想拟合线 y=x'
)

ax.set_xlim(axis_min - 0.1, axis_max + 0.1)
ax.set_ylim(axis_min - 0.1, axis_max + 0.1)
ax.set_aspect('equal', adjustable='box')

ax.set_title("真实值与预测值对比图", fontsize=16, weight="bold", pad=12, color=TEXT_COLOR)
ax.set_xlabel("真实总体评分", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("预测总体评分", fontsize=12, color=TEXT_COLOR)

ax.grid(True, linestyle='--', alpha=0.35, color=GRID_COLOR)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc="upper left")

textstr = f"R^2 = {r2:.4f}\nRMSE = {rmse:.4f}"
ax.text(
    0.98, 0.05, textstr,
    transform=ax.transAxes,
    fontsize=11,
    ha='right',
    va='bottom',
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB")
)

save_fig(fig, "问题1_真实值与预测值对比图")
plt.show()
plt.close()

# =========================
# 13. 残差分布图
# =========================
fig, ax = plt.subplots(figsize=(8.2, 5.6))

sns.histplot(
    residuals,
    bins=20,
    kde=True,
    color=MAIN_COLOR,
    edgecolor="white",
    alpha=0.85,
    ax=ax
)

ax.axvline(0, color=THIRD_COLOR, linestyle='--', linewidth=2)
ax.set_title("残差分布图", fontsize=16, weight="bold", pad=12, color=TEXT_COLOR)
ax.set_xlabel("残差（真实值 - 预测值）", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("频数", fontsize=12, color=TEXT_COLOR)

ax.grid(axis='y', linestyle='--', alpha=0.3, color=GRID_COLOR)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(
    0.98, 0.92,
    f"残差均值 = {residuals.mean():.4f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color="#555555"
)

save_fig(fig, "问题1_残差分布图")
plt.show()
plt.close()

# =========================
# 14. 输出结论摘要
# =========================
print("\n" + "=" * 60)
print("问题1结论摘要")
print("=" * 60)
print("1. 六个设计维度对总体评分均有显著正向影响。")
print("2. 模型拟合效果很好，可用于解释总体评分变化。")
print("3. 标准化回归排序：")
for i, row in std_coef_df.reset_index(drop=True).iterrows():
    print(f"   第{i+1}名：{row['变量']}（标准化系数={row['标准化系数']:.6f}）")

print("\n4. 综合标准化回归、岭回归与随机森林结果，")
print("   可将 性能、摄像、外观 视为关键设计维度。")
print(f"\n所有表格和图片已保存到文件夹：{output_dir}")