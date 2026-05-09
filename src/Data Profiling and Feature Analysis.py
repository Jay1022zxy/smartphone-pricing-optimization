from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# 0. 全局设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei", font_scale=1.0)

OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 配置区
# =========================
@dataclass(frozen=True)
class AttachmentConfig:
    title: str
    patterns: tuple[str, ...]
    read_kwargs: dict
    output_name: str


ATTACHMENT_1 = AttachmentConfig(
    title="附件1：用户评分数据",
    patterns=("*user_reviews_2decimal.xlsx", "*reviews*.xlsx"),
    read_kwargs={"skiprows": [1]},
    output_name="附件1_清洗后数据.xlsx",
)

ATTACHMENT_2 = AttachmentConfig(
    title="附件2：销量定价数据",
    patterns=("*sales_data.xlsx", "*sales*.xlsx"),
    read_kwargs={"nrows": 32},
    output_name="附件2_清洗后数据.xlsx",
)

REVIEW_RENAME_MAP = {
    "rating": "总体评分",
    "appearance": "外观",
    "screen": "屏幕",
    "camera": "摄像",
    "battery": "续航",
    "performance": "性能",
    "thermal": "发热控制",
}

REVIEW_DIM_COLS = ["外观", "屏幕", "摄像", "续航", "性能", "发热控制"]
REVIEW_TARGET_COL = "总体评分"

SALES_NUMERIC_COLS = [
    "price",
    "competitor_price",
    "product_score",
    "market_size",
    "simulated_sales",
]


# =========================
# 2. 通用工具函数
# =========================
def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def find_excel_file(*patterns: str) -> Path:
    for pattern in patterns:
        matches = sorted(Path(".").glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"未找到匹配文件: {patterns}")


def load_attachment(config: AttachmentConfig) -> pd.DataFrame:
    file_path = find_excel_file(*config.patterns)
    print(f"读取文件: {file_path}")
    return pd.read_excel(file_path, **config.read_kwargs)


def validate_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"缺少必要字段: {missing_cols}")


def report_frame_info(label: str, df: pd.DataFrame) -> None:
    print(f"{label}数据行列数: {df.shape}")
    print(f"\n{label}字段类型:")
    print(df.dtypes)


def report_missing_rate(label: str, df: pd.DataFrame) -> None:
    missing_rate = (df.isnull().sum() / len(df) * 100).round(4)
    print(f"\n{label}各字段缺失率(%):")
    print(missing_rate)


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    df.to_excel(ensure_output_dir() / filename, index=False)


def remove_outliers_by_masks(
    df: pd.DataFrame,
    masks: dict[str, pd.Series]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    error_frames = {name: df[mask].copy() for name, mask in masks.items()}
    combined_error_index = pd.Index([])
    for error_df in error_frames.values():
        combined_error_index = combined_error_index.union(error_df.index)

    clean_df = df.drop(index=combined_error_index).reset_index(drop=True)
    return clean_df, error_frames


def export_error_frames(error_frames: dict[str, pd.DataFrame], prefix: str) -> None:
    for name, error_df in error_frames.items():
        if not error_df.empty:
            save_dataframe(error_df, f"{prefix}_{name}.xlsx")


def calculate_vif_safe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    vif_df = df[columns].copy()

    vif_df = vif_df.dropna()

    valid_cols = [col for col in columns if vif_df[col].nunique(dropna=True) > 1]
    dropped_cols = [col for col in columns if col not in valid_cols]

    if len(valid_cols) < 2:
        result = pd.DataFrame({"变量": columns, "VIF值": [np.nan] * len(columns)})
        return result

    values = vif_df[valid_cols].values
    vif_values = [variance_inflation_factor(values, i) for i in range(len(valid_cols))]
    result = pd.DataFrame({"变量": valid_cols, "VIF值": vif_values})

    if dropped_cols:
        dropped_df = pd.DataFrame({"变量": dropped_cols, "VIF值": [np.nan] * len(dropped_cols)})
        result = pd.concat([result, dropped_df], ignore_index=True)

    return result.sort_values("VIF值", ascending=False, na_position="last").reset_index(drop=True)


def fill_missing_by_group_then_global(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str
) -> pd.DataFrame:
    df_filled = df.copy()

    for col in columns:
        if df_filled[col].isnull().sum() > 0:
            group_mean = df_filled.groupby(group_col)[col].transform("mean")
            df_filled[col] = df_filled[col].fillna(group_mean)

            if df_filled[col].isnull().sum() > 0:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    return df_filled


def plot_distribution_and_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    prefix: str,
    label_map: dict[str, str] | None = None,
) -> None:
    output_dir = ensure_output_dir()

    for col in columns:
        display_name = label_map.get(col, col) if label_map else col

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"{display_name}分布图")
        axes[0].set_xlabel(display_name)

        sns.boxplot(x=df[col].dropna(), ax=axes[1], color="#55A868")
        axes[1].set_title(f"{display_name}箱线图")
        axes[1].set_xlabel(display_name)

        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{display_name}_分布图_箱线图.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_heatmap(corr_df: pd.DataFrame, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ensure_output_dir() / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sales_relationships(df: pd.DataFrame, prefix: str) -> None:
    output_dir = ensure_output_dir()

    plot_specs = [
        ("price", "产品定价", "产品定价与销量关系图"),
        ("price_gap", "相对竞品价差 Δp = p - pc", "相对竞品价差与销量关系图"),
        ("product_score", "产品评分", "产品评分与销量关系图"),
    ]

    for col, xlabel, title in plot_specs:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        sns.regplot(
            data=df,
            x=col,
            y="simulated_sales",
            ax=ax,
            scatter_kws={"s": 70, "alpha": 0.8, "color": "#4C72B0"},
            line_kws={"color": "#C44E52", "linewidth": 2.5},
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("销量（万台）")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{title}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# =========================
# 3. 附件1处理
# =========================
def process_attachment_1() -> pd.DataFrame:
    print_section("开始处理 附件1：用户评分数据")

    df_raw = load_attachment(ATTACHMENT_1).rename(columns=REVIEW_RENAME_MAP)
    validate_columns(df_raw, ["review_id", "comment", REVIEW_TARGET_COL, *REVIEW_DIM_COLS])

    report_frame_info("附件1", df_raw)

    # 去重
    df_dedup = df_raw.drop_duplicates(subset=["review_id"]).copy()
    print(f"\n去重后剩余样本数: {len(df_dedup)}")

    # 异常值掩码
    masks = {
        "总体评分异常": (df_dedup[REVIEW_TARGET_COL] < 1) | (df_dedup[REVIEW_TARGET_COL] > 5),
        "维度得分异常": (df_dedup[REVIEW_DIM_COLS] < -1).any(axis=1) | (df_dedup[REVIEW_DIM_COLS] > 1).any(axis=1),
    }

    df_clean, error_frames = remove_outliers_by_masks(df_dedup, masks)

    print(f"\n总体评分异常样本数: {len(error_frames['总体评分异常'])}")
    print(f"维度得分异常样本数: {len(error_frames['维度得分异常'])}")
    print(f"异常值处理后剩余样本数: {len(df_clean)}")

    report_missing_rate("附件1", df_clean)
    print("异常值说明：本处按业务阈值筛除异常记录；如需补充统计口径，可再结合 IQR 或 3σ 方法。")

    df_clean = fill_missing_by_group_then_global(
        df_clean,
        columns=[*REVIEW_DIM_COLS, REVIEW_TARGET_COL],
        group_col=REVIEW_TARGET_COL
    )

    print("\n附件1核心字段描述性统计：")
    print(df_clean[REVIEW_DIM_COLS + [REVIEW_TARGET_COL]].describe().round(3))

    # Spearman相关
    corr_df = df_clean[REVIEW_DIM_COLS + [REVIEW_TARGET_COL]].corr(method="spearman")
    print("\n各维度与总体评分的 Spearman 相关系数（从高到低）：")
    print(corr_df[REVIEW_TARGET_COL].sort_values(ascending=False))

    plot_heatmap(
        corr_df,
        title="附件1：各维度满意度与总体评分相关性热力图",
        filename="附件1_相关性热力图.png"
    )

    plot_distribution_and_boxplots(
        df_clean,
        REVIEW_DIM_COLS + [REVIEW_TARGET_COL],
        "附件1"
    )

    vif_df = calculate_vif_safe(df_clean, REVIEW_DIM_COLS)
    print("\n各维度方差膨胀因子 VIF：")
    print(vif_df.round(4))

    # 导出
    save_dataframe(df_clean, ATTACHMENT_1.output_name)
    save_dataframe(vif_df, "附件1_VIF结果.xlsx")
    export_error_frames(error_frames, "附件1_异常样本")

    print("\n附件1处理完成，干净数据已导出。")
    return df_clean


# =========================
# 4. 附件2处理
# =========================
def process_attachment_2() -> pd.DataFrame:
    print_section("开始处理 附件2：销量定价数据")

    df_raw = load_attachment(ATTACHMENT_2)
    validate_columns(df_raw, SALES_NUMERIC_COLS)

    df_raw = df_raw.copy()
    df_raw[SALES_NUMERIC_COLS] = df_raw[SALES_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    df_raw["price_gap"] = df_raw["price"] - df_raw["competitor_price"]

    report_frame_info("附件2", df_raw)

    # 去重
    df_dedup = df_raw.drop_duplicates().copy()
    print(f"\n去重后剩余样本数: {len(df_dedup)}")

    # 异常值掩码：加入 competitor_price 和 market_size 的业务检查
    masks = {
        "本机定价异常": (df_dedup["price"] < 3000) | (df_dedup["price"] > 6000),
        "竞品定价异常": (df_dedup["competitor_price"] < 3000) | (df_dedup["competitor_price"] > 6000),
        "产品评分异常": (df_dedup["product_score"] < 0) | (df_dedup["product_score"] > 100),
        "市场容量异常": df_dedup["market_size"] <= 0,
        "销量异常": df_dedup["simulated_sales"] < 0,
    }

    df_clean, error_frames = remove_outliers_by_masks(df_dedup, masks)

    print(f"\n本机定价异常样本数: {len(error_frames['本机定价异常'])}")
    print(f"竞品定价异常样本数: {len(error_frames['竞品定价异常'])}")
    print(f"产品评分异常样本数: {len(error_frames['产品评分异常'])}")
    print(f"市场容量异常样本数: {len(error_frames['市场容量异常'])}")
    print(f"销量异常样本数: {len(error_frames['销量异常'])}")
    print(f"异常值处理后剩余样本数: {len(df_clean)}")

    report_missing_rate("附件2", df_clean)
    print("异常值说明：本处按业务阈值筛除异常记录；如需补充统计口径，可再结合 IQR 或 3σ 方法。")

    # 若有缺失，直接用中位数补
    numeric_fill_cols = SALES_NUMERIC_COLS + ["price_gap"]
    for col in numeric_fill_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    print("\n附件2核心字段描述性统计：")
    print(df_clean[numeric_fill_cols].describe().round(3))

    # Spearman相关
    corr_cols = SALES_NUMERIC_COLS + ["price_gap"]
    corr_df = df_clean[corr_cols].corr(method="spearman")
    print("\n各因素与销量的 Spearman 相关系数（从高到低）：")
    print(corr_df["simulated_sales"].sort_values(ascending=False))

    plot_heatmap(
        corr_df,
        title="附件2：销量影响因素相关性热力图",
        filename="附件2_相关性热力图.png"
    )

    plot_sales_relationships(df_clean, "附件2")

    vif_cols = ["price", "competitor_price", "product_score", "market_size"]
    vif_df = calculate_vif_safe(df_clean, vif_cols)
    print("\n附件2主要解释变量方差膨胀因子 VIF：")
    print(vif_df.round(4))

    # 导出
    save_dataframe(df_clean, ATTACHMENT_2.output_name)
    save_dataframe(vif_df, "附件2_VIF结果.xlsx")
    export_error_frames(error_frames, "附件2_异常样本")

    print("\n附件2处理完成，干净数据已导出。")
    return df_clean


# =========================
# 5. 主程序
# =========================
if __name__ == "__main__":
    df1 = process_attachment_1()
    df2 = process_attachment_2()

    print_section("全部处理完成")
    print(f"附件1清洗后样本量：{len(df1)}")
    print(f"附件2清洗后样本量：{len(df2)}")
    print(f"所有结果已保存至目录：{OUTPUT_DIR.resolve()}")