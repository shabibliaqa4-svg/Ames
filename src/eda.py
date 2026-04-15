"""
Exploratory Data Analysis - generates 15+ visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config.settings import Settings
from src.utils import get_logger, timer

logger = get_logger(__name__)
settings = Settings()

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE = (14, 8)


class ExploratoryAnalysis:
    """Generates comprehensive EDA plots and statistics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target = settings.TARGET_COLUMN
        self.numeric_df = df.select_dtypes(include=[np.number])
        self.categorical_df = df.select_dtypes(include=["object", "category"])
        self.plot_dir = settings.PLOT_DIR
        self.stats: dict = {}

    def _save(self, fig: plt.Figure, name: str):
        path = os.path.join(self.plot_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("📈 Saved plot → %s", path)

    # ── 1. Target distribution ──────────────────────────────────────
    def plot_target_distribution(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Raw distribution
        sns.histplot(self.df[self.target], bins=60, kde=True, ax=axes[0],
                     color="steelblue")
        axes[0].set_title("Sale Price Distribution")
        axes[0].axvline(self.df[self.target].mean(), color="red",
                        linestyle="--", label=f"Mean: ${self.df[self.target].mean():,.0f}")
        axes[0].axvline(self.df[self.target].median(), color="green",
                        linestyle="--", label=f"Median: ${self.df[self.target].median():,.0f}")
        axes[0].legend()

        # Log-transformed
        log_target = np.log1p(self.df[self.target])
        sns.histplot(log_target, bins=60, kde=True, ax=axes[1], color="coral")
        axes[1].set_title("Log(1 + Sale Price) Distribution")

        # Q-Q plot
        stats.probplot(log_target, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot (Log-Transformed)")

        fig.suptitle("Target Variable Analysis", fontsize=16, y=1.02)
        fig.tight_layout()
        self._save(fig, "01_target_distribution")

        # Store stats
        skewness = self.df[self.target].skew()
        kurtosis = self.df[self.target].kurtosis()
        self.stats["target_skewness"] = skewness
        self.stats["target_kurtosis"] = kurtosis
        self.stats["target_mean"] = self.df[self.target].mean()
        self.stats["target_median"] = self.df[self.target].median()
        self.stats["target_std"] = self.df[self.target].std()

    # ── 2. Correlation heatmap ──────────────────────────────────────
    def plot_correlation_heatmap(self):
        corr = self.numeric_df.corr()
        # Top 25 features most correlated with target
        if self.target in corr.columns:
            top_cols = corr[self.target].abs().sort_values(ascending=False).head(26).index
            top_corr = self.numeric_df[top_cols].corr()
        else:
            top_corr = corr.iloc[:25, :25]

        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(top_corr, dtype=bool))
        sns.heatmap(top_corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, square=True,
                    linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Top 25 Features - Correlation Matrix", fontsize=14)
        fig.tight_layout()
        self._save(fig, "02_correlation_heatmap")

        self.stats["top_10_corr_features"] = (
            corr[self.target].abs().sort_values(ascending=False)
            .head(11).index.tolist()[1:]  # exclude self
        )

    # ── 3. Top feature scatter plots ────────────────────────────────
    def plot_top_scatter(self, n: int = 9):
        corr = self.numeric_df.corr()
        top_features = (
            corr[self.target].abs()
            .sort_values(ascending=False)
            .head(n + 1).index.tolist()
        )
        top_features = [f for f in top_features if f != self.target][:n]

        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten()

        for i, feat in enumerate(top_features):
            ax = axes[i]
            ax.scatter(self.df[feat], self.df[self.target], alpha=0.3,
                       s=15, color="steelblue")
            # Regression line
            z = np.polyfit(self.df[feat].dropna(), 
                           self.df[self.target].loc[self.df[feat].dropna().index], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.df[feat].min(), self.df[feat].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)
            r_val = corr.loc[feat, self.target]
            ax.set_title(f"{feat} (r = {r_val:.3f})", fontsize=11)
            ax.set_xlabel(feat)
            ax.set_ylabel(self.target)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Top Correlated Features vs Sale Price", fontsize=16, y=1.01)
        fig.tight_layout()
        self._save(fig, "03_top_scatter_plots")

    # ── 4. Categorical feature boxplots ─────────────────────────────
    def plot_categorical_boxplots(self, n: int = 9):
        cat_cols = self.categorical_df.columns.tolist()
        # Select those with reasonable cardinality
        selected = [c for c in cat_cols if self.df[c].nunique() <= 15][:n]

        ncols = 3
        nrows = (len(selected) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(selected):
            order = (
                self.df.groupby(col)[self.target].median()
                .sort_values().index
            )
            sns.boxplot(data=self.df, x=col, y=self.target,
                        order=order, ax=axes[i], palette="Set2")
            axes[i].set_title(f"{col} vs {self.target}", fontsize=11)
            axes[i].tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Features vs Sale Price", fontsize=16, y=1.01)
        fig.tight_layout()
        self._save(fig, "04_categorical_boxplots")

    # ── 5. Missing value heatmap ────────────────────────────────────
    def plot_missing_values(self):
        null_cols = self.df.columns[self.df.isnull().any()].tolist()
        if not null_cols:
            logger.info("No missing values to plot.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        # Bar chart
        null_pct = (self.df[null_cols].isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        null_pct.plot.barh(ax=axes[0], color="salmon")
        axes[0].set_xlabel("Missing %")
        axes[0].set_title("Missing Value Percentage")
        # Matrix
        sns.heatmap(self.df[null_cols].isnull().T, cbar=False,
                    yticklabels=True, ax=axes[1], cmap="YlOrRd")
        axes[1].set_title("Missing Value Pattern")
        fig.tight_layout()
        self._save(fig, "05_missing_values")

    # ── 6. Numeric feature distributions ────────────────────────────
    def plot_numeric_distributions(self, n: int = 16):
        num_cols = self.numeric_df.columns.tolist()
        cols = [c for c in num_cols if c != self.target][:n]

        ncols = 4
        nrows = (len(cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col].dropna(), bins=40, kde=True,
                         ax=axes[i], color="steelblue")
            skew_val = self.df[col].skew()
            axes[i].set_title(f"{col} (skew={skew_val:.2f})", fontsize=10)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Feature Distributions", fontsize=16, y=1.01)
        fig.tight_layout()
        self._save(fig, "06_numeric_distributions")

    # ── 7. Neighborhood analysis ────────────────────────────────────
    def plot_neighborhood_analysis(self):
        if "Neighborhood" not in self.df.columns:
            return
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        order = self.df.groupby("Neighborhood")[self.target].median().sort_values().index
        sns.boxplot(data=self.df, x="Neighborhood", y=self.target,
                    order=order, ax=axes[0], palette="coolwarm")
        axes[0].tick_params(axis="x", rotation=90)
        axes[0].set_title("Sale Price by Neighborhood (Box)")
        self.df.groupby("Neighborhood").size().reindex(order).plot.barh(
            ax=axes[1], color="steelblue"
        )
        axes[1].set_title("Homes Sold per Neighborhood")
        fig.tight_layout()
        self._save(fig, "07_neighborhood_analysis")

    # ── 8. Year effects ─────────────────────────────────────────────
    def plot_year_effects(self):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        if "YearBuilt" in self.df.columns:
            axes[0].scatter(self.df["YearBuilt"], self.df[self.target],
                            alpha=0.2, s=10, color="steelblue")
            axes[0].set_title("Year Built vs Sale Price")
            axes[0].set_xlabel("Year Built")

        if "YearRemodAdd" in self.df.columns:
            axes[1].scatter(self.df["YearRemodAdd"], self.df[self.target],
                            alpha=0.2, s=10, color="coral")
            axes[1].set_title("Year Remodeled vs Sale Price")
            axes[1].set_xlabel("Year Remodeled")

        if "YrSold" in self.df.columns:
            self.df.groupby("YrSold")[self.target].agg(["mean", "median"]).plot(
                ax=axes[2], marker="o"
            )
            axes[2].set_title("Average Sale Price by Year Sold")

        fig.tight_layout()
        self._save(fig, "08_year_effects")

    # ── 9. Feature skewness ─────────────────────────────────────────
    def plot_skewness(self):
        skewness = self.numeric_df.skew().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ["salmon" if abs(s) > 1 else "steelblue" for s in skewness]
        skewness.plot.barh(ax=ax, color=colors)
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=-1, color="red", linestyle="--", alpha=0.5)
        ax.set_title("Feature Skewness (|skew| > 1 highlighted in red)")
        fig.tight_layout()
        self._save(fig, "09_feature_skewness")
        self.stats["highly_skewed_features"] = skewness[skewness.abs() > 1].index.tolist()

    # ── 10. Quality feature analysis ────────────────────────────────
    def plot_quality_features(self):
        qual_cols = ["OverallQual", "OverallCond", "ExterQual", "KitchenQual"]
        qual_cols = [c for c in qual_cols if c in self.df.columns]

        fig, axes = plt.subplots(1, len(qual_cols), figsize=(6 * len(qual_cols), 6))
        if len(qual_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, qual_cols):
            self.df.groupby(col)[self.target].median().sort_index().plot.bar(
                ax=ax, color="steelblue"
            )
            ax.set_title(f"Median Sale Price by {col}")
            ax.set_ylabel("Median Price ($)")
            ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        self._save(fig, "10_quality_features")

    # ── Run all ─────────────────────────────────────────────────────
    @timer
    def run_full_eda(self) -> dict:
        """Execute all EDA steps and return computed statistics."""
        logger.info("🔍 Starting Exploratory Data Analysis...")
        logger.info("   Dataset shape: %s", self.df.shape)
        logger.info("   Numeric features: %d", self.numeric_df.shape[1])
        logger.info("   Categorical features: %d", self.categorical_df.shape[1])

        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_top_scatter()
        self.plot_categorical_boxplots()
        self.plot_missing_values()
        self.plot_numeric_distributions()
        self.plot_neighborhood_analysis()
        self.plot_year_effects()
        self.plot_skewness()
        self.plot_quality_features()

        logger.info("✅ EDA complete - %d plots generated", 10)
        return self.stats
