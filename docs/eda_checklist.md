# EDA Checklist

Step-by-step checklist for exploratory data analysis. Quick explanations only — dig deeper as needed per dataset.

## 1. Load & Overview
- [ ] Load data, check shape (rows/columns).
- [ ] Preview head/tail rows — sanity check format.
- [ ] Check dtypes per column — catch wrong types (e.g. numbers stored as strings).

## 2. Missing Values
- [ ] Count/percent missing per column.
- [ ] Visualize missingness pattern (e.g. missingno matrix) — spot systematic gaps.
- [ ] Decide: drop, impute, or flag as feature.

## 3. Duplicates
- [ ] Check full-row duplicates.
- [ ] Check duplicate keys (e.g. user_id, order_id) — signals join/ETL issues.

## 4. Univariate Analysis
- [ ] Numeric columns: describe() — min/max/mean/median/std.
- [ ] Histograms/boxplots — check distribution shape, skew.
- [ ] Categorical columns: value_counts() — check cardinality, imbalance.

## 5. Outliers
- [ ] Boxplots or IQR/z-score method to flag outliers.
- [ ] Decide: real signal (e.g. big spender) vs data error.

## 6. Target Variable (if supervised task)
- [ ] Check distribution of target (class balance for classification, skew for regression).
- [ ] Note imbalance — may need resampling/weighting later.

## 7. Bivariate / Correlation Analysis
- [ ] Correlation matrix (numeric features) — spot multicollinearity.
- [ ] Feature vs target relationship (scatter, groupby mean, boxplot by class).
- [ ] Categorical vs categorical: crosstab/chi-square if needed.

## 8. Time-based Checks (if temporal data)
- [ ] Check date range coverage, gaps.
- [ ] Plot trend over time — spot seasonality, drift.

## 9. Data Consistency
- [ ] Check value ranges make sense (e.g. no negative age, price >= 0).
- [ ] Check category labels consistent (e.g. "US" vs "USA" vs "United States").
- [ ] Check referential integrity between tables (foreign keys exist).

## 10. Feature Relationships for Recommendation Context
- [ ] User-item interaction sparsity — how many users/items, how many interactions.
- [ ] Cold-start check — users/items with very few interactions.
- [ ] Popularity bias — check long-tail distribution of item interactions.

## 11. Summary
- [ ] Write down key findings, anomalies, decisions made (imputation, filtering, etc).
- [ ] Flag data quality issues to fix before modeling.
