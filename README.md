# Sendy Logistics — Delivery Time Prediction

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Sendy Logistics Challenge 🌾 - Win $7 000 USD](https://zindi.africa/competitions/sendy-logistics-challenge)

Ranked in the TOP 25%
---

Predicting delivery duration (pickup to arrival) for a logistics platform in Nairobi, Kenya.
Zindi ML Competition — Regression on tabular data.

---

## The Core Idea

A simple physics-based estimate of delivery time is easy to compute:

```
estimated_duration ≈ distance / rider_median_speed
```

Instead of using this formula as the prediction, or ignoring it and letting the model learn from scratch, this pipeline **feeds the analytical estimate into XGBoost as a feature** — and lets the model learn the residual correction.

The model's task becomes: *predict how wrong the baseline estimate is, and in which direction.*

This turns domain knowledge into a structured prior rather than treating it as a competing approach to ML.

---

## Key Engineering Decisions

### 1. Domain formula as feature (not as prediction)

```r
# Average of haversine and straight-line distance
df$distance <- ((df$distance / 1000) + df$Distance__KM_) / 2

# Baseline estimate from rider's historical speed
df$duree_estimee <- df$distance / df$speed_median * 60 * 60

# Normalized duration combining pickup behavior + estimate
df$duree_normalise <- df$arriv_pick_median + df$duree_estimee

# Improvement ratio: how much context deviates from expectation
df$improvement <- df$duree_normalise / df$arriv_pick
```

### 2. Rider profiling from historical behavior

Each rider was characterized by aggregated performance stats computed from training data: median speed, median duration, distance variability, and a composite quality score based on distance/duration flags.

### 3. Location clustering

K-Means clustering on all pickup/destination coordinates, then nearest-centroid assignment to encode geographic zones as features.

### 4. WOE-based feature selection

Information Value analysis on a binarized target (above/below duration threshold) to identify which features carry predictive signal vs. noise.

---

## Project Structure

```
├── 00_config.R                 # Libraries, paths, hyperparameters
├── 01_data_loading.R           # Load train/test/riders data
├── 02_data_cleaning.R          # Parse times, handle missing values, haversine
├── 03_feature_engineering.R    # Rider stats, clustering, bins, duration estimates
├── 03b_woe_analysis.R          # (Optional) Weight of Evidence feature selection
├── 04_model_xgboost.R          # XGBoost training, CV, prediction
├── 05_model_h2o.R              # H2O GBM, XGBoost, AutoML, stacking
├── 06_model_ensemble.R         # Caret-based stacking and ensembling
├── 06b_model_comparison.R      # (Optional) Multi-model benchmarking
├── 07_submission.R             # Submission file generation
├── MAIN.R                      # Full pipeline orchestration
└── README.md
```

---

## Technical Stack

- **Language**: R
- **Core packages**: xgboost, caret, h2o, ranger, data.table, tidyverse
- **Geospatial**: geosphere, FNN (nearest-neighbor cluster assignment)
- **Feature selection**: Information (IV/WOE), scorecard

---

## How to Run

```r
# Run the full pipeline
source("MAIN.R")
```

Requires `Train.csv`, `Test.csv`, and `Riders.csv` in the working directory.

---

## Scope

This repository shares the modeling pipeline and feature engineering approach. Competition data is not included. The focus is on the engineering decisions — particularly the hybrid analytical-ML design pattern.
- RMSE evaluation metric

## Alternative Models

### Weight of Evidence (WOE) Analysis
```r
source("organized/03b_woe_analysis.R")
woe_results <- run_woe_analysis(train)
important_features <- select_features_by_iv(woe_results, iv_threshold = 0.02)
```

### Model Comparison Framework
```r
source("organized/06b_model_comparison.R")
comparison <- run_model_comparison(
  train_set, test_set,
  models = c("lm", "rpart", "svm", "rf", "xgb")
)
```

### H2O GBM
```r
source("organized/05_model_h2o.R")
h2o_result <- train_h2o_pipeline(train, feature_cols, use_automl = FALSE)
```

### H2O AutoML
```r
h2o_result <- train_h2o_pipeline(train, feature_cols, use_automl = TRUE)
```

### Ensemble Stacking
```r
source("organized/06_model_ensemble.R")
ensemble <- train_ensemble_pipeline(train, feature_cols)
```

## Performance

Typical validation results:
- **RMSE**: ~600-700 seconds
- **MAE**: ~400-500 seconds

## Key Insights

1. **Rider Performance**: Rider historical performance (speed, delivery time) is highly predictive
2. **Distance**: Both straight-line and actual distance matter
3. **Time of Day**: Hour affects delivery time significantly
4. **Location Clusters**: Some pickup/destination areas are faster than others
5. **Wait Time**: Time spent waiting for pickup correlates with total delivery time

## Requirements

### R Packages
```r
# Core
library(data.table)
library(tidyverse)
library(dplyr)
library(lubridate)
library(sqldf)

# Geospatial
library(FNN)
library(geosphere)
library(pracma)

# Machine Learning
library(xgboost)
library(caret)
library(Matrix)

# Optional: H2O
library(h2o)

# Visualization
library(ggplot2)
```

## Notes

1. **Missing Times**: Some orders have missing confirmation/arrival times. These are handled by creating a "missing hour" category (Hour = 25)

2. **Speed Outliers**: Riders with extremely low (<7 km/h) or high (>50 km/h) average speeds are flagged

3. **Negative Predictions**: Any negative predictions are clipped to 1 second

4. **Location Clustering**: K-means clustering (k=10, k=15) is used to group pickup/destination locations

## Author

Sendy Logistics Delivery Time Prediction
Zindi Competition Entry
