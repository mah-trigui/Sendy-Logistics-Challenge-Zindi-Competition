# ==============================================================================
# 00_config.R - Configuration & Libraries
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Global configuration, libraries, and settings
# ==============================================================================

cat("\n========================================\n")
cat("  SENDY LOGISTICS - Delivery Time Prediction\n")
cat("========================================\n\n")

# ==============================================================================
# 1. LIBRARIES
# ==============================================================================
cat("Loading libraries...\n")

# Core packages
library(here)
library(data.table)

# SQL operations
library(sqldf)
options(sqldf.driver = "RSQLite")

# Data manipulation
library(tidyverse)
library(dplyr)
library(tidyr)
library(magrittr)
library(lubridate)

# Geospatial
library(FNN)
library(geosphere)
library(pracma)

# Machine Learning
library(caret)
library(xgboost)
library(Matrix)
library(ranger)

# Ensemble learning
library(caretEnsemble)
library(randomForest)

# H2O (optional)
suppressWarnings(suppressMessages({
    if (requireNamespace("h2o", quietly = TRUE)) {
        library(h2o)
    }
}))

# Visualization
library(ggplot2)
library(ggthemes)
library(corrplot)
library(ggcorrplot)
library(ggforce)
library(car)
library(psych)

# Feature engineering
library(fastDummies)

# Data quality & exploration
library(skimr)
library(RANN)
library(PerformanceAnalytics)
library(woe)
library(scorecard)

# Weight of Evidence & Information Value (optional)
suppressWarnings(suppressMessages({
    if (requireNamespace("Information", quietly = TRUE)) {
        library(Information)
    }
}))

# Model interpretation (optional)
suppressWarnings(suppressMessages({
    if (requireNamespace("SHAPforxgboost", quietly = TRUE)) {
        library(SHAPforxgboost)
    }
    if (requireNamespace("iml", quietly = TRUE)) {
        library(iml)
    }
}))

# Normalization (optional)
suppressWarnings(suppressMessages({
    if (requireNamespace("bestNormalize", quietly = TRUE)) {
        library(bestNormalize)
    }
}))

# OpenRouteService for route data (optional)
suppressWarnings(suppressMessages({
    if (requireNamespace("openrouteservice", quietly = TRUE)) {
        library(openrouteservice)
        # Set API keys if needed
        # ors_api_key("YOUR_API_KEY_HERE")
    }
}))

cat("Libraries loaded!\n")

# ==============================================================================
# 2. GLOBAL SETTINGS
# ==============================================================================
cat("\nSetting global parameters...\n")

# Random seed for reproducibility
GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)

# Train/test split ratio
TRAIN_RATIO <- 0.75

# Cross-validation folds
CV_FOLDS <- 12 # Updated to 12-fold CV

# Number of parallel threads
N_THREADS <- 10

# ==============================================================================
# 3. FILE PATHS
# ==============================================================================

# Input files
TRAIN_FILE <- "Train.csv"
TEST_FILE <- "Test.csv"
RIDERS_FILE <- "Riders.csv"

# Output directories
OUTPUT_DIR <- "output"
SUBMISSION_DIR <- "submissions"

# Create output directories
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
if (!dir.exists(SUBMISSION_DIR)) dir.create(SUBMISSION_DIR, recursive = TRUE)

# ==============================================================================
# 4. XGBOOST DEFAULT PARAMETERS
# ==============================================================================

XGBOOST_PARAMS <- list(
    booster = "gbtree",
    objective = "reg:squarederror",
    eta = 0.01,
    max_depth = 6,
    subsample = 0.75,
    colsample_bytree = 0.7,
    min_child_weight = 3,
    gamma = 0,
    alpha = 0.7 # L1 regularization
)

XGBOOST_NROUNDS <- 3000
XGBOOST_EARLY_STOP <- 10

# ==============================================================================
# 5. FEATURE BINNING THRESHOLDS
# ==============================================================================

# Distance bins (km)
DIST_BINS <- list(
    VS = c(0, 4), # Very Short
    S = c(5, 9), # Short
    M = c(10, 13), # Medium
    L = c(14, Inf) # Long
)

# Rider age bins (days)
AGE_BINS <- list(
    VLV = c(0, 448),
    LV = c(449, 739),
    MV = c(740, 964),
    HV = c(965, 1426),
    VHV = c(1427, Inf)
)

# Rider order count bins
ORDER_BINS <- list(
    VLV = c(0, 889),
    LV = c(890, 1591),
    MV = c(1592, 2608),
    VHV = c(2609, Inf)
)

# Rider speed bins (km/h)
SPEED_BINS <- list(
    VLV = c(0, 19),
    LV = c(19, 24),
    MV = c(24, 30),
    HV = c(30, Inf)
)

# Rider score bins
SCORE_BINS <- list(
    VLV = c(0, 0.9),
    LV = c(0.9, 1.12),
    MV = c(1.12, 1.21),
    HV = c(1.21, Inf)
)

# Hour bins
HOUR_BINS <- list(
    A = c(0, 9),
    B = c(10, 11),
    C = c(12, 13),
    D = c(14, 14),
    E = c(15, 23),
    F = c(25, 25) # Missing hour
)

# ==============================================================================
# 6. HELPER FUNCTIONS
# ==============================================================================

#' Print section header
print_header <- function(title) {
    cat("\n")
    cat("========================================\n")
    cat(" ", title, "\n")
    cat("========================================\n\n")
}

#' Safe column selection (handles missing columns)
safe_select <- function(df, cols) {
    existing_cols <- cols[cols %in% names(df)]
    df[, existing_cols, drop = FALSE]
}

#' Calculate RMSE
calc_rmse <- function(actual, predicted) {
    sqrt(mean((actual - predicted)^2))
}

#' Calculate MAE
calc_mae <- function(actual, predicted) {
    mean(abs(actual - predicted))
}

#' Calculate R-squared
calc_r2 <- function(actual, predicted) {
    ss_res <- sum((actual - predicted)^2)
    ss_tot <- sum((actual - mean(actual))^2)
    1 - (ss_res / ss_tot)
}

cat("Configuration loaded!\n")
cat("========================================\n\n")
