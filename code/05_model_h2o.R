# ==============================================================================
# 05_model_h2o.R - H2O AutoML & Ensemble Models
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: H2O-based models including AutoML and GBM
# ==============================================================================

source("00_config.R")
print_header("Step 5: H2O Models")

# ==============================================================================
# 1. H2O INITIALIZATION
# ==============================================================================

#' Initialize H2O cluster
init_h2o <- function(nthreads = -1) {
    cat("Initializing H2O...\n")

    if (!requireNamespace("h2o", quietly = TRUE)) {
        stop("H2O package not installed. Install with: install.packages('h2o')")
    }

    h2o.init(nthreads = nthreads)
    cat("H2O initialized!\n")
}

#' Shutdown H2O cluster
shutdown_h2o <- function() {
    h2o.shutdown(prompt = FALSE)
    cat("H2O shutdown complete.\n")
}

# ==============================================================================
# 2. DATA PREPARATION FOR H2O
# ==============================================================================

#' Convert data to H2O frame
to_h2o_frame <- function(df, name = "data") {
    cat("Converting to H2O frame:", name, "\n")
    h2o_frame <- as.h2o(df)
    cat("  Rows:", nrow(h2o_frame), "\n")
    cat("  Columns:", ncol(h2o_frame), "\n")
    return(h2o_frame)
}

#' Prepare train/test split for H2O
prepare_h2o_data <- function(train, target_col = "Time_from_Pickup_to_Arrival",
                             val_ratio = 0.25) {
    cat("Preparing H2O data...\n")

    # Split
    set.seed(GLOBAL_SEED)
    n <- nrow(train)
    val_idx <- sample(seq_len(n), size = floor(n * val_ratio))

    train_subset <- train[-val_idx, ]
    val_subset <- train[val_idx, ]

    # Convert to H2O
    h2o_train <- to_h2o_frame(train_subset, "train")
    h2o_val <- to_h2o_frame(val_subset, "validation")

    # Feature and target columns
    y <- target_col
    x <- setdiff(names(h2o_train), y)

    cat("  Target:", y, "\n")
    cat("  Features:", length(x), "\n")

    return(list(
        train = h2o_train,
        val = h2o_val,
        x = x,
        y = y
    ))
}

# ==============================================================================
# 3. H2O GBM MODEL
# ==============================================================================

#' Train H2O GBM model
train_h2o_gbm <- function(h2o_data, ntrees = 1500, learn_rate = 0.01,
                          max_depth = 7, min_rows = 5, sample_rate = 0.8) {
    cat("Training H2O GBM...\n")

    model <- h2o.gbm(
        x = h2o_data$x,
        y = h2o_data$y,
        training_frame = h2o_data$train,
        ntrees = ntrees,
        learn_rate = learn_rate,
        max_depth = max_depth,
        min_rows = min_rows,
        sample_rate = sample_rate,
        nfolds = CV_FOLDS,
        fold_assignment = "Modulo",
        keep_cross_validation_predictions = TRUE,
        seed = GLOBAL_SEED,
        stopping_rounds = 50,
        stopping_metric = "RMSE",
        stopping_tolerance = 0
    )

    # Print performance
    perf <- h2o.performance(model, newdata = h2o_data$val)
    rmse <- h2o.rmse(perf)
    cat("\n  Validation RMSE:", round(rmse, 4), "\n")

    return(model)
}

# ==============================================================================
# 4. H2O XGBOOST MODEL
# ==============================================================================

#' Train H2O XGBoost model
train_h2o_xgb <- function(h2o_data, ntrees = 1500, learn_rate = 0.05,
                          max_depth = 6, min_rows = 3, sample_rate = 0.8) {
    cat("Training H2O XGBoost...\n")

    model <- h2o.xgboost(
        x = h2o_data$x,
        y = h2o_data$y,
        training_frame = h2o_data$train,
        ntrees = ntrees,
        learn_rate = learn_rate,
        max_depth = max_depth,
        min_rows = min_rows,
        sample_rate = sample_rate,
        categorical_encoding = "Enum",
        nfolds = CV_FOLDS,
        fold_assignment = "Modulo",
        keep_cross_validation_predictions = TRUE,
        seed = GLOBAL_SEED,
        stopping_rounds = 50,
        stopping_metric = "RMSE",
        stopping_tolerance = 0
    )

    # Print performance
    perf <- h2o.performance(model, newdata = h2o_data$val)
    rmse <- h2o.rmse(perf)
    cat("\n  Validation RMSE:", round(rmse, 4), "\n")

    return(model)
}

# ==============================================================================
# 5. H2O AUTOML
# ==============================================================================

#' Run H2O AutoML
run_h2o_automl <- function(h2o_data, max_models = 20, max_runtime_secs = 3600) {
    cat("Running H2O AutoML...\n")
    cat("  Max models:", max_models, "\n")
    cat("  Max runtime:", max_runtime_secs, "seconds\n")

    aml <- h2o.automl(
        x = h2o_data$x,
        y = h2o_data$y,
        training_frame = h2o_data$train,
        max_models = max_models,
        max_runtime_secs = max_runtime_secs,
        seed = GLOBAL_SEED
    )

    # Print leaderboard
    lb <- aml@leaderboard
    cat("\nAutoML Leaderboard:\n")
    print(lb, n = min(nrow(lb), 10))

    return(aml)
}

# ==============================================================================
# 6. STACKED ENSEMBLE
# ==============================================================================

#' Create stacked ensemble from multiple models
train_h2o_ensemble <- function(h2o_data, base_models) {
    cat("Training stacked ensemble...\n")

    ensemble <- h2o.stackedEnsemble(
        x = h2o_data$x,
        y = h2o_data$y,
        training_frame = h2o_data$train,
        base_models = base_models,
        metalearner_algorithm = "drf"
    )

    # Print performance
    perf <- h2o.performance(ensemble, newdata = h2o_data$val)
    rmse <- h2o.rmse(perf)
    cat("\n  Ensemble Validation RMSE:", round(rmse, 4), "\n")

    return(ensemble)
}

# ==============================================================================
# 7. PREDICTION
# ==============================================================================

#' Make predictions with H2O model
predict_h2o <- function(model, h2o_data) {
    cat("Making H2O predictions...\n")

    predictions <- h2o.predict(model, h2o_data)
    pred_vector <- as.vector(predictions[, 1])

    # Ensure non-negative
    pred_vector[pred_vector < 0] <- 1

    cat("  Predictions range:", round(min(pred_vector), 2), "-", round(max(pred_vector), 2), "\n")

    return(pred_vector)
}

# ==============================================================================
# 8. FULL H2O PIPELINE
# ==============================================================================

#' Complete H2O training pipeline
train_h2o_pipeline <- function(train, feature_cols = NULL,
                               target_col = "Time_from_Pickup_to_Arrival",
                               use_automl = FALSE) {
    print_header("H2O Training Pipeline")

    # Initialize H2O
    init_h2o()

    # Select features if needed
    if (!is.null(feature_cols)) {
        cols_to_keep <- c(feature_cols, target_col)
        train <- train[, cols_to_keep[cols_to_keep %in% names(train)]]
    }

    # Prepare data
    h2o_data <- prepare_h2o_data(train, target_col)

    if (use_automl) {
        # Run AutoML
        aml <- run_h2o_automl(h2o_data)
        best_model <- aml@leader
    } else {
        # Train individual models
        gbm_model <- train_h2o_gbm(h2o_data)
        xgb_model <- train_h2o_xgb(h2o_data)

        # Create ensemble
        best_model <- train_h2o_ensemble(h2o_data, list(gbm_model, xgb_model))
    }

    cat("\nH2O pipeline complete!\n")

    return(list(
        model = best_model,
        h2o_data = h2o_data
    ))
}

cat("\nH2O model functions ready!\n")
