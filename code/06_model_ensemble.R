# ==============================================================================
# 06_model_ensemble.R - Ensemble & Stacking Models
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Model stacking and ensemble methods using caret
# ==============================================================================

source("00_config.R")
print_header("Step 6: Ensemble Models")

# ==============================================================================
# 1. TRAIN/VALIDATION SPLIT
# ==============================================================================

#' Create train/validation split
create_train_val_split <- function(train, target_col = "Time_from_Pickup_to_Arrival",
                                   val_ratio = 0.25) {
    cat("Creating train/validation split...\n")

    set.seed(GLOBAL_SEED)
    n <- nrow(train)
    val_idx <- sample(seq_len(n), size = floor(n * val_ratio))

    train_subset <- train[-val_idx, ]
    val_subset <- train[val_idx, ]

    cat("  Training set:", nrow(train_subset), "rows\n")
    cat("  Validation set:", nrow(val_subset), "rows\n")

    return(list(
        train = train_subset,
        val = val_subset,
        val_idx = val_idx
    ))
}

# ==============================================================================
# 2. CARET MODEL TRAINING
# ==============================================================================

#' Train control for caret
get_train_control <- function(method = "cv", number = 5, repeats = 3) {
    trainControl(
        method = method,
        number = number,
        repeats = if (method == "repeatedcv") repeats else NULL,
        savePredictions = TRUE,
        verboseIter = TRUE
    )
}

#' Train XGBoost via caret
train_caret_xgb <- function(train, feature_cols, target_col = "Time_from_Pickup_to_Arrival") {
    cat("Training XGBoost via caret...\n")

    # Prepare formula
    train_data <- train[, c(feature_cols, target_col)]
    formula <- as.formula(paste(target_col, "~ ."))

    # Train control
    ctrl <- get_train_control()

    # Train model
    set.seed(GLOBAL_SEED)
    model <- train(
        formula,
        data = train_data,
        method = "xgbTree",
        trControl = ctrl,
        metric = "RMSE",
        verbosity = 0
    )

    cat("  Best RMSE:", round(min(model$results$RMSE), 4), "\n")

    return(model)
}

#' Train GBM via caret
train_caret_gbm <- function(train, feature_cols, target_col = "Time_from_Pickup_to_Arrival") {
    cat("Training GBM via caret...\n")

    # Prepare data
    train_data <- train[, c(feature_cols, target_col)]
    formula <- as.formula(paste(target_col, "~ ."))

    # Train control
    ctrl <- get_train_control()

    # Train model
    set.seed(GLOBAL_SEED)
    model <- train(
        formula,
        data = train_data,
        method = "gbm",
        trControl = ctrl,
        metric = "RMSE",
        verbose = FALSE
    )

    cat("  Best RMSE:", round(min(model$results$RMSE), 4), "\n")

    return(model)
}

#' Train Random Forest via caret
train_caret_rf <- function(train, feature_cols, target_col = "Time_from_Pickup_to_Arrival") {
    cat("Training Random Forest via caret...\n")

    # Prepare data
    train_data <- train[, c(feature_cols, target_col)]
    formula <- as.formula(paste(target_col, "~ ."))

    # Train control
    ctrl <- get_train_control()

    # Train model
    set.seed(GLOBAL_SEED)
    model <- train(
        formula,
        data = train_data,
        method = "rf",
        trControl = ctrl,
        metric = "RMSE"
    )

    cat("  Best RMSE:", round(min(model$results$RMSE), 4), "\n")

    return(model)
}

# ==============================================================================
# 3. MODEL STACKING
# ==============================================================================

#' Train multiple base models
train_base_models <- function(train, feature_cols, target_col = "Time_from_Pickup_to_Arrival") {
    cat("Training base models for stacking...\n\n")

    models <- list()

    # XGBoost
    models$xgb <- train_caret_xgb(train, feature_cols, target_col)

    # GBM
    models$gbm <- train_caret_gbm(train, feature_cols, target_col)

    cat("\nBase models trained!\n")

    return(models)
}

#' Create stacking predictions (level 1)
create_stacking_preds <- function(models, train, val, feature_cols,
                                  target_col = "Time_from_Pickup_to_Arrival") {
    cat("Creating stacking predictions...\n")

    # Prepare feature data
    train_features <- train[, feature_cols]
    val_features <- val[, feature_cols]

    # Get predictions from each model
    train_preds <- data.frame(
        actual = train[[target_col]]
    )

    val_preds <- data.frame(
        actual = val[[target_col]]
    )

    for (name in names(models)) {
        cat("  Getting predictions from:", name, "\n")

        # Training predictions (use cross-validation predictions if available)
        if ("pred" %in% names(models[[name]])) {
            # Use CV predictions
            cv_preds <- models[[name]]$pred
            cv_preds <- cv_preds[order(cv_preds$rowIndex), ]
            train_preds[[name]] <- cv_preds$pred[1:nrow(train)]
        } else {
            train_preds[[name]] <- predict(models[[name]], train_features)
        }

        # Validation predictions
        val_preds[[name]] <- predict(models[[name]], val_features)
    }

    return(list(train = train_preds, val = val_preds))
}

#' Train meta-learner (level 2)
train_meta_learner <- function(stacking_preds, method = "rf") {
    cat("Training meta-learner...\n")

    # Prepare data (exclude actual for features)
    train_data <- stacking_preds$train

    # Train control
    ctrl <- get_train_control()

    # Train meta-learner
    set.seed(GLOBAL_SEED)
    meta_model <- train(
        actual ~ .,
        data = train_data,
        method = method,
        trControl = ctrl,
        metric = "RMSE"
    )

    cat("  Meta-learner RMSE:", round(min(meta_model$results$RMSE), 4), "\n")

    return(meta_model)
}

#' Make stacked predictions
predict_stacked <- function(base_models, meta_model, new_data, feature_cols) {
    cat("Making stacked predictions...\n")

    # Prepare features
    features <- new_data[, feature_cols]

    # Get base model predictions
    base_preds <- data.frame(row.names = 1:nrow(new_data))

    for (name in names(base_models)) {
        base_preds[[name]] <- predict(base_models[[name]], features)
    }

    # Get meta-learner prediction
    final_preds <- predict(meta_model, base_preds)

    # Ensure non-negative
    final_preds[final_preds < 0] <- 1

    cat("  Predictions range:", round(min(final_preds), 2), "-", round(max(final_preds), 2), "\n")

    return(final_preds)
}

# ==============================================================================
# 4. SIMPLE AVERAGING ENSEMBLE
# ==============================================================================

#' Average predictions from multiple models
average_predictions <- function(predictions_list, weights = NULL) {
    cat("Averaging predictions...\n")

    n_models <- length(predictions_list)

    if (is.null(weights)) {
        weights <- rep(1 / n_models, n_models)
    }

    # Weighted average
    avg_preds <- rep(0, length(predictions_list[[1]]))

    for (i in seq_along(predictions_list)) {
        avg_preds <- avg_preds + weights[i] * predictions_list[[i]]
    }

    cat("  Ensemble weights:", paste(round(weights, 3), collapse = ", "), "\n")

    return(avg_preds)
}

# ==============================================================================
# 5. FULL ENSEMBLE PIPELINE
# ==============================================================================

#' Complete ensemble training pipeline
train_ensemble_pipeline <- function(train, feature_cols,
                                    target_col = "Time_from_Pickup_to_Arrival",
                                    val_ratio = 0.25) {
    print_header("Ensemble Training Pipeline")

    # Split data
    split <- create_train_val_split(train, target_col, val_ratio)

    # Train base models
    base_models <- train_base_models(split$train, feature_cols, target_col)

    # Create stacking predictions
    stacking_preds <- create_stacking_preds(
        base_models, split$train, split$val, feature_cols, target_col
    )

    # Train meta-learner
    meta_model <- train_meta_learner(stacking_preds)

    # Evaluate on validation
    val_preds <- predict_stacked(base_models, meta_model, split$val, feature_cols)
    val_rmse <- calc_rmse(split$val[[target_col]], val_preds)

    cat("\nEnsemble Validation RMSE:", round(val_rmse, 4), "\n")

    return(list(
        base_models = base_models,
        meta_model = meta_model,
        feature_cols = feature_cols,
        val_rmse = val_rmse
    ))
}

cat("\nEnsemble model functions ready!\n")
