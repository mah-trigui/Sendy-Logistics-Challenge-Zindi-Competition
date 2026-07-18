# ==============================================================================
# 04_model_xgboost.R - XGBoost Model Training
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: XGBoost model training, CV, and prediction
# ==============================================================================

source("00_config.R")
print_header("Step 4: XGBoost Model")

# ==============================================================================
# 1. PREPARE DATA FOR XGBOOST
# ==============================================================================

#' Select features for modeling
select_model_features <- function(df, target_col = "Time_from_Pickup_to_Arrival") {
    cat("Selecting model features...\n")

    # Define feature columns
    feature_cols <- c(
        # Order features
        "Distance__KM_", "d_haver", "P_diff_dist",
        # Time features
        "plac_confir", "confir_arriv", "arriv_pick",
        "P_conf_plac", "P_arriv_conf",
        # Location clusters
        "class_pick_10", "class_dest_10", "Hour_f",
        # Rider features
        "Age", "P_order_rated",
        "speed_med", "speed_avg",
        "arriv_pick_med",
        "dur_avg", "dur_med",
        "dist_variat_med", "dur_variat_avg",
        "score",
        "P00", "P01", "P11", "P10",
        "Nb_Time_miss", "Outlier_Speed_Less", "Outlier_Speed_More",
        # Duration estimates
        "dur_estim", "dur_estim_avg", "dur_nor", "imp", "imp_b"
    )

    # Keep only available columns
    available_cols <- feature_cols[feature_cols %in% names(df)]
    cat("  Available features:", length(available_cols), "/", length(feature_cols), "\n")

    return(available_cols)
}

#' Create sparse matrix for XGBoost
create_train_matrix <- function(df, feature_cols, target_col = "Time_from_Pickup_to_Arrival") {
    cat("Creating training matrix...\n")

    # Get label
    label <- df[[target_col]]

    # Select features
    df_features <- df[, feature_cols, drop = FALSE]

    # Identify factor columns for contrasts
    factor_cols <- names(df_features)[sapply(df_features, is.factor)]

    # Create sparse matrix
    if (length(factor_cols) > 0) {
        trainMatrix <- sparse.model.matrix(
            ~.,
            data = df_features,
            contrasts.arg = setNames(rep(list("contr.treatment"), length(factor_cols)), factor_cols),
            sparse = FALSE, sci = FALSE
        )
    } else {
        trainMatrix <- sparse.model.matrix(~., data = df_features, sparse = FALSE, sci = FALSE)
    }

    cat("  Matrix dimensions:", nrow(trainMatrix), "x", ncol(trainMatrix), "\n")

    return(list(matrix = trainMatrix, label = label))
}

#' Create DMatrix for XGBoost
create_dmatrix <- function(matrix, label = NULL) {
    if (!is.null(label)) {
        dmatrix <- xgb.DMatrix(data = matrix, label = label)
    } else {
        dmatrix <- xgb.DMatrix(data = matrix)
    }

    return(dmatrix)
}

# ==============================================================================
# 2. CROSS-VALIDATION
# ==============================================================================

#' Run XGBoost cross-validation
run_xgb_cv <- function(dtrain, params = XGBOOST_PARAMS,
                       nrounds = XGBOOST_NROUNDS,
                       nfold = CV_FOLDS,
                       early_stop = XGBOOST_EARLY_STOP) {
    cat("Running XGBoost cross-validation...\n")
    cat("  Folds:", nfold, "\n")
    cat("  Max rounds:", nrounds, "\n")

    set.seed(GLOBAL_SEED)

    cv_result <- xgb.cv(
        data = dtrain,
        params = params,
        nrounds = nrounds,
        nfold = nfold,
        maximize = FALSE,
        eval_metric = "rmse",
        early_stopping_rounds = early_stop,
        nthread = N_THREADS,
        verbose = 1,
        print_every_n = 100
    )

    best_iter <- cv_result$best_iteration
    best_rmse <- cv_result$evaluation_log$test_rmse_mean[best_iter]

    cat("\n  Best iteration:", best_iter, "\n")
    cat("  Best CV RMSE:", round(best_rmse, 4), "\n")

    return(cv_result)
}

# ==============================================================================
# 3. MODEL TRAINING
# ==============================================================================

#' Train XGBoost model
train_xgboost <- function(dtrain, params = XGBOOST_PARAMS, nrounds = NULL) {
    cat("Training XGBoost model...\n")

    # If nrounds not specified, run CV first
    if (is.null(nrounds)) {
        cv_result <- run_xgb_cv(dtrain, params)
        nrounds <- cv_result$best_iteration
    }

    set.seed(GLOBAL_SEED)

    model <- xgb.train(
        data = dtrain,
        params = params,
        nrounds = nrounds,
        maximize = FALSE,
        eval_metric = "rmse",
        nthread = N_THREADS,
        verbose = 1,
        print_every_n = 100
    )

    cat("\nModel trained with", nrounds, "rounds\n")

    return(model)
}

#' Train with custom parameters
train_xgboost_custom <- function(dtrain, eta, max_depth, subsample, colsample_bytree,
                                 min_child_weight, nrounds = NULL) {
    params <- list(
        booster = "gbtree",
        objective = "reg:squarederror",
        eta = eta,
        max_depth = max_depth,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = min_child_weight
    )

    cat("\nCustom parameters:\n")
    cat("  eta:", eta, "\n")
    cat("  max_depth:", max_depth, "\n")
    cat("  subsample:", subsample, "\n")
    cat("  colsample_bytree:", colsample_bytree, "\n")
    cat("  min_child_weight:", min_child_weight, "\n")

    return(train_xgboost(dtrain, params, nrounds))
}

# ==============================================================================
# 4. PREDICTION
# ==============================================================================

#' Create test matrix (matching training features)
create_test_matrix <- function(df, feature_cols) {
    cat("Creating test matrix...\n")

    # Select features
    df_features <- df[, feature_cols, drop = FALSE]

    # Identify factor columns
    factor_cols <- names(df_features)[sapply(df_features, is.factor)]

    # Create sparse matrix
    if (length(factor_cols) > 0) {
        testMatrix <- sparse.model.matrix(
            ~.,
            data = df_features,
            contrasts.arg = setNames(rep(list("contr.treatment"), length(factor_cols)), factor_cols),
            sparse = FALSE, sci = FALSE
        )
    } else {
        testMatrix <- sparse.model.matrix(~., data = df_features, sparse = FALSE, sci = FALSE)
    }

    cat("  Matrix dimensions:", nrow(testMatrix), "x", ncol(testMatrix), "\n")

    return(testMatrix)
}

#' Make predictions
predict_xgboost <- function(model, test_matrix) {
    cat("Making predictions...\n")

    predictions <- predict(model, test_matrix)

    # Ensure non-negative predictions
    predictions[predictions < 0] <- 1

    cat("  Predictions range:", round(min(predictions), 2), "-", round(max(predictions), 2), "\n")
    cat("  Mean prediction:", round(mean(predictions), 2), "\n")

    return(predictions)
}

# ==============================================================================
# 5. FEATURE IMPORTANCE
# ==============================================================================

#' Get feature importance
get_feature_importance <- function(model, top_n = 30) {
    cat("Calculating feature importance...\n")

    importance <- xgb.importance(model = model)

    cat("\nTop", top_n, "features:\n")
    print(head(importance, top_n))

    return(importance)
}

#' Plot feature importance
plot_feature_importance <- function(model, top_n = 30, title = "XGBoost Feature Importance") {
    importance <- xgb.importance(model = model)
    xgb.plot.importance(importance, top_n = top_n, main = title)
}

# ==============================================================================
# 6. MODEL VALIDATION
# ==============================================================================

#' Validate model on holdout set
validate_model <- function(model, train, feature_cols, target_col = "Time_from_Pickup_to_Arrival",
                           val_ratio = 0.25) {
    cat("Validating model...\n")

    # Split data
    set.seed(GLOBAL_SEED)
    n <- nrow(train)
    val_idx <- sample(seq_len(n), size = floor(n * val_ratio))

    train_subset <- train[-val_idx, ]
    val_subset <- train[val_idx, ]

    # Create matrices
    train_data <- create_train_matrix(train_subset, feature_cols, target_col)
    val_data <- create_train_matrix(val_subset, feature_cols, target_col)

    # Predictions on validation set
    val_matrix <- val_data$matrix
    val_pred <- predict(model, val_matrix)
    val_pred[val_pred < 0] <- 1

    # Calculate metrics
    rmse <- calc_rmse(val_data$label, val_pred)
    mae <- calc_mae(val_data$label, val_pred)

    cat("\nValidation Results:\n")
    cat("  RMSE:", round(rmse, 4), "\n")
    cat("  MAE:", round(mae, 4), "\n")

    return(list(
        rmse = rmse,
        mae = mae,
        predictions = val_pred,
        actual = val_data$label
    ))
}

# ==============================================================================
# 7. FULL TRAINING PIPELINE
# ==============================================================================

#' Complete XGBoost training pipeline
train_xgb_pipeline <- function(train, feature_cols = NULL,
                               target_col = "Time_from_Pickup_to_Arrival",
                               params = XGBOOST_PARAMS,
                               run_cv = TRUE) {
    print_header("XGBoost Training Pipeline")

    # Select features if not provided
    if (is.null(feature_cols)) {
        feature_cols <- select_model_features(train, target_col)
    }

    # Create training data
    train_data <- create_train_matrix(train, feature_cols, target_col)
    dtrain <- create_dmatrix(train_data$matrix, train_data$label)

    # Cross-validation
    nrounds <- NULL
    if (run_cv) {
        cv_result <- run_xgb_cv(dtrain, params)
        nrounds <- cv_result$best_iteration
    }

    # Train final model
    model <- train_xgboost(dtrain, params, nrounds)

    # Feature importance
    importance <- get_feature_importance(model)

    cat("\nTraining pipeline complete!\n")

    return(list(
        model = model,
        feature_cols = feature_cols,
        importance = importance,
        nrounds = nrounds
    ))
}

cat("\nXGBoost model functions ready!\n")
