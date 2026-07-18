# ==============================================================================
# 06b_model_comparison.R - Compare Multiple Models (Optional)
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Train and compare multiple ML models using caret
# ==============================================================================
# Based on Tutorial.R - comprehensive model comparison framework
# ==============================================================================

source("00_config.R")
print_header("Step 6b: Model Comparison")

# ==============================================================================
# TRAIN CONTROL SETUP
# ==============================================================================

#' Create train control for cross-validation
#'
#' @param method CV method (default: "cv")
#' @param n_folds Number of folds (default: 10)
#' @param verbose Show progress (default: TRUE)
#' @return trainControl object
#'
setup_train_control <- function(method = "cv", n_folds = 10, verbose = TRUE) {
    cat("Setting up train control...\n")
    cat("  Method:", method, "\n")
    cat("  Folds:", n_folds, "\n")

    ctrl <- trainControl(
        method = method,
        number = n_folds,
        verboseIter = verbose,
        savePredictions = "final",
        returnResamp = "all"
    )

    return(ctrl)
}

# ==============================================================================
# INDIVIDUAL MODEL TRAINING
# ==============================================================================

#' Train Linear Model
train_lm_model <- function(train_data, target_col, ctrl) {
    cat("\n--- Training Linear Model ---\n")

    formula_str <- paste(target_col, "~ .")
    model <- train(
        as.formula(formula_str),
        data = train_data,
        method = "lm",
        trControl = ctrl
    )

    cat("Linear Model Results:\n")
    print(model$results)

    return(model)
}

#' Train RPART (Decision Tree)
train_rpart_model <- function(train_data, target_col, ctrl, tune_grid = NULL) {
    cat("\n--- Training RPART Model ---\n")

    if (is.null(tune_grid)) {
        tune_grid <- expand.grid(cp = seq(0.000, 0.02, 0.0025))
    }

    formula_str <- paste(target_col, "~ .")
    model <- train(
        as.formula(formula_str),
        data = train_data,
        method = "rpart",
        trControl = ctrl,
        tuneGrid = tune_grid
    )

    cat("RPART Model Results:\n")
    print(model$results[1, ])

    return(model)
}

#' Train SVM with Radial Kernel
train_svm_model <- function(train_data, target_col, ctrl) {
    cat("\n--- Training SVM (Radial) Model ---\n")
    cat("WARNING: This may take several minutes...\n")

    formula_str <- paste(target_col, "~ .")
    model <- train(
        as.formula(formula_str),
        data = train_data,
        method = "svmRadial",
        trControl = ctrl
    )

    cat("SVM Model Results:\n")
    print(model$results[1, ])

    return(model)
}

#' Train Random Forest
train_rf_model <- function(train_data, target_col, ctrl) {
    cat("\n--- Training Random Forest Model ---\n")
    cat("WARNING: This may take several minutes...\n")

    formula_str <- paste(target_col, "~ .")
    model <- train(
        as.formula(formula_str),
        data = train_data,
        method = "rf",
        trControl = ctrl
    )

    cat("Random Forest Results:\n")
    print(model$results[1, ])

    return(model)
}

#' Train XGBoost via caret
train_xgb_caret_model <- function(train_data, target_col, ctrl) {
    cat("\n--- Training XGBoost Model (via caret) ---\n")
    cat("WARNING: This may take several minutes...\n")

    formula_str <- paste(target_col, "~ .")
    model <- train(
        as.formula(formula_str),
        data = train_data,
        method = "xgbTree",
        trControl = ctrl
    )

    cat("XGBoost Results:\n")
    print(model$results[1, ])

    return(model)
}

# ==============================================================================
# COMPARE MODELS
# ==============================================================================

#' Compare multiple models using resamples
#'
#' @param model_list Named list of trained models
#' @return Resamples object with comparison results
#'
compare_models <- function(model_list) {
    cat("\n==============================================\n")
    cat("  MODEL COMPARISON\n")
    cat("==============================================\n\n")

    # Combine resamples
    model_comp <- resamples(model_list)

    # Summary
    cat("Model Performance Summary:\n")
    print(summary(model_comp))

    # Correlation between models
    cat("\nModel Correlation:\n")
    print(modelCor(model_comp))

    # Create comparison plot
    tryCatch(
        {
            plot_obj <- dotplot(model_comp)
            print(plot_obj)
        },
        error = function(e) {
            cat("Could not create dotplot:", e$message, "\n")
        }
    )

    return(model_comp)
}

# ==============================================================================
# PREDICT AND EVALUATE
# ==============================================================================

#' Make predictions with all models and calculate RMSE
#'
#' @param model_list Named list of models
#' @param test_data Test data
#' @param target_col Target column name
#' @return Data frame with RMSE for each model
#'
evaluate_all_models <- function(model_list, test_data, target_col) {
    cat("\n==============================================\n")
    cat("  MODEL EVALUATION ON TEST SET\n")
    cat("==============================================\n\n")

    actual <- test_data[[target_col]]
    results <- data.frame()

    for (model_name in names(model_list)) {
        cat("Evaluating", model_name, "...\n")

        # Predict
        pred <- predict(model_list[[model_name]], test_data)

        # Calculate metrics
        rmse_val <- RMSE(pred, actual)
        mae_val <- MAE(pred, actual)
        r2_val <- R2(pred, actual)

        # Store results
        results <- rbind(results, data.frame(
            Model = model_name,
            RMSE = rmse_val,
            MAE = mae_val,
            R2 = r2_val
        ))

        cat("  RMSE:", round(rmse_val, 2), "\n")
    }

    # Sort by RMSE
    results <- results[order(results$RMSE), ]

    cat("\n--- Final Rankings (by RMSE) ---\n")
    print(results)

    # Highlight best model
    best_model <- results$Model[1]
    cat("\n*** BEST MODEL:", best_model, "***\n")
    cat("    RMSE:", round(results$RMSE[1], 2), "\n")
    cat("    MAE:", round(results$MAE[1], 2), "\n")
    cat("    R²:", round(results$R2[1], 4), "\n")

    return(results)
}

# ==============================================================================
# MASTER COMPARISON FUNCTION
# ==============================================================================

#' Full model comparison pipeline
#'
#' @param train_data Training data (with target)
#' @param test_data Test data (with target)
#' @param target_col Target column name
#' @param models Vector of model names to train
#' @return List with models, comparison, and evaluation results
#'
run_model_comparison <- function(train_data, test_data, target_col = "Time_from_Pickup_to_Arrival",
                                 models = c("lm", "rpart", "svm", "rf", "xgb")) {
    print_header("Full Model Comparison Pipeline")

    cat("Models to train:", paste(models, collapse = ", "), "\n\n")

    # Setup train control
    ctrl <- setup_train_control(method = "cv", n_folds = 10, verbose = TRUE)

    # Train models
    model_list <- list()

    if ("lm" %in% models) {
        model_list$LinearModel <- train_lm_model(train_data, target_col, ctrl)
    }

    if ("rpart" %in% models) {
        model_list$RPART <- train_rpart_model(train_data, target_col, ctrl)
    }

    if ("svm" %in% models) {
        model_list$SVM <- train_svm_model(train_data, target_col, ctrl)
    }

    if ("rf" %in% models) {
        model_list$RandomForest <- train_rf_model(train_data, target_col, ctrl)
    }

    if ("xgb" %in% models) {
        model_list$XGBoost <- train_xgb_caret_model(train_data, target_col, ctrl)
    }

    # Compare models
    comparison <- compare_models(model_list)

    # Evaluate on test set
    evaluation <- evaluate_all_models(model_list, test_data, target_col)

    cat("\n==============================================\n")
    cat("  MODEL COMPARISON COMPLETE!\n")
    cat("==============================================\n\n")

    return(list(
        models = model_list,
        comparison = comparison,
        evaluation = evaluation
    ))
}

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if (FALSE) {
    # This is example code - not executed automatically

    # Prepare data
    source("organized/01_data_loading.R")
    source("organized/02_data_cleaning.R")
    source("organized/03_feature_engineering.R")

    train_raw <- load_train_data()
    riders <- load_riders_data()

    train_raw <- clean_train_data(train_raw)
    result <- engineer_train_features(train_raw, riders)
    train_full <- result$train

    # Select features for comparison (simplified for speed)
    features <- c(
        "Pickup___Weekday__Mo___1_",
        "Distance__KM_", "Hour",
        "Temperature", "Precipitation_in_millimeters",
        "No_Of_Orders", "Age", "Average_Rating",
        "speed_med", "dur_avg", "best_rider",
        "Time_from_Pickup_to_Arrival"
    )

    train_subset <- train_full[, features]

    # One-hot encode categorical variables
    dmy <- dummyVars(" ~ .", data = train_subset, fullRank = FALSE)
    train_encoded <- data.frame(predict(dmy, newdata = train_subset))

    # Split into train/test
    set.seed(123)
    idx <- createDataPartition(train_encoded$Time_from_Pickup_to_Arrival, p = 0.75, list = FALSE)
    train_set <- train_encoded[idx, ]
    test_set <- train_encoded[-idx, ]

    # Run comparison (start with fast models)
    results <- run_model_comparison(
        train_set,
        test_set,
        models = c("lm", "rpart") # Add "svm", "rf", "xgb" for full comparison
    )

    # Access results
    print(results$evaluation)
    print(results$comparison)

    # Use best model for prediction
    best_model_name <- results$evaluation$Model[1]
    best_model <- results$models[[best_model_name]]
}

cat("\nModel comparison functions ready!\n")
cat("Use run_model_comparison() to compare multiple ML models\n\n")
