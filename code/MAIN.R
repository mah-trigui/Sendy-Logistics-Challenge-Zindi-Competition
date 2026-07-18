# ==============================================================================
# MAIN.R - Sendy Logistics Delivery Time Prediction Pipeline
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Competition: Zindi Challenge
# Description: Main orchestration script to run the complete pipeline
# ==============================================================================

cat("\n")
cat("================================================================\n")
cat("   SENDY LOGISTICS - DELIVERY TIME PREDICTION PIPELINE\n")
cat("   Nairobi, Kenya\n")
cat("================================================================\n\n")

# ==============================================================================
# 1. LOAD ALL MODULES
# ==============================================================================
cat("Loading modules...\n")

source("00_config.R")
source("01_data_loading.R")
source("02_data_cleaning.R")
source("03_feature_engineering.R")
source("04_model_xgboost.R")
# source("05_model_h2o.R")      # Optional: H2O models
source("06_model_ensemble.R")
source("07_submission.R")

cat("All modules loaded!\n\n")

# ==============================================================================
# 2. LOAD RAW DATA
# ==============================================================================
print_header("Loading Raw Data")

train <- load_train_data(TRAIN_FILE)
test <- load_test_data(TEST_FILE)
riders <- load_riders_data(RIDERS_FILE)

# Validate data
validate_data(train, test, riders)

cat("\nRaw data loaded!\n")

# ==============================================================================
# 3. CLEAN DATA
# ==============================================================================
print_header("Cleaning Data")

train <- clean_train_data(train)
test <- clean_test_data(test)

cat("\nData cleaned!\n")

# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print_header("Feature Engineering")

# Engineer training features (creates enriched riders and location clusters)
train_result <- engineer_train_features(train, riders)
train <- train_result$train
riders <- train_result$riders
cluster_data <- train_result$cluster_data

# Engineer test features (using pre-computed values from training)
test <- engineer_test_features(test, riders, cluster_data)

cat("\nFeatures engineered!\n")

# ==============================================================================
# 5. SELECT FEATURES FOR MODELING
# ==============================================================================
print_header("Selecting Model Features")

feature_cols <- select_model_features(train)
cat("Selected", length(feature_cols), "features for modeling\n")

# ==============================================================================
# 6. TRAIN XGBOOST MODEL
# ==============================================================================
print_header("Training XGBoost Model")

# Full pipeline: CV + training
xgb_result <- train_xgb_pipeline(
    train = train,
    feature_cols = feature_cols,
    target_col = "Time_from_Pickup_to_Arrival",
    params = XGBOOST_PARAMS,
    run_cv = TRUE
)

model <- xgb_result$model
importance <- xgb_result$importance

cat("\nXGBoost model trained!\n")

# ==============================================================================
# 7. VALIDATE MODEL
# ==============================================================================
print_header("Model Validation")

val_result <- validate_model(
    model = model,
    train = train,
    feature_cols = feature_cols,
    target_col = "Time_from_Pickup_to_Arrival",
    val_ratio = 0.25
)

cat("\nValidation complete!\n")

# ==============================================================================
# 8. GENERATE PREDICTIONS
# ==============================================================================
print_header("Generating Predictions")

# Create test matrix
test_matrix <- create_test_matrix(test, feature_cols)

# Make predictions
predictions <- predict_xgboost(model, test_matrix)

cat("\nPredictions generated!\n")

# ==============================================================================
# 9. CREATE SUBMISSION
# ==============================================================================
print_header("Creating Submission")

# Create submission
submission <- create_submission(test, predictions, id_col = "Order_No")

# Validate submission
validate_submission(submission, expected_rows = nrow(test))

# Show statistics
submission_stats(submission)

# Save submission
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
filename <- paste0("submission_xgb_", timestamp, ".csv")
filepath <- save_submission(submission, filename)

# ==============================================================================
# 10. SAVE MODEL
# ==============================================================================
print_header("Saving Model")

if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
}

# Save XGBoost model
model_path <- file.path(OUTPUT_DIR, "xgboost_model.rds")
saveRDS(list(
    model = model,
    feature_cols = feature_cols,
    importance = importance,
    params = XGBOOST_PARAMS,
    timestamp = Sys.time()
), model_path)

cat("Model saved to:", model_path, "\n")

# ==============================================================================
# 11. SUMMARY
# ==============================================================================
cat("\n")
cat("================================================================\n")
cat("                    PIPELINE COMPLETE!\n")
cat("================================================================\n")
cat("\n")
cat("Summary:\n")
cat("  - Training rows:", nrow(train), "\n")
cat("  - Test rows:", nrow(test), "\n")
cat("  - Features:", length(feature_cols), "\n")
cat("  - XGBoost rounds:", xgb_result$nrounds, "\n")
cat("  - Validation RMSE:", round(val_result$rmse, 4), "\n")
cat("  - Validation MAE:", round(val_result$mae, 4), "\n")
cat("\n")
cat("Output files:\n")
cat("  - Submission:", filepath, "\n")
cat("  - Model:", model_path, "\n")
cat("\n")
cat("================================================================\n")
