# ==============================================================================
# 07_submission.R - Generate Competition Submission
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Generate submission file in required format
# ==============================================================================

source("00_config.R")
print_header("Step 7: Generate Submission")

# ==============================================================================
# 1. CREATE SUBMISSION
# ==============================================================================

#' Generate submission dataframe
create_submission <- function(test, predictions, id_col = "Order_No") {
    cat("Creating submission...\n")

    # Ensure non-negative predictions
    predictions[predictions < 0] <- 1

    # Create submission
    submission <- data.frame(
        Order_No = test[[id_col]],
        Time_from_Pickup_to_Arrival = predictions
    )

    # Rename for submission format
    names(submission) <- c("Order No", "Time from Pickup to Arrival")

    cat("  Rows:", nrow(submission), "\n")
    cat("  Predictions range:", round(min(predictions), 2), "-", round(max(predictions), 2), "\n")
    cat("  Mean prediction:", round(mean(predictions), 2), "\n")

    return(submission)
}

# ==============================================================================
# 2. SAVE SUBMISSION
# ==============================================================================

#' Save submission to CSV
save_submission <- function(submission, filename = NULL, output_dir = SUBMISSION_DIR) {
    # Create directory if needed
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    # Generate filename if not provided
    if (is.null(filename)) {
        timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
        filename <- paste0("submission_", timestamp, ".csv")
    }

    filepath <- file.path(output_dir, filename)

    # Write CSV
    write.csv(submission, filepath, row.names = FALSE)

    cat("\nSubmission saved to:", filepath, "\n")

    return(filepath)
}

# ==============================================================================
# 3. VALIDATE SUBMISSION
# ==============================================================================

#' Validate submission format
validate_submission <- function(submission, expected_rows = NULL) {
    cat("Validating submission...\n")

    valid <- TRUE

    # Check columns
    expected_cols <- c("Order No", "Time from Pickup to Arrival")
    if (!all(expected_cols %in% names(submission))) {
        cat("  ERROR: Missing columns\n")
        valid <- FALSE
    }

    # Check for missing values
    if (any(is.na(submission))) {
        cat("  ERROR: Contains NA values\n")
        valid <- FALSE
    }

    # Check for negative predictions
    if (any(submission$`Time from Pickup to Arrival` < 0)) {
        cat("  ERROR: Contains negative predictions\n")
        valid <- FALSE
    }

    # Check row count
    if (!is.null(expected_rows) && nrow(submission) != expected_rows) {
        cat("  ERROR: Expected", expected_rows, "rows, got", nrow(submission), "\n")
        valid <- FALSE
    }

    if (valid) {
        cat("  Submission is valid!\n")
    }

    return(valid)
}

# ==============================================================================
# 4. COMPARE SUBMISSIONS
# ==============================================================================

#' Compare two submissions
compare_submissions <- function(sub1, sub2, name1 = "Submission 1", name2 = "Submission 2") {
    cat("Comparing submissions...\n")

    # Merge on Order No
    merged <- merge(sub1, sub2, by = "Order No", suffixes = c("_1", "_2"))

    # Calculate differences
    diff <- merged$`Time from Pickup to Arrival_1` - merged$`Time from Pickup to Arrival_2`

    cat("\n  Mean difference:", round(mean(diff), 2), "\n")
    cat("  SD of differences:", round(sd(diff), 2), "\n")
    cat("  Max absolute difference:", round(max(abs(diff)), 2), "\n")

    # Correlation
    corr <- cor(merged$`Time from Pickup to Arrival_1`, merged$`Time from Pickup to Arrival_2`)
    cat("  Correlation:", round(corr, 4), "\n")

    return(invisible(diff))
}

# ==============================================================================
# 5. SUBMISSION STATISTICS
# ==============================================================================

#' Print submission statistics
submission_stats <- function(submission) {
    preds <- submission$`Time from Pickup to Arrival`

    cat("\nSubmission Statistics:\n")
    cat("  Count:", length(preds), "\n")
    cat("  Min:", round(min(preds), 2), "\n")
    cat("  Max:", round(max(preds), 2), "\n")
    cat("  Mean:", round(mean(preds), 2), "\n")
    cat("  Median:", round(median(preds), 2), "\n")
    cat("  SD:", round(sd(preds), 2), "\n")

    # Distribution
    cat("\n  Distribution:\n")
    cat("    < 1000:", sum(preds < 1000), "\n")
    cat("    1000-2000:", sum(preds >= 1000 & preds < 2000), "\n")
    cat("    2000-3000:", sum(preds >= 2000 & preds < 3000), "\n")
    cat("    3000-4000:", sum(preds >= 3000 & preds < 4000), "\n")
    cat("    >= 4000:", sum(preds >= 4000), "\n")

    return(invisible(NULL))
}

# ==============================================================================
# 6. ENSEMBLE SUBMISSION
# ==============================================================================

#' Generate ensemble submission from multiple predictions
create_ensemble_submission <- function(test, predictions_list, weights = NULL,
                                       id_col = "Order_No") {
    cat("Creating ensemble submission...\n")

    n_models <- length(predictions_list)

    if (is.null(weights)) {
        weights <- rep(1 / n_models, n_models)
    }

    # Weighted average
    ensemble_preds <- rep(0, length(predictions_list[[1]]))

    for (i in seq_along(predictions_list)) {
        ensemble_preds <- ensemble_preds + weights[i] * predictions_list[[i]]
    }

    cat("  Ensemble weights:", paste(round(weights, 3), collapse = ", "), "\n")

    # Create submission
    submission <- create_submission(test, ensemble_preds, id_col)

    return(submission)
}

# ==============================================================================
# 7. FULL SUBMISSION PIPELINE
# ==============================================================================

#' Complete submission pipeline
generate_submission <- function(test, model, feature_cols,
                                model_type = "xgboost",
                                filename = NULL) {
    print_header("Generating Submission")

    # Make predictions based on model type
    if (model_type == "xgboost") {
        # Create test matrix
        test_matrix <- create_test_matrix(test, feature_cols)
        predictions <- predict_xgboost(model, test_matrix)
    } else if (model_type == "h2o") {
        h2o_test <- to_h2o_frame(test[, feature_cols])
        predictions <- predict_h2o(model, h2o_test)
    } else if (model_type == "ensemble") {
        predictions <- predict_stacked(
            model$base_models,
            model$meta_model,
            test,
            feature_cols
        )
    } else {
        # Generic caret model
        predictions <- predict(model, test[, feature_cols])
        predictions[predictions < 0] <- 1
    }

    # Create submission
    submission <- create_submission(test, predictions)

    # Validate
    validate_submission(submission, expected_rows = nrow(test))

    # Show statistics
    submission_stats(submission)

    # Save
    filepath <- save_submission(submission, filename)

    cat("\nSubmission pipeline complete!\n")

    return(list(
        submission = submission,
        predictions = predictions,
        filepath = filepath
    ))
}

cat("\nSubmission functions ready!\n")
