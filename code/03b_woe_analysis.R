# ==============================================================================
# 03b_woe_analysis.R - Weight of Evidence Analysis (Optional)
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: WOE/IV analysis for feature selection and binning
# ==============================================================================
# This script is optional and requires the 'Information' package
# It helps identify important features using Information Value
# ==============================================================================

source("00_config.R")
print_header("WoE & Information Value Analysis")

# ==============================================================================
# WEIGHT OF EVIDENCE (WOE) FUNCTIONS
# ==============================================================================

#' Calculate Information Value tables for feature selection
#'
#' @param data Data frame with features
#' @param target_col Name of target variable (must be binary 0/1 for WOE)
#' @param n_bins Number of bins for continuous variables
#' @return Information tables with IV scores
#'
calculate_woe_tables <- function(data, target_col = "flag_dur", n_bins = 5) {
    if (!requireNamespace("Information", quietly = TRUE)) {
        cat("WARNING: Information package not installed. Skipping WOE analysis.\n")
        cat("Install with: install.packages('Information')\n")
        return(NULL)
    }

    cat("Calculating WoE and Information Value...\n")
    cat("  Target variable:", target_col, "\n")
    cat("  Number of bins:", n_bins, "\n")

    # Ensure target is binary
    if (!all(unique(data[[target_col]]) %in% c(0, 1))) {
        cat("ERROR: Target variable must be binary (0/1) for WOE analysis\n")
        return(NULL)
    }

    tryCatch(
        {
            info_tables <- Information::create_infotables(
                data = data,
                y = target_col,
                bins = n_bins,
                parallel = TRUE
            )

            cat("\nInformation Value Summary:\n")
            print(info_tables$Summary)

            cat("\nVariable Importance (by IV):\n")
            iv_summary <- info_tables$Summary[order(-info_tables$Summary$IV), ]
            print(head(iv_summary, 20))

            return(info_tables)
        },
        error = function(e) {
            cat("ERROR in WOE calculation:", e$message, "\n")
            return(NULL)
        }
    )
}

#' Create binary flag for duration outliers (for WOE analysis)
#'
#' @param data Training data with Time_from_Pickup_to_Arrival
#' @param threshold Threshold in seconds for outlier classification
#' @return Data with flag_dur column
#'
create_duration_flag <- function(data, threshold = NULL) {
    cat("Creating duration flag for WOE analysis...\n")

    if (is.null(threshold)) {
        # Use median + 1.5 * IQR as threshold
        q1 <- quantile(data$Time_from_Pickup_to_Arrival, 0.25, na.rm = TRUE)
        q3 <- quantile(data$Time_from_Pickup_to_Arrival, 0.75, na.rm = TRUE)
        iqr <- q3 - q1
        threshold <- q3 + 1.5 * iqr
        cat("  Auto-calculated threshold:", threshold, "seconds\n")
    }

    data$flag_dur <- ifelse(
        data$Time_from_Pickup_to_Arrival > threshold,
        1, # Outlier
        0 # Normal
    )

    outlier_pct <- mean(data$flag_dur, na.rm = TRUE) * 100
    cat("  Outliers:", outlier_pct, "%\n")

    return(data)
}

#' Run WOE analysis on selected features
#'
#' @param train Training data with features
#' @param feature_cols Columns to analyze
#' @return WOE tables
#'
run_woe_analysis <- function(train, feature_cols = NULL) {
    print_header("Weight of Evidence Analysis")

    # Default feature selection
    if (is.null(feature_cols)) {
        feature_cols <- c(
            "Platform_Type", "Personal_or_Business",
            "Pickup___Weekday__Mo___1_",
            "Distance__KM_", "Hour",
            "plac_confir", "confir_arriv", "plac_arriv",
            "arriv_pick", "confir_pick",
            "No_Of_Orders", "Age", "Average_Rating", "No_of_Ratings",
            "avg_order_day", "P_order_rated",
            "speed_med", "dist_avg", "dur_avg", "dur_med",
            "Nb_ord", "best_rider", "score",
            "P00", "P01", "P11", "P10",
            "Outlier_Speed", "Nb_Time_miss",
            "split_dist", "split_score", "split_age",
            "split_No_Order", "split_speed",
            "Temp"
        )

        # Keep only columns that exist
        feature_cols <- feature_cols[feature_cols %in% names(train)]
    }

    # Create duration flag
    train <- create_duration_flag(train)

    # Select data for analysis
    woe_data <- train[, c(feature_cols, "flag_dur")]

    # Calculate WOE tables
    woe_tables <- calculate_woe_tables(woe_data, "flag_dur", n_bins = 5)

    if (!is.null(woe_tables)) {
        cat("\nWOE Analysis Complete!\n")
        cat("Use woe_tables$Tables to view detailed binning\n")
        cat("Use woe_tables$Summary to view IV scores\n")
    }

    return(woe_tables)
}

# ==============================================================================
# FEATURE SELECTION BASED ON IV
# ==============================================================================

#' Select features based on Information Value threshold
#'
#' @param woe_tables Output from calculate_woe_tables
#' @param iv_threshold Minimum IV to consider variable useful
#' @return Vector of selected feature names
#'
select_features_by_iv <- function(woe_tables, iv_threshold = 0.02) {
    if (is.null(woe_tables)) {
        cat("No WOE tables provided\n")
        return(NULL)
    }

    cat("Selecting features by Information Value...\n")
    cat("  IV Threshold:", iv_threshold, "\n")

    # IV interpretation:
    # < 0.02: Not useful
    # 0.02 - 0.1: Weak
    # 0.1 - 0.3: Medium
    # 0.3 - 0.5: Strong
    # > 0.5: Suspicious (too good, check for leakage)

    summary_df <- woe_tables$Summary
    selected <- summary_df[summary_df$IV >= iv_threshold, ]

    cat("\nFeature Selection Results:\n")
    cat("  Total features analyzed:", nrow(summary_df), "\n")
    cat("  Features selected (IV >=", iv_threshold, "):", nrow(selected), "\n")

    # Breakdown by strength
    weak <- sum(selected$IV >= 0.02 & selected$IV < 0.1)
    medium <- sum(selected$IV >= 0.1 & selected$IV < 0.3)
    strong <- sum(selected$IV >= 0.3 & selected$IV < 0.5)
    suspicious <- sum(selected$IV >= 0.5)

    cat("    Weak (0.02-0.1):", weak, "\n")
    cat("    Medium (0.1-0.3):", medium, "\n")
    cat("    Strong (0.3-0.5):", strong, "\n")
    if (suspicious > 0) {
        cat("    Suspicious (>0.5):", suspicious, "*** CHECK FOR LEAKAGE ***\n")
    }

    cat("\nSelected Features:\n")
    print(selected[order(-selected$IV), c("Variable", "IV")])

    return(selected$Variable)
}

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if (FALSE) {
    # This is example code - not executed automatically

    # Load and prepare data
    source("organized/01_data_loading.R")
    source("organized/02_data_cleaning.R")
    source("organized/03_feature_engineering.R")

    train <- load_train_data()
    riders <- load_riders_data()

    train <- clean_train_data(train)
    result <- engineer_train_features(train, riders)
    train_final <- result$train

    # Run WOE analysis
    woe_results <- run_woe_analysis(train_final)

    # Select important features
    if (!is.null(woe_results)) {
        important_features <- select_features_by_iv(woe_results, iv_threshold = 0.02)

        # View detailed tables for specific features
        print(woe_results$Tables$Distance__KM_)
        print(woe_results$Tables$speed_med)
    }
}

cat("\nWOE analysis functions ready!\n")
cat("Note: Requires 'Information' package\n")
cat("Install with: install.packages('Information')\n\n")
