# ==============================================================================
# 02_data_cleaning.R - Data Cleaning & Preprocessing
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Clean data, parse times, handle missing values
# ==============================================================================

source("00_config.R")
print_header("Step 2: Data Cleaning")

# ==============================================================================
# 1. PARSE TIME COLUMNS
# ==============================================================================

#' Parse time columns from string to time format
parse_time_columns <- function(df) {
    cat("Parsing time columns...\n")

    time_cols <- c(
        "Placement___Time", "Confirmation___Time",
        "Arrival_at_Pickup___Time", "Pickup___Time",
        "Arrival_at_Destination___Time"
    )

    for (col in time_cols) {
        if (col %in% names(df)) {
            df[[col]] <- parse_time(as.character(df[[col]]), "%I:%M:%S %p")
        }
    }

    cat("  Time columns parsed\n")
    return(df)
}

# ==============================================================================
# 2. CONVERT DATA TYPES
# ==============================================================================

#' Convert columns to appropriate data types
convert_data_types <- function(df) {
    cat("Converting data types...\n")

    # Factors
    factor_cols <- c("Platform_Type", "Personal_or_Business")
    for (col in factor_cols) {
        if (col %in% names(df)) {
            df[[col]] <- as.factor(df[[col]])
        }
    }

    # Numeric columns
    numeric_cols <- c(
        "Distance__KM_", "Time_from_Pickup_to_Arrival",
        "Pickup_Lat", "Pickup_Long",
        "Destination_Lat", "Destination_Long",
        "Temperature"
    )
    for (col in numeric_cols) {
        if (col %in% names(df)) {
            df[[col]] <- as.numeric(df[[col]])
        }
    }

    cat("  Data types converted\n")
    return(df)
}

# ==============================================================================
# 3. EXTRACT TIME FEATURES
# ==============================================================================

#' Extract hour from pickup time
extract_hour_feature <- function(df) {
    cat("Extracting hour feature...\n")

    if ("Pickup___Time" %in% names(df)) {
        df$Pick_H <- hour(df$Pickup___Time)
        df$Hour <- ifelse(is.na(df$Pick_H), 25, df$Pick_H) # 25 = missing
        df$Hour_f <- as.factor(df$Hour)
    }

    cat("  Hour extracted\n")
    return(df)
}

#' Calculate time differences between stages
calculate_time_diffs <- function(df) {
    cat("Calculating time differences...\n")

    # Placement to Confirmation
    if (all(c("Confirmation___Time", "Placement___Time") %in% names(df))) {
        df$plac_confir <- as.numeric(df$Confirmation___Time - df$Placement___Time)
    }

    # Confirmation to Arrival at Pickup
    if (all(c("Arrival_at_Pickup___Time", "Confirmation___Time") %in% names(df))) {
        df$confir_arriv <- as.numeric(df$Arrival_at_Pickup___Time - df$Confirmation___Time)
    }

    # Placement to Arrival at Pickup
    if (all(c("Arrival_at_Pickup___Time", "Placement___Time") %in% names(df))) {
        df$plac_arriv <- as.numeric(df$Arrival_at_Pickup___Time - df$Placement___Time)
    }

    # Arrival at Pickup to Pickup
    if (all(c("Pickup___Time", "Arrival_at_Pickup___Time") %in% names(df))) {
        df$arriv_pick <- as.numeric(df$Pickup___Time - df$Arrival_at_Pickup___Time)
    }

    # Confirmation to Pickup
    if (all(c("Pickup___Time", "Confirmation___Time") %in% names(df))) {
        df$confir_pick <- as.numeric(df$Pickup___Time - df$Confirmation___Time)
    }

    cat("  Time differences calculated\n")
    return(df)
}

# ==============================================================================
# 4. HANDLE MISSING VALUES
# ==============================================================================

#' Handle missing temperature values
handle_missing_temperature <- function(df) {
    if ("Temperature" %in% names(df)) {
        cat("Handling missing temperature...\n")
        mean_temp <- mean(df$Temperature, na.rm = TRUE)
        df$Temp <- ifelse(is.na(df$Temperature), mean_temp, df$Temperature)
        cat("  Missing temperature filled with mean:", round(mean_temp, 2), "\n")
    }

    return(df)
}

#' Convert NA to NaN for numeric columns (for XGBoost compatibility)
convert_na_to_nan <- function(df) {
    cat("Converting NA to NaN...\n")

    df <- as.data.table(df)

    numeric_cols <- names(df)[sapply(df, is.numeric)]
    for (col in numeric_cols) {
        df[is.na(get(col)), (col) := NaN]
    }

    df <- as.data.frame(df)
    cat("  NAs converted\n")
    return(df)
}

# ==============================================================================
# 5. CALCULATE HAVERSINE DISTANCE
# ==============================================================================

#' Calculate haversine distance between pickup and destination
calculate_haversine <- function(df) {
    cat("Calculating haversine distances...\n")

    if (all(c("Pickup_Lat", "Pickup_Long", "Destination_Lat", "Destination_Long") %in% names(df))) {
        df$d_haver <- NA

        for (i in 1:nrow(df)) {
            tryCatch(
                {
                    df$d_haver[i] <- haversine(
                        c(df$Pickup_Lat[i], df$Pickup_Long[i]),
                        c(df$Destination_Lat[i], df$Destination_Long[i])
                    ) * 1000 # Convert to meters
                },
                error = function(e) {}
            )
        }

        cat("  Haversine distances calculated\n")
    }

    return(df)
}

# ==============================================================================
# 6. MASTER CLEANING FUNCTION
# ==============================================================================

#' Remove invalid delivery times (< 3 minutes)
filter_invalid_deliveries <- function(df) {
    cat("Filtering invalid delivery times...\n")

    if ("Time_from_Pickup_to_Arrival" %in% names(df)) {
        original_count <- nrow(df)
        df <- df[df$Time_from_Pickup_to_Arrival > 180, ] # Remove < 3 minutes
        removed_count <- original_count - nrow(df)
        cat("  Removed", removed_count, "invalid deliveries (< 180 seconds)\n")
    }

    return(df)
}

#' Handle precipitation missing values (assume 0)
handle_missing_precipitation <- function(df) {
    if ("Precipitation_in_millimeters" %in% names(df)) {
        cat("Handling missing precipitation...\n")
        na_count <- sum(is.na(df$Precipitation_in_millimeters))
        df$Precipitation_in_millimeters <- ifelse(
            is.na(df$Precipitation_in_millimeters),
            0,
            df$Precipitation_in_millimeters
        )
        cat("  Filled", na_count, "missing precipitation values with 0\n")
    }
    return(df)
}

#' Clean training data (full pipeline)
clean_train_data <- function(df) {
    print_header("Cleaning Training Data")

    df <- parse_time_columns(df)
    df <- convert_data_types(df)
    df <- filter_invalid_deliveries(df) # Remove invalid deliveries
    df <- extract_hour_feature(df)
    df <- calculate_time_diffs(df)
    df <- handle_missing_precipitation(df) # Handle precipitation
    df <- handle_missing_temperature(df)
    df <- calculate_haversine(df)

    # Calculate speed
    if (all(c("Distance__KM_", "Time_from_Pickup_to_Arrival") %in% names(df))) {
        df$speed <- df$Distance__KM_ / (df$Time_from_Pickup_to_Arrival / 60 / 60)
        cat("  Speed calculated\n")
    }

    cat("\nTraining data cleaned!\n")
    return(df)
}

#' Clean test data (full pipeline)
clean_test_data <- function(df) {
    print_header("Cleaning Test Data")

    df <- parse_time_columns(df)
    df <- convert_data_types(df)
    df <- extract_hour_feature(df)
    df <- calculate_time_diffs(df)
    df <- handle_missing_precipitation(df) # Handle precipitation
    df <- handle_missing_temperature(df)
    df <- calculate_haversine(df)

    cat("\nTest data cleaned!\n")
    return(df)
}

cat("\nData cleaning functions ready!\n")
