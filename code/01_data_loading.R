# ==============================================================================
# 01_data_loading.R - Load Raw Data
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Load training, test, and riders data
# ==============================================================================

source("00_config.R")
print_header("Step 1: Data Loading")

# ==============================================================================
# 1. LOAD TRAINING DATA
# ==============================================================================

#' Load training dataset
load_train_data <- function(filepath = TRAIN_FILE) {
    cat("Loading training data...\n")

    df <- as.data.frame(read.csv(filepath, stringsAsFactors = FALSE))

    # Standardize column names (replace dots with underscores)
    names(df) <- gsub("\\.", "_", names(df))

    cat("  Rows:", nrow(df), "\n")
    cat("  Columns:", ncol(df), "\n")
    cat(
        "  Target (Time_from_Pickup_to_Arrival) range:",
        min(df$Time_from_Pickup_to_Arrival, na.rm = TRUE), "-",
        max(df$Time_from_Pickup_to_Arrival, na.rm = TRUE), "\n"
    )

    return(df)
}

# ==============================================================================
# 2. LOAD TEST DATA
# ==============================================================================

#' Load test dataset
load_test_data <- function(filepath = TEST_FILE) {
    cat("Loading test data...\n")

    df <- as.data.frame(read.csv(filepath, stringsAsFactors = FALSE))

    # Standardize column names
    names(df) <- gsub("\\.", "_", names(df))

    cat("  Rows:", nrow(df), "\n")
    cat("  Columns:", ncol(df), "\n")

    return(df)
}

# ==============================================================================
# 3. LOAD RIDERS DATA
# ==============================================================================

#' Load riders dataset
load_riders_data <- function(filepath = RIDERS_FILE) {
    cat("Loading riders data...\n")

    rd <- as.data.frame(read.csv(filepath, stringsAsFactors = FALSE))

    # Standardize column names
    names(rd) <- gsub("\\.", "_", names(rd))

    cat("  Rows:", nrow(rd), "\n")
    cat("  Columns:", ncol(rd), "\n")
    cat("  Unique riders:", length(unique(rd$Rider_Id)), "\n")

    return(rd)
}

# ==============================================================================
# 4. LOAD ROUTE DATA (IF AVAILABLE)
# ==============================================================================

#' Load pre-computed route data
load_route_data <- function(filepath) {
    if (!file.exists(filepath)) {
        cat("Route data file not found:", filepath, "\n")
        return(NULL)
    }

    cat("Loading route data...\n")

    route <- as.data.frame(read.csv(filepath, stringsAsFactors = FALSE))
    names(route) <- gsub("\\.", "_", names(route))

    cat("  Rows:", nrow(route), "\n")

    return(route)
}

# ==============================================================================
# 5. LOAD ALL DATA
# ==============================================================================

#' Load all datasets at once
load_all_data <- function(train_file = TRAIN_FILE,
                          test_file = TEST_FILE,
                          riders_file = RIDERS_FILE) {
    print_header("Loading All Data")

    train <- load_train_data(train_file)
    test <- load_test_data(test_file)
    riders <- load_riders_data(riders_file)

    cat("\nAll data loaded successfully!\n")

    return(list(
        train = train,
        test = test,
        riders = riders
    ))
}

# ==============================================================================
# 6. DATA VALIDATION
# ==============================================================================

#' Validate data structure
validate_data <- function(train, test, riders) {
    cat("\nValidating data structure...\n")

    # Check required columns in train
    train_required <- c(
        "Order_No", "Rider_Id", "Time_from_Pickup_to_Arrival",
        "Distance__KM_", "Pickup_Lat", "Pickup_Long",
        "Destination_Lat", "Destination_Long"
    )

    missing_train <- setdiff(train_required, names(train))
    if (length(missing_train) > 0) {
        warning("Missing columns in train: ", paste(missing_train, collapse = ", "))
    }

    # Check required columns in test
    test_required <- c(
        "Order_No", "Rider_Id", "Distance__KM_",
        "Pickup_Lat", "Pickup_Long",
        "Destination_Lat", "Destination_Long"
    )

    missing_test <- setdiff(test_required, names(test))
    if (length(missing_test) > 0) {
        warning("Missing columns in test: ", paste(missing_test, collapse = ", "))
    }

    # Check riders linkage
    train_riders <- unique(train$Rider_Id)
    test_riders <- unique(test$Rider_Id)
    known_riders <- unique(riders$Rider_Id)

    missing_train_riders <- setdiff(train_riders, known_riders)
    missing_test_riders <- setdiff(test_riders, known_riders)

    cat("  Training riders not in Riders file:", length(missing_train_riders), "\n")
    cat("  Test riders not in Riders file:", length(missing_test_riders), "\n")

    cat("Data validation complete!\n")

    return(invisible(TRUE))
}

cat("\nData loading functions ready!\n")
