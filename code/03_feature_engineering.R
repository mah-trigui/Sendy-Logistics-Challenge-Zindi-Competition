# ==============================================================================
# 03_feature_engineering.R - Feature Engineering
# ==============================================================================
# Project: Sendy Logistics - Delivery Time Prediction
# Description: Create features from raw data
# ==============================================================================

source("00_config.R")
print_header("Step 3: Feature Engineering")

# ==============================================================================
# 1. RIDER FEATURES
# ==============================================================================

#' Enrich riders data with basic features
enrich_riders_basic <- function(rd) {
    cat("Enriching riders with basic features...\n")

    # Average orders per day
    rd$avg_order_day <- rd$No_Of_Orders / rd$Age

    # Percentage of orders rated
    rd$P_order_rated <- rd$No_of_Ratings / rd$No_Of_Orders

    cat("  Basic rider features created\n")
    return(rd)
}

#' Calculate rider performance stats from training data
calculate_rider_stats <- function(train, rd) {
    cat("Calculating rider performance statistics...\n")

    # Speed statistics
    Rider_speed_mean <- train %>%
        group_by(Rider_Id) %>%
        summarize(speed_avg = mean(speed, na.rm = TRUE))
    Rider_speed_med <- train %>%
        group_by(Rider_Id) %>%
        summarize(speed_med = median(speed, na.rm = TRUE))

    # Time statistics
    Rider_arriv_pick_med <- train %>%
        group_by(Rider_Id) %>%
        summarize(arriv_pick_med = median(arriv_pick, na.rm = TRUE))

    # Distance statistics
    Rider_dist_mean <- train %>%
        group_by(Rider_Id) %>%
        summarize(dist_avg = mean(Distance__KM_, na.rm = TRUE))
    Rider_dist_med <- train %>%
        group_by(Rider_Id) %>%
        summarize(dist_med = median(Distance__KM_, na.rm = TRUE))

    # Duration statistics
    Rider_dur_mean <- train %>%
        group_by(Rider_Id) %>%
        summarize(dur_avg = mean(Time_from_Pickup_to_Arrival, na.rm = TRUE))
    Rider_dur_med <- train %>%
        group_by(Rider_Id) %>%
        summarize(dur_med = median(Time_from_Pickup_to_Arrival, na.rm = TRUE))

    # Order count
    Rider_n <- train %>%
        group_by(Rider_Id) %>%
        summarize(Nb_ord = n())

    # Join all stats to riders
    rd <- rd %>%
        left_join(Rider_speed_mean, by = "Rider_Id") %>%
        left_join(Rider_speed_med, by = "Rider_Id") %>%
        left_join(Rider_arriv_pick_med, by = "Rider_Id") %>%
        left_join(Rider_dist_mean, by = "Rider_Id") %>%
        left_join(Rider_dist_med, by = "Rider_Id") %>%
        left_join(Rider_dur_mean, by = "Rider_Id") %>%
        left_join(Rider_dur_med, by = "Rider_Id") %>%
        left_join(Rider_n, by = "Rider_Id")

    cat("  Rider statistics calculated\n")
    return(rd)
}

#' Calculate rider quality flags from training data
calculate_rider_flags <- function(train, rd) {
    cat("Calculating rider quality flags...\n")

    # Distance/Duration flags for each order
    train$flag_dist <- ifelse(train$diff_dist <= 0, 1, 0)
    train$flag_dur <- ifelse(train$diff_dur <= 0, 1, 0)
    train$best <- train$flag_dist + train$flag_dur

    # Aggregate flags by rider using SQL
    vars <- sqldf("SELECT Rider_Id,
    COUNT(CASE WHEN flag_dist=0 AND flag_dur=0 THEN Order_No END) AS V00,
    COUNT(CASE WHEN flag_dist=0 AND flag_dur=1 THEN Order_No END) AS V01,
    COUNT(CASE WHEN flag_dist=1 AND flag_dur=0 THEN Order_No END) AS V10,
    COUNT(CASE WHEN flag_dist=1 AND flag_dur=1 THEN Order_No END) AS V11,
    COUNT(CASE WHEN speed < 7 THEN Order_No END) AS Outlier_Speed_Less_,
    COUNT(CASE WHEN speed > 50 THEN Order_No END) AS Outlier_Speed_More_,
    COUNT(CASE WHEN Confirmation___Time IS NULL OR Confirmation___Time='NA'
          THEN Order_No END) AS Nb_Time_miss_
    FROM train GROUP BY Rider_Id")

    # Join to riders
    rd <- rd %>% left_join(vars, by = "Rider_Id")

    # Calculate percentages
    rd$P00 <- rd$V00 / rd$Nb_ord
    rd$P01 <- rd$V01 / rd$Nb_ord
    rd$P10 <- rd$V10 / rd$Nb_ord
    rd$P11 <- rd$V11 / rd$Nb_ord
    rd$Nb_Time_miss <- rd$Nb_Time_miss_ / rd$Nb_ord
    rd$Outlier_Speed_Less <- rd$Outlier_Speed_Less_ / rd$Nb_ord
    rd$Outlier_Speed_More <- rd$Outlier_Speed_More_ / rd$Nb_ord

    # Best rider score
    Rider_best <- train %>%
        group_by(Rider_Id) %>%
        summarize(best_rider = mean(best, na.rm = TRUE))
    rd <- rd %>% left_join(Rider_best, by = "Rider_Id")

    # Overall score
    rd$score <- rd$best_rider / rd$Nb_ord

    cat("  Rider quality flags calculated\n")
    return(rd)
}

# ==============================================================================
# 2. LOCATION CLUSTERING
# ==============================================================================

#' Create location clusters using K-Means
create_location_clusters <- function(train, n_clusters = c(10, 15)) {
    cat("Creating location clusters...\n")

    # Get unique locations
    location_pick <- train %>% select(Pickup_Lat, Pickup_Long)
    location_dest <- train %>% select(Destination_Lat, Destination_Long)

    colnames(location_pick) <- c("lat", "long")
    colnames(location_dest) <- c("lat", "long")

    location <- rbind(location_pick, location_dest)
    location <- sqldf("SELECT DISTINCT lat, long FROM location")

    # K-means clustering
    set.seed(GLOBAL_SEED)
    clusters <- list()

    for (k in n_clusters) {
        k_model <- kmeans(location, k)
        location[[paste0("cluster_", k)]] <- k_model$cluster
        clusters[[paste0("k", k)]] <- k_model
    }

    cat("  Created clusters with k =", paste(n_clusters, collapse = ", "), "\n")

    return(list(location = location, clusters = clusters))
}

#' Assign cluster IDs to a dataset
assign_clusters <- function(df, cluster_data, n_clusters = c(10, 15)) {
    cat("Assigning cluster IDs...\n")

    for (k in n_clusters) {
        k_model <- cluster_data$clusters[[paste0("k", k)]]

        # Pickup clusters
        pickup_cluster <- get.knnx(
            k_model$centers,
            df[, c("Pickup_Lat", "Pickup_Long")], 1
        )$nn.index[, 1]
        df[[paste0("class_pick_", k)]] <- as.factor(pickup_cluster)

        # Destination clusters
        dest_cluster <- get.knnx(
            k_model$centers,
            df[, c("Destination_Lat", "Destination_Long")], 1
        )$nn.index[, 1]
        df[[paste0("class_dest_", k)]] <- as.factor(dest_cluster)
    }

    cat("  Clusters assigned\n")
    return(df)
}

# ==============================================================================
# 3. CATEGORICAL BINNING
# ==============================================================================

#' Create distance bins
create_distance_bins <- function(df) {
    cat("Creating distance bins...\n")

    df$split_dist <- NA
    df$split_dist[df$Distance__KM_ <= 4] <- "VS"
    df$split_dist[df$Distance__KM_ >= 5 & df$Distance__KM_ <= 9] <- "S"
    df$split_dist[df$Distance__KM_ >= 10 & df$Distance__KM_ <= 13] <- "M"
    df$split_dist[df$Distance__KM_ >= 14] <- "L"
    df$split_dist <- as.factor(df$split_dist)

    return(df)
}

#' Create hour bins
create_hour_bins <- function(df) {
    cat("Creating hour bins...\n")

    df$split_hour <- NA
    df$split_hour[df$Hour >= 0 & df$Hour <= 9] <- "A"
    df$split_hour[df$Hour >= 10 & df$Hour <= 11] <- "B"
    df$split_hour[df$Hour >= 12 & df$Hour <= 13] <- "C"
    df$split_hour[df$Hour == 14] <- "D"
    df$split_hour[df$Hour >= 15 & df$Hour <= 23] <- "E"
    df$split_hour[df$Hour == 25] <- "F"
    df$split_hour <- as.factor(df$split_hour)

    return(df)
}

#' Create rider-based bins after joining rider data
create_rider_bins <- function(df) {
    cat("Creating rider-based bins...\n")

    # Age bins
    df$split_age <- NA
    df$split_age[df$Age <= 448] <- "VLV"
    df$split_age[df$Age >= 449 & df$Age <= 739] <- "LV"
    df$split_age[df$Age >= 740 & df$Age <= 964] <- "MV"
    df$split_age[df$Age >= 965 & df$Age <= 1426] <- "HV"
    df$split_age[df$Age >= 1427] <- "VHV"
    df$split_age <- as.factor(df$split_age)

    # Order count bins
    df$split_No_Order <- NA
    df$split_No_Order[df$No_Of_Orders <= 889] <- "VLV"
    df$split_No_Order[df$No_Of_Orders >= 890 & df$No_Of_Orders <= 1591] <- "LV"
    df$split_No_Order[df$No_Of_Orders >= 1592 & df$No_Of_Orders <= 2608] <- "MV"
    df$split_No_Order[df$No_Of_Orders >= 2609] <- "VHV"
    df$split_No_Order <- as.factor(df$split_No_Order)

    # Speed bins
    df$split_speed <- NA
    df$split_speed[df$speed_med <= 19] <- "VLV"
    df$split_speed[df$speed_med > 19 & df$speed_med <= 24] <- "LV"
    df$split_speed[df$speed_med > 24 & df$speed_med <= 30] <- "MV"
    df$split_speed[df$speed_med > 30] <- "HV"
    df$split_speed <- as.factor(df$split_speed)

    # Score bins
    df$split_score <- NA
    df$split_score[df$best_rider <= 0.9] <- "VLV"
    df$split_score[df$best_rider > 0.9 & df$best_rider <= 1.12] <- "LV"
    df$split_score[df$best_rider > 1.12 & df$best_rider <= 1.21] <- "MV"
    df$split_score[df$best_rider > 1.21] <- "HV"
    df$split_score <- as.factor(df$split_score)

    return(df)
}

# ==============================================================================
# 4. CALCULATED FEATURES
# ==============================================================================

#' Create outlier speed flag
create_speed_outliers <- function(df) {
    cat("Creating speed outlier flags...\n")

    if ("speed" %in% names(df)) {
        df$Outlier_Speed <- ifelse(df$speed < 7 | df$speed > 50, 1, 0)
        df$Outlier_Speed_Less <- ifelse(df$speed < 7, 1, 0)
        df$Outlier_Speed_More <- ifelse(df$speed > 50, 1, 0)
        cat("  Speed outlier flags created\n")
    }

    return(df)
}

#' Create ratio and difference features
create_calculated_features <- function(df) {
    cat("Creating calculated features...\n")

    # Distance difference (from route data if available)
    if (all(c("dist", "Distance__KM_") %in% names(df))) {
        df$diff_dist <- (df$Distance__KM_ * 1000) - df$dist
        df$P_diff_dist <- df$diff_dist / (df$Distance__KM_ * 1000)
    }

    # Time ratios
    if (all(c("confir_arriv", "plac_confir") %in% names(df))) {
        df$P_conf_plac <- df$confir_arriv / df$plac_confir
    }

    if (all(c("arriv_pick", "confir_arriv") %in% names(df))) {
        df$P_arriv_conf <- df$arriv_pick / df$confir_arriv
    }

    if (all(c("confir_pick", "Time_from_Pickup_to_Arrival") %in% names(df))) {
        df$P_pick_time <- df$confir_pick / df$Time_from_Pickup_to_Arrival
    }

    cat("  Calculated features created\n")
    return(df)
}

#' Create estimated duration features
create_duration_estimates <- function(df) {
    cat("Creating duration estimates...\n")

    if (all(c("dist", "Distance__KM_", "speed_med", "speed_avg", "arriv_pick_med") %in% names(df))) {
        # Cap outliers
        df$speed_med[df$speed_med > 48] <- 48
        df$arriv_pick_med[df$arriv_pick_med > 528] <- 528

        # Average distance
        df$dis <- ((df$dist / 1000) + df$Distance__KM_) / 2

        # Estimated duration
        df$dur_estim <- df$dis / df$speed_med * 60 * 60
        df$dur_estim_avg <- df$dis / df$speed_avg * 60 * 60

        # Normalized duration
        df$dur_nor <- df$arriv_pick_med + df$dur_estim

        # Improvement ratio
        df$imp <- df$dur_nor / df$arriv_pick

        if ("dur" %in% names(df)) {
            df$imp_b <- df$dur_nor / df$dur
        }

        cat("  Duration estimates created\n")
    }

    return(df)
}

# ==============================================================================
# 5. MASTER FEATURE ENGINEERING
# ==============================================================================

#' Full feature engineering pipeline for training data
engineer_train_features <- function(train, riders) {
    print_header("Feature Engineering - Training Data")

    # 1. Enrich riders
    riders <- enrich_riders_basic(riders)
    riders <- calculate_rider_stats(train, riders)
    riders <- calculate_rider_flags(train, riders)

    # 2. Create location clusters
    cluster_data <- create_location_clusters(train)

    # 3. Assign clusters to training data
    train <- assign_clusters(train, cluster_data)

    # 4. Join rider data
    rider_cols <- c(
        "Rider_Id", "No_Of_Orders", "Age", "Average_Rating", "No_of_Ratings",
        "avg_order_day", "P_order_rated", "speed_avg", "speed_med",
        "arriv_pick_med", "dist_avg", "dist_med", "dur_avg", "dur_med",
        "Nb_ord", "best_rider", "score",
        "P00", "P01", "P10", "P11",
        "Nb_Time_miss", "Outlier_Speed_Less", "Outlier_Speed_More"
    )
    rider_cols <- rider_cols[rider_cols %in% names(riders)]
    train <- train %>% left_join(riders[, rider_cols], by = "Rider_Id")

    # 5. Create bins
    train <- create_distance_bins(train)
    train <- create_hour_bins(train)
    train <- create_rider_bins(train)

    # 6. Create calculated features and outliers
    train <- create_speed_outliers(train)
    train <- create_calculated_features(train)
    train <- create_duration_estimates(train)

    cat("\nTraining features engineered!\n")

    return(list(
        train = train,
        riders = riders,
        cluster_data = cluster_data
    ))
}

#' Feature engineering for test data (using pre-computed values)
engineer_test_features <- function(test, riders, cluster_data) {
    print_header("Feature Engineering - Test Data")

    # 1. Assign clusters
    test <- assign_clusters(test, cluster_data)

    # 2. Join rider data
    rider_cols <- names(riders)[names(riders) %in% c(
        "Rider_Id", "No_Of_Orders", "Age", "Average_Rating", "No_of_Ratings",
        "avg_order_day", "P_order_rated", "speed_avg", "speed_med",
        "arriv_pick_med", "dist_avg", "dist_med", "dur_avg", "dur_med",
        "Nb_ord", "best_rider", "score",
        "P00", "P01", "P10", "P11",
        "Nb_Time_miss", "Outlier_Speed_Less", "Outlier_Speed_More"
    )]
    test <- test %>% left_join(riders[, rider_cols], by = "Rider_Id")

    # 3. Create bins
    test <- create_distance_bins(test)
    test <- create_hour_bins(test)
    test <- create_rider_bins(test)

    # 4. Create calculated features and outliers
    test <- create_speed_outliers(test)
    test <- create_calculated_features(test)
    test <- create_duration_estimates(test)

    cat("\nTest features engineered!\n")

    return(test)
}

cat("\nFeature engineering functions ready!\n")
