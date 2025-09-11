# Install packages
install.packages("caret")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("randomForest")
install.packages("xgboost")
install.packages("doParallel") # New package for parallel processing

library(caret)
library(ggplot2)
library(corrplot)
library(randomForest)
library(xgboost)
library(doParallel) # Load the library
# Load required library
library(caret)
# 2) DATA COLLECTION

# Read the Airbnb dataset for Seattle location
listings <- read.csv("listings.csv", stringsAsFactors = FALSE)

# View basic structure and first few rows
str(listings)
head(listings)

# 3) SELECT RELEVANT COLUMNS

# Keep only useful features for price prediction
listings <- listings[, c("id", "price", "neighbourhood", "property_type",
                         "room_type", "bedrooms", "bathrooms", "amenities",
                         "host_since", "host_is_superhost", "number_of_reviews",
                         "review_scores_rating", "availability_30")]
# 4) DATA CLEANING

# 4.1 Remove duplicate listings
listings <- listings[!duplicated(listings$id), ]
# 4.2 Remove rows with missing price values
listings <- listings[!is.na(listings$price), ]
# 4.3 Convert 'price' column from string to numeric
listings$price <- as.numeric(gsub("[\\$,]", "", listings$price))
# 4.4 Remove unrealistic outliers (keep prices between $10 and $1000 per night)
listings <- listings[listings$price > 10 & listings$price < 1000, ]
# 4.5 Handle missing numeric values by replacing with median
listings$number_of_reviews[is.na(listings$number_of_reviews)] <-
    median(listings$number_of_reviews, na.rm = TRUE)

listings$review_scores_rating[is.na(listings$review_scores_rating)] <-
    median(listings$review_scores_rating, na.rm = TRUE)
# 4.6 Handle missing categorical values by replacing with 'Unknown'
listings$host_is_superhost[is.na(listings$host_is_superhost)] <- "Unknown"
# 5) FEATURE ENGINEERING

# 5.1 Total number of amenities
listings$amenities_count <- sapply(strsplit(listings$amenities, ","), length)

# 5.2 Host experience in years
# Assuming the date format is month/day/year, e.g., "4/26/2009"
listings$host_since <- as.Date(listings$host_since, format = "%m/%d/%Y")
listings$host_experience_years <- as.numeric(difftime(Sys.Date(),
                                                     listings$host_since,
                                                     units = "days")) / 365
# Replace NA values with 0
listings$host_experience_years[is.na(listings$host_experience_years)] <- 0
# 5.3 Price per bedroom (avoid division by zero)
listings$price_per_bedroom <- listings$price / pmax(listings$bedrooms, 1)
# 6) ENCODING CATEGORICAL VARIABLES

## Convert to factors
listings$property_type <- as.factor(listings$property_type)
listings$neighbourhood <- as.factor(listings$neighbourhood)
listings$room_type <- as.factor(listings$room_type)
listings$host_is_superhost <- as.factor(listings$host_is_superhost)

## One-hot encoding (dummy variables)
dummy_vars <- dummyVars(" ~ .", data = listings, fullRank = TRUE)
listings_encoded <- data.frame(predict(dummy_vars, newdata = listings))

# 7) NORMALIZATION / SCALING

# Scale numeric features (optional - useful for some models like linear regression)
num_cols <- sapply(listings_encoded, is.numeric)
listings_encoded[num_cols] <- scale(listings_encoded[num_cols])
# 8) TRAIN-TEST SPLIT

set.seed(123) # for reproducibility
train_index <- createDataPartition(listings_encoded$price, p = 0.8, list = FALSE)
train_data <- listings_encoded[train_index, ]
test_data <- listings_encoded[-train_index, ]

# Final check
dim(train_data)
dim(test_data)

# -------------------------------
# 2) EXPLORATORY DATA ANALYSIS (EDA) & FEATURE SELECTION
# -------------------------------

# 2.1 Price distribution
png("price_distribution.png", width = 800, height = 600)
ggplot(listings, aes(x = price)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "black") +
  theme_minimal() +
  ggtitle("Price Distribution") +
  xlab("Price ($)") + ylab("Count")
dev.off()

# 2.2 Price vs Room Type
png("price_vs_room_type.png", width = 800, height = 600)
ggplot(listings, aes(x = room_type, y = price)) +
  geom_boxplot(fill = "lightgreen") +
  theme_minimal() +
  ggtitle("Price vs Room Type") +
  ylab("Price ($)")
dev.off()

# 2.3 Price vs Neighbourhood
png("price_vs_neighbourhood.png", width = 1000, height = 600)
ggplot(listings, aes(x = neighbourhood, y = price)) +
  geom_boxplot(fill = "lightpink") +
  theme_minimal() +
  ggtitle("Price vs Neighbourhood") +
  ylab("Price ($)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
dev.off()

# 2.4 Correlation Heatmap (numeric features)
num_features <- listings[, sapply(listings, is.numeric)]
cor_matrix <- cor(num_features, use = "complete.obs")

png("correlation_heatmap.png", width = 1000, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = "black", addCoef.col = "black")
dev.off()

set.seed(123)

# Train RF on full encoded data
listings_encoded_clean <- na.omit(listings_encoded)

# Use parallel processing to speed up Random Forest training
num_cores <- detectCores() - 1 # Use all but one core
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Reduced 'ntree' for faster training
rf_model <- randomForest(price ~ ., data = listings_encoded_clean, ntree = 100, importance = TRUE)
varImpPlot(rf_model)

# Stop the parallel cluster when done with Random Forest
stopCluster(cl)
registerDoSEQ()

# Save RF feature importance to file
png("rf_feature_importance.png", width = 1000, height = 600)
varImpPlot(rf_model)
dev.off()

# Optional: Faster numeric importance table
rf_importance <- importance(rf_model)
rf_importance_sorted <- rf_importance[order(rf_importance[,1], decreasing = TRUE), ]
print(rf_importance_sorted)

# -------------------------------
# 3) MODEL BUILDING & TRAINING
# -------------------------------

# -------------------------------
# 3) Linear Regression
# -------------------------------
lm_model <- lm(price ~ ., data = train_data)
summary(lm_model)

lm_pred <- predict(lm_model, newdata = test_data)
lm_rmse <- sqrt(mean((test_data$price - lm_pred)^2))
print(paste("Linear Regression RMSE:", round(lm_rmse, 2)))

# -------------------------------
# 4) Random Forest Regression
# -------------------------------
set.seed(123)
# Reduced 'ntree' for faster training
rf_model2 <- randomForest(price ~ ., data = train_data, ntree = 100, mtry = 5)
rf_pred <- predict(rf_model2, newdata = test_data)
rf_rmse <- sqrt(mean((test_data$price - rf_pred)^2))
print(paste("Random Forest RMSE:", round(rf_rmse, 2)))

# Hyperparameter tuning with caret
rf_grid <- expand.grid(mtry = c(3, 5)) # Reduced search grid
train_control <- trainControl(method = "cv", number = 3) # Reduced cross-validation folds
rf_tuned <- train(price ~ ., data = train_data, method = "rf",
                  tuneGrid = rf_grid, trControl = train_control, ntree = 50) # Reduced ntree for tuning
print(rf_tuned)

# -------------------------------
# 5) XGBoost Regression
# -------------------------------
# Prepare data for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "price")]),
                            label = train_data$price)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "price")]),
                           label = test_data$price)

# Train XGBoost model
set.seed(123)
# Further reduced 'nrounds' and 'max.depth' for faster training
# Increased eta slightly to compensate for the reduced nrounds
# Added 'nthread' for explicit parallel processing
xgb_model <- xgboost(data = train_matrix, max.depth = 4, eta = 0.2, nrounds = 30,
                     objective = "reg:squarederror", verbose = 0, nthread = 4)

# Predictions & RMSE
xgb_pred <- predict(xgb_model, test_matrix)
xgb_rmse <- sqrt(mean((test_data$price - xgb_pred)^2))
print(paste("XGBoost RMSE:", round(xgb_rmse, 2)))

# Feature importance for XGBoost
importance_matrix <- xgb.importance(feature_names = colnames(train_data[, -which(names(train_data) == "price")]),
                                    model = xgb_model)

# Display importance
xgb.plot.importance(importance_matrix)

# Save XGBoost importance plot
png("xgb_feature_importance.png", width = 1000, height = 600)
xgb.plot.importance(importance_matrix)
dev.off()

# -------------------------------
# 4) MODEL EVALUATION & INTERPRETATION
# -------------------------------
 
library(Metrics)   # for MAE
library(caret)     # for R2
library(SHAPforxgboost)
 
# ---- 4.1 Evaluation Metrics ----
# MAE
lm_mae <- mae(test_data$price, lm_pred)
rf_mae <- mae(test_data$price, rf_pred)
xgb_mae <- mae(test_data$price, xgb_pred)
 
# R-squared
lm_r2 <- R2(lm_pred, test_data$price)
rf_r2 <- R2(rf_pred, test_data$price)
xgb_r2 <- R2(xgb_pred, test_data$price)
 
# Print results
cat("\n--- Model Performance ---\n")
cat(sprintf("Linear Regression -> RMSE: %.2f | MAE: %.2f | R²: %.3f\n", lm_rmse, lm_mae, lm_r2))
cat(sprintf("Random Forest     -> RMSE: %.2f | MAE: %.2f | R²: %.3f\n", rf_rmse, rf_mae, rf_r2))
cat(sprintf("XGBoost          -> RMSE: %.2f | MAE: %.2f | R²: %.3f\n", xgb_rmse, xgb_mae, xgb_r2))
 
# ---- 4.2 ±10-15% Accuracy Check ----
within_10 <- mean(abs((xgb_pred - test_data$price) / test_data$price) <= 0.10) * 100
within_15 <- mean(abs((xgb_pred - test_data$price) / test_data$price) <= 0.15) * 100
 
cat(sprintf("\nXGBoost: %.1f%% of predictions are within ±10%% of actual price", within_10))
cat(sprintf("\nXGBoost: %.1f%% of predictions are within ±15%% of actual price\n", within_15))
 
# ---- 4.3 SHAP Values for Interpretability ----
shap_values <- shap.values(xgb_model, X_train = as.matrix(train_data[, -which(names(train_data) == "price")]))
shap_long <- shap.prep(shap_contrib = shap_values$shap_score,
                       X_train = as.matrix(train_data[, -which(names(train_data) == "price")]))
 
# Top features SHAP summary plot
png("xgb_shap_summary.png", width = 1000, height = 600)
shap.plot.summary(shap_long)
dev.off()
 
cat("\nSHAP summary plot saved as xgb_shap_summary.png (top features influencing price).")