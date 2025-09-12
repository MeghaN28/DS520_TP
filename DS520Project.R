# === Required packages ===
install.packages("caret")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("randomForest")
install.packages("xgboost")
install.packages("doParallel")     # parallel processing
install.packages("Metrics")       # mae
install.packages("SHAPforxgboost")

library(caret)
library(ggplot2)
library(corrplot)
library(randomForest)
library(xgboost)
library(doParallel)
library(Metrics)
library(SHAPforxgboost)
library(parallel)  # for detectCores()

# === 1) DATA COLLECTION & INITIAL CLEANING ===
listings <- read.csv("listings.csv", stringsAsFactors = FALSE)

# Keep only useful features for price prediction (drop raw amenities text to avoid huge dummies)
listings <- listings[, c("id", "price", "neighbourhood", "property_type",
                         "room_type", "bedrooms", "bathrooms",
                         "host_since", "host_is_superhost", "number_of_reviews",
                         "review_scores_rating", "availability_30", "amenities")]

# Basic cleaning
listings <- listings[!duplicated(listings$id), ]
listings <- listings[!is.na(listings$price), ]
listings$price <- as.numeric(gsub("[\\$,]", "", listings$price))
listings <- listings[listings$price > 10 & listings$price < 1000, ]

# Handle missing numeric values
listings$number_of_reviews[is.na(listings$number_of_reviews)] <-
  median(listings$number_of_reviews, na.rm = TRUE)
listings$review_scores_rating[is.na(listings$review_scores_rating)] <-
  median(listings$review_scores_rating, na.rm = TRUE)

# Handle missing categorical
listings$host_is_superhost[is.na(listings$host_is_superhost)] <- "Unknown"

# === 2) FEATURE ENGINEERING ===
# amenities_count (safe: count commas + 1 when non-empty)
listings$amenities_count <- sapply(strsplit(as.character(listings$amenities), ","), function(x) {
  if (length(x)==1 && (is.na(x) || x == "")) return(0) else return(length(x))
})

# host_experience_years
listings$host_since <- as.Date(listings$host_since, format = "%m/%d/%Y")
listings$host_experience_years <- as.numeric(difftime(Sys.Date(),
                                                     listings$host_since,
                                                     units = "days")) / 365
listings$host_experience_years[is.na(listings$host_experience_years)] <- 0

# price_per_bedroom
listings$bedrooms[is.na(listings$bedrooms)] <- 0
listings$price_per_bedroom <- listings$price / pmax(listings$bedrooms, 1)

# Drop raw amenities text and id (not useful for modeling)
listings$amenities <- NULL
listings$id <- NULL

# Convert some columns to factor
listings$property_type <- as.factor(listings$property_type)
listings$neighbourhood <- as.factor(listings$neighbourhood)
listings$room_type <- as.factor(listings$room_type)
listings$host_is_superhost <- as.factor(listings$host_is_superhost)

# Remove any rows with remaining NA (safer to do this after feature engineering)
listings <- na.omit(listings)

# === 3) TRAIN-TEST SPLIT (on the raw listings dataframe) ===
set.seed(123)
train_idx <- createDataPartition(listings$price, p = 0.8, list = FALSE)
train_raw <- listings[train_idx, ]
test_raw  <- listings[-train_idx, ]

# === 4) ENCODING: create dummyVars based on TRAIN only, then apply to TEST ===
# Use formula excluding price so dummyVars doesn't try to encode the target
dummy_formula <- as.formula("~ . - price")
dummy_vars <- dummyVars(dummy_formula, data = train_raw, fullRank = TRUE)

train_encoded <- data.frame(predict(dummy_vars, newdata = train_raw))
test_encoded  <- data.frame(predict(dummy_vars, newdata = test_raw))

# Add back the price column (unchanged raw numeric target)
train_encoded$price <- train_raw$price
test_encoded$price  <- test_raw$price

# Ensure column order identical
train_encoded <- train_encoded[, c(setdiff(names(train_encoded), "price"), "price")]
test_encoded  <- test_encoded[, c(setdiff(names(test_encoded), "price"), "price")]

# === 5) SCALING: scale numeric predictors ONLY (exclude price) ===
predictor_names <- setdiff(names(train_encoded), "price")
# Find numeric predictors in train
numeric_preds <- predictor_names[sapply(train_encoded[predictor_names], is.numeric)]

# Compute scaling from train and apply same scale to test
preproc_vals <- preProcess(train_encoded[, numeric_preds], method = c("center", "scale"))
train_encoded[, numeric_preds] <- predict(preproc_vals, train_encoded[, numeric_preds])
test_encoded[, numeric_preds]  <- predict(preproc_vals, test_encoded[, numeric_preds])

# Final datasets
train_data <- train_encoded
test_data  <- test_encoded

# Quick sanity checks
cat("Train dim:", dim(train_data), "\n")
cat("Test dim :", dim(test_data), "\n")
cat("Any NA in train? ", any(is.na(train_data)), "\n")
cat("Any NA in test?  ", any(is.na(test_data)), "\n")

# === 6) EDA (plots) ===
png("price_distribution.png", width = 800, height = 600)
ggplot(train_raw, aes(x = price)) +
  geom_histogram(bins = 50) + theme_minimal() + ggtitle("Price Distribution (train)")
dev.off()

png("price_vs_room_type.png", width = 800, height = 600)
ggplot(train_raw, aes(x = room_type, y = price)) + geom_boxplot() + theme_minimal() + ggtitle("Price vs Room Type")
dev.off()

png("price_vs_neighbourhood.png", width = 1000, height = 600)
ggplot(train_raw, aes(x = neighbourhood, y = price)) + geom_boxplot() + theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Price vs Neighbourhood")
dev.off()

# Correlation heatmap on numeric columns of the raw data (before encoding)
num_features <- train_raw[, sapply(train_raw, is.numeric)]
cor_matrix <- cor(num_features, use = "complete.obs")
png("correlation_heatmap.png", width = 1000, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, tl.col = "black", addCoef.col = "black")
dev.off()

# === 7) MODELING & EVALUATION ===

# ---- 7.1 Linear Regression ----
lm_model <- lm(price ~ ., data = train_data)
summary(lm_model)

lm_pred <- predict(lm_model, newdata = test_data)
# If some predictions are NA (shouldn't be now), handle:
if (any(is.na(lm_pred))) {
  warning("Some lm predictions are NA. Check test_data columns and NA values.")
}

lm_rmse <- sqrt(mean((test_data$price - lm_pred)^2))
lm_mae  <- mae(test_data$price, lm_pred)
lm_r2   <- R2(lm_pred, test_data$price)

# ---- 7.2 Random Forest (single-step training on train_data) ----
set.seed(123)
# Use parallel if available
num_cores <- max(1, detectCores() - 1)
cl <- makeCluster(num_cores)
registerDoParallel(cl)

rf_model <- randomForest(price ~ ., data = train_data, ntree = 100, importance = TRUE)
stopCluster(cl)
registerDoSEQ()

rf_pred <- predict(rf_model, newdata = test_data)
rf_rmse <- sqrt(mean((test_data$price - rf_pred)^2))
rf_mae  <- mae(test_data$price, rf_pred)
rf_r2   <- R2(rf_pred, test_data$price)

# Feature importance (save plot)
png("rf_feature_importance.png", width = 1000, height = 600)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()

rf_importance <- importance(rf_model)
rf_importance_sorted <- rf_importance[order(rf_importance[,1], decreasing = TRUE), ]
print(head(rf_importance_sorted, 20))

# Optional: caret hyperparameter tuning (quick)
rf_grid <- expand.grid(mtry = c(3, 5))
train_control <- trainControl(method = "cv", number = 3)
rf_tuned <- train(price ~ ., data = train_data, method = "rf",
                  tuneGrid = rf_grid, trControl = train_control, ntree = 50)
print(rf_tuned)

# ---- 7.3 XGBoost ----
# Prepare matrices: ensure same predictors order
preds <- setdiff(names(train_data), "price")
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, preds]), label = train_data$price)
test_matrix  <- xgb.DMatrix(data = as.matrix(test_data[, preds]), label = test_data$price)

set.seed(123)
xgb_model <- xgboost(data = train_matrix, max.depth = 4, eta = 0.2, nrounds = 30,
                     objective = "reg:squarederror", verbose = 0, nthread = 4)

xgb_pred <- predict(xgb_model, test_matrix)
xgb_rmse <- sqrt(mean((test_data$price - xgb_pred)^2))
xgb_mae  <- mae(test_data$price, xgb_pred)
xgb_r2   <- R2(xgb_pred, test_data$price)

# XGBoost importance & plot
importance_matrix <- xgb.importance(feature_names = preds, model = xgb_model)
png("xgb_feature_importance.png", width = 1000, height = 600)
xgb.plot.importance(importance_matrix)
dev.off()

# ---- 7.4 Evaluation Summary ----
cat("\n--- Model Performance ---\n")
cat(sprintf("Linear Regression -> RMSE: %.2f | MAE: %.2f | R²: %.3f\n", lm_rmse, lm_mae, lm_r2))
cat(sprintf("Random Forest     -> RMSE: %.2f | MAE: %.2f | R²: %.3f\n", rf_rmse, rf_mae, rf_r2))
cat(sprintf("XGBoost           -> RMSE: %.2f | MAE: %.2f | R²: %.3f\n", xgb_rmse, xgb_mae, xgb_r2))

# ---- 7.5 ±10-15% check for the best model (here XGBoost) ----
within_10 <- mean(abs((xgb_pred - test_data$price) / test_data$price) <= 0.10) * 100
within_15 <- mean(abs((xgb_pred - test_data$price) / test_data$price) <= 0.15) * 100
cat(sprintf("\nXGBoost: %.1f%% of predictions are within ±10%% of actual price\n", within_10))
cat(sprintf("XGBoost: %.1f%% of predictions are within ±15%% of actual price\n", within_15))

# ---- 7.6 SHAP for XGBoost (interpretability) ----
shap_values <- shap.values(xgb_model, X_train = as.matrix(train_data[, preds]))
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = as.matrix(train_data[, preds]))

png("xgb_shap_summary.png", width = 1000, height = 600)
shap.plot.summary(shap_long)
dev.off()
cat("\nSHAP summary plot saved as xgb_shap_summary.png (top features influencing price).\n")
