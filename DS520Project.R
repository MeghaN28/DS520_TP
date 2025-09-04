# Install packages
install.packages("caret")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("randomForest")
install.packages("xgboost")

library(caret)
library(ggplot2)
library(corrplot)
library(randomForest)
library(xgboost)
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

# 2.5 Feature importance using Random Forest
set.seed(123)
rf_model <- randomForest(price ~ ., data = listings_encoded, ntree = 100, importance = TRUE)

# Save RF feature importance plot
png("rf_feature_importance.png", width = 1000, height = 600)
varImpPlot(rf_model)
dev.off()

# -------------------------------
# 3) MODEL BUILDING & TRAINING
# -------------------------------

# 3.1 Train-test split already done
# train_data, test_data from previous steps

# 3.2 Linear Regression
lm_model <- lm(price ~ ., data = train_data)
summary(lm_model)

# Predictions
lm_pred <- predict(lm_model, newdata = test_data)

# RMSE
lm_rmse <- sqrt(mean((test_data$price - lm_pred)^2))
print(paste("Linear Regression RMSE:", round(lm_rmse, 2)))

# 3.3 Random Forest Regression
rf_model2 <- randomForest(price ~ ., data = train_data, ntree = 200, mtry = 5)
rf_pred <- predict(rf_model2, newdata = test_data)
rf_rmse <- sqrt(mean((test_data$price - rf_pred)^2))
print(paste("Random Forest RMSE:", round(rf_rmse, 2)))

# Hyperparameter tuning (Random Forest)
rf_grid <- expand.grid(mtry = c(3, 5, 7))
train_control <- trainControl(method = "cv", number = 5)
rf_tuned <- train(price ~ ., data = train_data, method = "rf",
                  tuneGrid = rf_grid, trControl = train_control, ntree = 200)
print(rf_tuned)

# 3.4 XGBoost Regression
# Prepare matrix for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "price")]),
                            label = train_data$price)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "price")]),
                           label = test_data$price)

# Train XGBoost model
xgb_model <- xgboost(data = train_matrix, max.depth = 6, eta = 0.1, nrounds = 100,
                     objective = "reg:squarederror", verbose = 0)

# Predictions
xgb_pred <- predict(xgb_model, test_matrix)
xgb_rmse <- sqrt(mean((test_data$price - xgb_pred)^2))
print(paste("XGBoost RMSE:", round(xgb_rmse, 2)))

# Feature importance for XGBoost
importance_matrix <- xgb.importance(feature_names = colnames(train_data[, -which(names(train_data) == "price")]),
                                    model = xgb_model)

# Save XGBoost importance plot
png("xgb_feature_importance.png", width = 1000, height = 600)
xgb.plot.importance(importance_matrix)
dev.off()
