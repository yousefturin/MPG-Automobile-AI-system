library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(Metrics)
library(ggplot2)
library(mlr)
library(tidymodels)
library(utils)
install.packages("tidyverse");
install.packages("caret");
install.packages("randomForest");
install.packages("rpart");
install.packages("Metrics");
install.packages("ggplot2");
install.packages("mlr");
install.packages("tidymodels");
install.packages("utils");
predict_mpg <- function(){
  # Read in the data set
  df <- read.csv("data/auto-mpg.csv")

  # Check if the non_numeric_variable column is numeric
  # If not, remove it from the data frame
  if (!is.numeric(df$non_numeric_variable)) {
    df <- df[, !names(df) %in% c("car.name")]
  }
  print(summary(df))
  # Print the first few rows of the data frame
  print(head(df))
  # Print the column names of the data frame
  print(names(df))
  ggplot(df, aes(x = mpg)) + geom_histogram()
  for (col in names(df)) {
    ggplot(df, aes(x = col, y = mpg)) + geom_point()
  }
  # Check if all columns are numeric
  # If not, remove them from the data frame
    non_numeric_cols <- sapply(df, function(x) !is.numeric(x))
  if (any(non_numeric_cols)) {
    df <- df[, !non_numeric_cols]
  }
  # Calculate the correlation between the variables
  cor(df, method = "kendall")
  # Set up cross-validation
  folds <- caret::createFolds(df$mpg, k = 5)
  train_control <- caret::trainControl(
    method = "cv",
    number = 5,
    savePredictions = TRUE,
    index = folds
  )
  # Create a random forest model object
  rfr <- randomForest(x = df[, !names(df) %in% c("mpg")], y = df$mpg,
    n_estimators = 100,
    criterion = "mae",
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0,
    max_leaf_nodes = None,
    min_impurity_decrease = 0,
    bootstrap = TRUE,
    oob_score = FALSE,
    n_jobs = None,
    random_state = None,
    verbose = 0,
    warm_start = FALSE,
    ccp_alpha = 0,
    max_samples = None
  )
# Fit the model to the data using cross-validation
  model <- caret::train(
    x = df[, !names(df) %in% c("mpg")],
    y = df$mpg,
    method = "rf",
    metric = "mae",
    tuneGrid = data.frame(mtry = c(1, 2, 3)),
    trControl = train_control,
    preProcess = c("center", "scale"),
    tuneLength = 3,
    model = rfr # pass the model object directly
  )
  # Make predictions on the data
  y_pred <- predict(model, df)
  # Calculate evaluation metrics
  mae <- Metrics::mae(y_pred, df$mpg)
  rmse <- Metrics::rmse(y_pred, df$mpg)
  mape <- Metrics::mape(y_pred, df$mpg)
  # Print the evaluation metrics
  cat("Mean Absolute Error:", mae, "\n")
  cat("Root Mean Squared Error:", rmse, "\n")
  cat("Mean Absolute Percentage Error:", mape)
    # Create a named list containing the model and predictions
  result <- list(model = model, y_pred = y_pred)
  print(result)
  # Return the named list
  return(result)
}

save_model_predictions <- function(model, y_pred, filepath) {
    save(list = c("model", "y_pred"), file = filepath)
}

# Call the predict_mpg() function
result <- predict_mpg()

# Extract the model object and predictions
model <- result$model
y_pred <- result$y_pred

# Save the model and predictions to a file
save_model_predictions(model, y_pred, "model_predictions.Rdata")
