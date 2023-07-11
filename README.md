# MPG-Automobile-AI-system
This research paper is about the final review of the MPG - Automobile AI system and what processes must be followed to create the algorithm that will predict the Average Mile Per Gallon using two different approaches first one is linear and the second one is random forest. The steps of making such an AI system will be followed get the Dataset, Download the independence, read the data and analysis it, repair the missing data if exist, list the columns of the data, Data analysis, understand the correlation between the columns, create the model, train the data, create a pipeline, fit the data throw pipeline, test the data and store the data in a file where this process will be implemented in python and R using artificial intelligence.

# Python Programming Language
## Data
The data determine how efficient the AI system is so with that said, the data must be treated with precautions, so to understand the data it needs to be read it using the Pandas library,

    df = pd.read_csv('data/auto-mpg.csv')    

Then get a proper description of the data as the count and the mean value as well as the standard deviation,

    print(df.describe())

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/b7390588-2aed-4cd2-ab08-0af65b315a44)

Dropping unwanted columns that do not make any changes to the algorithm, where the car name is to complex to be filtered in the algorithm,

    list(df.columns)

['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

    df.drop(labels = ['car name'],axis = 1, inplace = True )

    list(df.columns)

['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']

## Data Analysis

The data must be understandable to make the connection between the columns and the wanted item ’mpg’ and do understand it better to visualize it using the seaborn library and make the Y axis as ‘mpg’ where the X axis as the rest of the columns “create a correlation”, and understand where exactly the mpg is focus on,

    sns.displot(df['mpg'])

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/03ead83f-8a85-4a56-9bc2-2e562968f96e)

Display all the columns in a relationship with ‘mpg’,

    for col in df.columns:
    sns.scatterplot(x = col, y = 'mpg', data = df)
    plt.show()
    
![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/4373c8ea-20a6-4dde-923d-032ac79307a6)
 

As it is seen from the graphs, none of these functions are really linear functions and the acceleration and mpg relationship is complex whereas, for cylinders, displacement and horsepower are almost the same functions where in this case they needed to be combined into one single function and drop them individually.


Then display all the correlations in the table,

    df.corr('kendall')

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/7c5c1cc6-654b-4442-8aa0-569fab344dcb)


## Engineering Feature

Dropping and combining the columns to make better raw data,
Combine the horsepower and weight by dividing them, 

    df['displacement_per_cylinder'] = df['displacement']/df['cylinders']

combine the displacement, horsepower, and weight by multiplying them, 

    df['displacement_power_weight'] = df['horsepower']*df['weight']*df['displacement']

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/540fec57-5af4-4f2c-83d3-924b59b3f72c)


## Training Data
Split the data into Train and test arrays and drop unwanted columns,

    y_4= df['mpg']


    df_4  = df.drop('mpg', 'weight','horsepower','cylinders','displacement', axis=1)
    X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split( df_4, y_4, test_size=0.33, random_state=42)

Create a pipeline to Sequentially apply a list of transforms and a final estimator, then the final estimator needs to implement to a fit,


    pipe_3 = Pipeline(steps=[('scaler', StandardScaler(),),
                        ('Random_forest', RandomForestRegressor(n_estimators=100, 
                        criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                        min_weight_fraction_leaf=0.0,
                        max_leaf_nodes=None, min_impurity_decrease=0.0,
                        bootstrap=True, oob_score=False, n_jobs= None, random_state= None, verbose=0,
                        warm_start=False, ccp_alpha=0.0,max_samples=None),)]) 

fit the result of the train,

    pipe_3.fit(X_train_3, y_train_3)

## Testing Data

Run a test to see what are the predictions that are model is making

    prediction_3 = pipe_3.predict(X_test_3)

Assigning variables to some functions,

    mae_3 = mean_absolute_error(y_test_3, prediction_3)

    r2_3 = r2_score(y_test_3, prediction_3)

    mse_3 = mean_squared_error(y_test_3, prediction_3)

    mape_3 = mean_absolute_percentage_error(y_test_3, prediction_3)
____________________________

    Representation of the percentage of the variance,
    r2_3: 0.8852189559676944

    Find the mean absolute error of this model,
    mae_3: 1.8619393939393933

    Find the average error of this model,
    mse_3: 6.606451378787877

    Find out off set percentage of this model from the origin point,
    mape_3: 0.08018727779203869

## Save Model
Saving the model can be done using the joblib library,

    filename = "MPG_automobile_0.88_model_randomForest.joblib"

The first parameter is for the model’s name and the second parameter is the file name that it will be stored,

    joblib.dump(pipe_4, filename)

## Load Model
Loading the model to test the data is also done by the same library 

    loaded_model = joblib.load("MPG_automobile_0.88_model_randomForest.joblib")

    result = loaded_model.score(X_test_4, y_test_4)

# R Programming language 
## Packages 

library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(Metrics)
library(ggplot2)
library(mlr)
library(tidymodels)
library(utils)

## Data
Reading the data is done by using read.csv function:

    df <- read.csv("data/auto-mpg.csv")

Displaying the data mean values and a small brief:

    print(summary(df))

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/22413b16-48d3-424e-b145-da35fda1f759)


Filtering the data to remove non numerical variables as in this case car.name:

    if (!is.numeric(df$non_numeric_variable)) {
    df <- df[, !names(df) %in% c("car.name")]}

Display the data again:


    print(head(df))

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/08f7776b-8e09-43a8-af6b-dae6d34114cb)


## Data Analysis
 The data must be understandable to make the connection between the columns and the wanted item ’mpg’ and to understand it better and visualize it using the ggplot library and make the Y axis as ‘mpg’ where the X axis as the rest of the columns “create a correlation”, and understand where exactly the mpg is focus on, 

    ggplot(df, aes(x = mpg)) + geom_histogram()
    for (col in names(df)) {
    ggplot(df, aes(x = col, y = mpg)) + geom_point()}

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/7c810a7a-4279-464c-a706-d4c6ad72e388)

Calculate the correlation between the variables using the cor method:

    cor(df, method = "kendall")
    
![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/1d15678f-38fa-47e2-b399-80ed721be4da)


## Processing Data 
Create a train control object for cross-validation:

    folds <- caret::createFolds(df$mpg, k = 5)
    train_control <- caret::trainControl(
    method = "cv",
    number = 5,
    savePredictions = TRUE, 
    index = folds
    )

## Training Data 
Create a Random Forest model for the object:

    rfr <- randomForest(x = df[, !names(df) %in% c("mpg")], y = df$mpg,
    n_estimators = 100,criterion = "mae", max_depth = None, min_samples_split = 2,
    min_samples_leaf = 1, min_weight_fraction_leaf = 0, max_leaf_nodes = None,
    min_impurity_decrease = 0, bootstrap = TRUE,   oob_score = FALSE,
    n_jobs = None,    random_state = None,    verbose = 0,    warm_start = FALSE,
    ccp_alpha = 0,    max_samples = None
    )

## Fitting Data
Fit the model to the data using cross-validation

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

## Test Data
Make predictions on the data:

    y_pred <- predict(model, df)
Calculate evaluation metrics:

    > mae <- Metrics::mae(y_pred, df$mpg)
    > cat("Mean Absolute Error:", mae, "\n")

Mean Absolute Error: 0.9672114

    > rmse <- Metrics::rmse(y_pred, df$mpg)
    > cat("Root Mean Squared Error:", rmse, "\n")

Root Mean Squared Error: 1.357074

    > mape <- Metrics::mape(y_pred, df$mpg)

An Absolute Percentage Error: 0.04144842

## Store Model
Create a named list containing the model and predictions to return it:

    result <- list(model = model, y_pred = y_pred)

save function that the model can be saved from:

    save_model_predictions <- function(model, y_pred, filepath) {
        save(list = c("model", "y_pred"), file = filepath)
    }

    save_model_predictions(model, y_pred, "model_predictions.Rdata")
