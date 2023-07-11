![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/65453f80-d038-469b-8f21-d8db48d71728)# MPG-Automobile-AI-system
This research paper is about the final review of the MPG - Automobile AI system and what processes must be followed to create the algorithm that will predict the Average Mile Per Gallon using two different approaches first one is linear and the second one is random forest. The steps of making such an AI system will be followed get the Dataset, Download the independence, read the data and analysis it, repair the missing data if exist, list the columns of the data, Data analysis, understand the correlation between the columns, create the model, train the data, create a pipeline, fit the data throw pipeline, test the data and store the data in a file where this process will be implemented in python and R using artificial intelligence.

# Python Programming Language
## Data
The data determine how efficient the AI system is so with that said, the data must be treated with precautions, so to understand the data it need to be read it using the Pandas library,

`df = pd.read_csv('data/auto-mpg.csv')`

Then get a proper description of the data as the count and the mean value as well as the standard deviation,

`print(df.describe())`

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/b7390588-2aed-4cd2-ab08-0af65b315a44)

Dropping unwanted columns that do not make any changes to the algorithm, where the car name is to complex to be filtered in the algorithm,

`list(df.columns)`

['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

`df.drop(labels = ['car name'],axis = 1, inplace = True )`

`list(df.columns)`

['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']

## Data Analysis

The data must be understandable to make the connection between the columns and the wanted item ’mpg’ and do understand it better to visualize it using the seaborn library and make the Y axis as ‘mpg’ where the X axis as the rest of the columns “create a correlation”, and understand where exactly the mpg is focus on,

`sns.displot(df['mpg'])`

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/03ead83f-8a85-4a56-9bc2-2e562968f96e)

Display all the columns in a relationship with ‘mpg’,

`for col in df.columns:
    sns.scatterplot(x = col, y = 'mpg', data = df)
    plt.show()`
    
![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/4373c8ea-20a6-4dde-923d-032ac79307a6)
 

As it is seen from the graphs, none of these functions are really linear functions and the acceleration and mpg relationship is complex whereas, for cylinders, displacement and horsepower are almost the same functions where in this case they needed to be combined into one single function and drop them individually.


Then display all the correlations in the table,

`df.corr('kendall')`

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/7c5c1cc6-654b-4442-8aa0-569fab344dcb)


## Engineering Feature

Dropping and combining the columns to make better raw data,
Combine the horsepower and weight by dividing them, 

`df['displacement_per_cylinder'] = df['displacement']/df['cylinders']`

combine the displacement, horsepower, and weight by multiplying them, 

`df['displacement_power_weight'] = df['horsepower']*df['weight']*df['displacement']`

![image](https://github.com/yousefturin/MPG-Automobile-AI-system/assets/94796673/540fec57-5af4-4f2c-83d3-924b59b3f72c)


## Training Data
Split the data into Train and test arrays and drop unwanted columns,

`y_4= df['mpg']`

`    df_4  = df.drop('mpg', 'weight','horsepower','cylinders','displacement', axis=1)
    X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split( df_4, y_4, test_size=0.33, random_state=42)`

Create a pipeline to Sequentially apply a list of transforms and a final estimator, then the final estimator needs to implement to a fit,
`pipe_3 = Pipeline(steps=[('scaler', StandardScaler(),),
                        ('Random_forest', RandomForestRegressor(n_estimators=100, 
                        criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                        min_weight_fraction_leaf=0.0,
                        max_leaf_nodes=None, min_impurity_decrease=0.0,
                        bootstrap=True, oob_score=False, n_jobs= None, random_state= None, verbose=0,
                        warm_start=False, ccp_alpha=0.0,max_samples=None),)]) `

fit the result of the train,

`pipe_3.fit(X_train_3, y_train_3)`

## Testing Data

Run a test to see what are the predictions that are model is making

`prediction_3 = pipe_3.predict(X_test_3)`

Assigning variables to some functions,

`mae_3 = mean_absolute_error(y_test_3, prediction_3)`

`r2_3 = r2_score(y_test_3, prediction_3)`

`mse_3 = mean_squared_error(y_test_3, prediction_3)`

`mape_3 = mean_absolute_percentage_error(y_test_3, prediction_3)`

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

`filename = "MPG_automobile_0.88_model_randomForest.joblib"`

The first parameter is for the model’s name and the second parameter is the file name that it will be stored,

`joblib.dump(pipe_4, filename)`

## Load Model
Loading the model to test the data is also done by the same library 

`loaded_model = joblib.load("MPG_automobile_0.88_model_randomForest.joblib")`

`result = loaded_model.score(X_test_4, y_test_4)`
