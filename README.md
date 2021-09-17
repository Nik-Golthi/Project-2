# *Housing Market Analysis: Seatle-Tacoma-Bellevue CBSA*

## **Overview**

For our machine learning project, we conducted model performance analysis for two sets of models built using Seattle-Tacoma-Bellevue CBSA housing data. Specifically, we compared a set of regression models and a set of LSTM models.

Our regression models used a dataset consisting of [home sales and characteristics data for roughly 21,000 properties in King County, WA](https://www.kaggle.com/esratmaria/house-price-dataset-with-other-information). We built multiple regression models that used the property characteristics to predict home sales price, and then compared the models’ performance to determine which method of regression produced the best results.

For the LSTM models, we used the following Seattle-Tacoma-Bellevue CBSA housing inventory data to predict monthly median listing price: 

1) [Monthly Median Home Listing Price](https://fred.stlouisfed.org/series/MEDLISPRI42660#0)
2) [Monthly Active Listings Count](https://fred.stlouisfed.org/series/ACTLISCOU42660)
3) [Monthly Median Days on Market](https://fred.stlouisfed.org/series/MEDDAYONMAR42660)

We used each of the above statistics as a predictive feature for the monthly median listing price, building three different LSTM models. Finally, we compared the performance of the three LSTM models to determine which inventory statistic best predicts the median home listing price.

## **Regression Analysis**

We used the King County home sales and characteristics to build different regression models. In the following sections we'll describe each model and summarize model results.

### **4) Polynomial Regression**

#### Data Preparation:
Similar to the Random Forest data preparation, we removed the extraneous "id" and "date" columns. Then, we split the data into features and target ("price") dataset. Initially, we used all 18 features to train our model. The model performed quite poorly in this initial run, so we decided to remove 6 features that showed weak correlation to price ("zipcode", "long", "condition", "yr_built", "sqft_lot15", "sqft_lot"), and re-run.

For the polynomial regression model, we scaled our features data using the PolynomialFeatures scaler from sklearn.

#### Model Set-up:
To perform the actual regression, we used a combination of the PolynomialFeatures scaler from sklearn.preprocessing and the LinearRegression model from sklearn.linear_model. As we never covered Polynomial Regression in class, we referenced an article from the [Towards Data Science](https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386) blog to assist us with building the model.

The key parameter of sklearn's PolynomialFeatures is degrees. We set degrees equal to 3.

#### Model Results:
We tested the Polynomial Regression on out-of-sample and in-sample data. In addition, we ran the model twice, once with all available features, and a second time with a more refined set of features. Below are the model's performance metrics from the first run:

##### Polynomial Regression Performance - All Features:
![PR Performance - All Features]()

The results of the initial run indicated that the model features were weak predictors on new data; the out-of-sample R-squared score was extremely low (31.71%), especially in comparison to the in-sample R-squared score (88.04%).

In an attempt to address this, we refined the predictive features by removing 6 features that showed weak correlation to price in our original dataset. Our second run produced an improved out-of-sample R-squared score, while posting similar RMSE, MAE, MAPE, and accuracy to the initial run:

##### Polynomial Regression Performance - Refined Features:
![PR Performance - Refined  Features]()

### **5) Random Forest Regression**

#### Data Preparation:
Before setting up the Random Forest Regressor, we perfomed some basic clean-up on the King County housing data. We removed extraneous "id" and "date" columns, and then split the data into features and target datasets. In this case, our target was the "price" column, while the features (X) dataset consisted of the 18 remaining characteristic columns in the dataset.

We did not scale our data prior to fitting the Random Forest Regression Model; we learned that scaling was [not necessary for the Random Forest Regressor](https://stackoverflow.com/questions/8961586/do-i-need-to-normalize-or-scale-data-for-randomforest-r-package) in our research. As a result, the last step of our data preparation was to split our data into training and testing sets.

#### Model Set-up:
To perform the actual regression, we used the RandomForestRegressor from sklearn.ensemble. As we never covered the RandomForestRegressor in class, we referenced an article from the [Towards Data Science](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0 ) blog to assist us with building the model.

The key parameter of sklearn's RandomForestRegressors is n_estimators. The parameter determines the number of decision trees used by the model. We decided to use 1,000 estimators

#### Model Results:
We tested the Random Forest Regressor model on out-of-sample and in-sample data. Below are the model's performance metrics:

##### RF Regressor Performance:
![RF Regressor Performance]()

Based on the performance metrics, the Random Forest Regressor model appears to be a relatively strong model for predicting the home sales price. The model posts a high R-squared score on the testing data (87.1%), suggesting that the predictive features used are significant. Additionally, the mean absolute error percentage is relatively low at 12.8%. The strength of the model is also apparent when charting the actual test sales prices versus the predicted test sales prices. The predicted sales prices track the actual sales prices relatively well:

#### Test Actual vs. Predicted, First 500 properties
![RF1](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_sales_price_hv1.png)

#### Test Actual vs. Predicted, Last 500 properties
![RF2](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_sales_price_hv2.png)

There are some potential concerns with this model however. The model appears to perform signficantly better on the training data. For example, the test RMSE ($125,000) is quite a bit higher than the training RMSE ($48,000) which raises the possibility of overfitting.

## Conclusions: Regression Analysis

## **LSTM Analysis**

We used the Seattle-Tacoma-Bellevue CBSA housing inventory data to build three LSTM models. In the following sections we'll  describe each model and summarize model results.

### **1) Active Listings Count LSTM**

#### Data Preparation:
Prior to building our model, we created a combined dataset that contained all available monthly active listings counts and the corresponding monthly median listing prices. Then, we broke the combined dataset into a features (active listings counts) and target (median listing prices) dataset, and further split these datasets into training and testing datasets. Lastly, we used the MinMaxScaler from sklearn to scale our training and testing datasets. 

#### Model / Run Set-up:
We used the Sequential model from Keras and LSTM, Dropout, and Dense from Keras.layers to build a model with 3 layers. Below are the various parameters used to build the model:
- Lookback Window: 7 months
- Dropout_fraction: 0.2
- 3 layers

Next, we compiled and fit the model with the following parameters:
- Optimizer = “adam”
- Loss = “mean_squared_error”
- Epochs = 1,000
- Shuffle = False
- Batch_Size = 50

#### Model Results:
We tested the Random Forest Regressor model on out-of-sample and in-sample data. Below are the model's performance metrics:

##### RF Regressor Performance:
![RF Regressor Performance]()

Based on the performance metrics, the Random Forest Regressor model appears to be a relatively strong model for predicting the home sales price. The model posts a high R-squared score on the testing data (87.1%), suggesting that the predictive features used are significant. Additionally, the mean absolute error percentage is relatively low at 12.8%. The strength of the model is also apparent when charting the actual test sales prices versus the predicted test sales prices. The predicted sales prices track the actual sales prices relatively well:

#### Test Actual vs. Predicted, First 500 properties
![RF1](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_sales_price_hv1.png)

#### Test Actual vs. Predicted, Last 500 properties
![RF2](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_sales_price_hv2.png)

There are some potential concerns with this model however. The model appears to perform significantly better on the training data. For example, the test RMSE ($125,000) is quite a bit higher than the training RMSE ($48,000) which raises the possibility of overfitting.
