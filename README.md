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

We used the King County home sales and characteristics to build different regression models. In the following sections, we'll describe each model and summarize model results.

### **1)Linear Regression**
Before performing the linear regression models, we first produced some graphical representations of the data.  We wanted to see if any of the features in the dataset correlated more closely than the others to the housing price.  The seaborn heatmap shows those correlations.  From that heatmap, we then plotted the following graphs; Square Footage vs. Price and Total Square Footage vs. Price.  This particular dataset separates “living square footage” from “basement square footage.”  In some housing markets, the basement is not considered in the square footage of the home, and in this market, we were unable to find out if that was the case.  However, looking at the plots, we made our “x” variable the living square footage plus the basement square footage.  

![Correlation Plot](https://github.com/Nik-Golthi/Project-2/blob/main/Images/correlation_plot.png)
![Square Feet vs. Price](https://github.com/Nik-Golthi/Project-2/blob/main/Images/Price%20vs%20Square%20Feet.png)
![Total Square Footage vs. Price](https://github.com/Nik-Golthi/Project-2/blob/main/Images/total%20square%20footage.png)

The linear regression model was not very efficient at predicting the housing prices in King County. The R-squared value for the model was 0.45, indicating that approximately 45% of the predicted values were accurate.

![Predicted Values](https://github.com/Nik-Golthi/Project-2/blob/main/Images/predicted%20values.png)

### **2)Multiple Linear Regression**
In order to try and improve the R-squared score we thought we could implement Multiple Linear Regression.  Multiple Linear Regression is closely related to linear regression, with the exception that several variables (features) are being looked at to predict the outcome (y).  In this case, rather than just using the square footage, we will use all the available features in the dataset.  In so doing, each feature will have a coefficient, indicating that there are several lines that can be drawn to predict the housing price.  The coefficients that were generated for each of the features used are in the attached csv file “model_coefficients.csv”, and we see that the variance score, which can be looked at as the same as R-squared, has increased to 0.70. Just by changing the model we were able to see a significant increase in accuracy.

### **3)Logistic Regression**
Given the results from the linear regression and multiple linear regression, we wanted to check to see what kind of results logistic regression would give us.  Although this is a non-linear model, it is best used to predict binary, yes or no, data.  Housing prices are obviously not binary.  This was clearly the case after running the model.  The scores for the training data and testing data were 0.01 and 0.008, respectively, indicating a model that is not very strong.  We examined the first 19 predictions vs. the actual as generated by the model as seen on slide seven of the presentation.

### **4) Polynomial Regression**

#### Data Preparation:
For the polynomial regression, we started out by removing the extraneous "id" and "date" columns from our dataset. Then, we split the data into features and target ("price") datasets. Initially, we used all 18 features to train our model. The model performed quite poorly in this initial run, so we decided to remove 6 features that showed weak correlation to price ("zipcode", "long", "condition", "yr_built", "sqft_lot15", "sqft_lot"), and re-run.

For the polynomial regression model, we scaled our features data using the PolynomialFeatures scaler from sklearn.

#### Model Set-up:
To perform the actual regression, we used a combination of the PolynomialFeatures scaler from sklearn.preprocessing and the LinearRegression model from sklearn.linear_model. As we never covered polynomial regression in class, we referenced an article from the [Towards Data Science](https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386) blog to assist us with building the model.

The key parameter of sklearn's PolynomialFeatures is degrees. We set degrees equal to 3.

#### Model Results:
We tested the Polynomial Regression on out-of-sample and in-sample data. In addition, we ran the model twice, once with all available features, and a second time with a more refined set of features. Below are the model's performance metrics from the first run:

##### Polynomial Regression Performance - All Features:
![PR Performance - All Features](https://github.com/Nik-Golthi/Project-2/blob/main/Images/poly_reg_metrics1.png)

The results of the initial run indicated that the model features were weak predictors on new data; the out-of-sample R-squared score was extremely low (31.71%), especially in comparison to the in-sample R-squared score (88.04%).

In an attempt to address this, we refined the predictive features by removing the 6 features that showed the weakest correlation to price in our original dataset. Our second run produced an improved out-of-sample R-squared score, while posting similar RMSE, MAE, MAPE, and accuracy to the initial run:

##### Polynomial Regression Performance - Refined Features:
![PR Performance - Refined  Features](https://github.com/Nik-Golthi/Project-2/blob/main/Images/poly_reg_metrics2.png)

### **5) Random Forest Regression**

#### Data Preparation:
Before setting up the Random Forest Regressor, we performed some basic clean-up on the King County housing data. We removed extraneous "id" and "date" columns, and then split the data into features and target datasets. In this case, our target was the "price" column, while the features (X) dataset consisted of the 18 remaining characteristic columns in the dataset.

We did not scale our data prior to fitting the Random Forest Regression Model; we learned that scaling was [not necessary for the Random Forest Regressor](https://stackoverflow.com/questions/8961586/do-i-need-to-normalize-or-scale-data-for-randomforest-r-package) in our research. As a result, the last step of our data preparation was to split our data into training and testing sets.

#### Model Set-up:
To perform the actual regression, we used the RandomForestRegressor from sklearn.ensemble. As we never covered the RandomForestRegressor in class, we referenced an article from the [Towards Data Science](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0 ) blog to assist us with building the model.

The key parameter of sklearn's RandomForestRegressors is n_estimators. The parameter determines the number of decision trees used by the model. We decided to use 1,000 estimators.

#### Model Results:
We tested the Random Forest Regressor model on out-of-sample and in-sample data. Below are the model's performance metrics:

##### RF Regressor Performance:
![RF Regressor Performance](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_metrics.png)

Based on the performance metrics, the Random Forest Regressor model appears to be a relatively strong model for predicting the home sales price. The model posts a high R-squared score on the testing data (87.1%), suggesting that the predictive features used are significant. Additionally, the mean absolute error percentage is relatively low at 12.8%. The strength of the model is also apparent when charting the actual test sales prices versus the predicted test sales prices. The predicted sales prices track the actual sales prices pretty well:

#### Test Actual vs. Predicted, First 500 properties
![RF1](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_sales_price_hv1.png)

#### Test Actual vs. Predicted, Last 500 properties
![RF2](https://github.com/Nik-Golthi/Project-2/blob/main/Images/rf_reg_sales_price_hv2.png)

There are some potential concerns with this model however. The model appears to perform significantly better on the training data. For example, the test RMSE ($125,000) is quite a bit higher than the training RMSE ($48,000) which raises the possibility of overfitting.

## Conclusions: Regression Analysis

Based on the model results, the Random Forest Regression model appears to be strongest predictor of the home sales price. The RF regression produces the highest R-squared score and accuracy, while also posting the lowest RMSE and mean absolute percentage error of all the regression models. Aside from the RF regression model, the polynomial regression and multiple linear regression models also proved to be effective regression methods.

## **LSTM Analysis**

We used the Seattle-Tacoma-Bellevue CBSA housing inventory data to build three LSTM models. In the following sections we'll  describe each model and summarize model performance.

### **1) Active Listings Count LSTM**

#### Data Preparation:
Prior to building our model, we created a combined dataset that contained all available monthly active listings counts and the corresponding monthly median listing prices. Then, we broke the combined dataset into a features (active listings counts) and target (median listing prices) dataset, and further split these datasets into training and testing datasets. Lastly, we used the MinMaxScaler from sklearn to scale our training and testing sets. 

#### Model / Run Set-up:
We used the Sequential model from Keras and LSTM, Dropout, and Dense from Keras.layers to build a model with 3 layers. Below are the various parameters used to build the model:

- Lookback Window: 7 months
- Dropout_fraction: 0.2
- 3 layers

Next, we compiled and fit the model with the following parameters:
- Optimizer = “adam”
- Loss = “mean_squared_error”
- Epochs = 100
- Shuffle = False
- Batch_Size = 50

#### Model Results:
The Active Listings Count LSTM model appears to be a moderate predictor for median home listing price. The model produced a mean squared error of 0.22.

##### Active Listings Count LSTM MSE - Model Output:
![Active Listings LSTM MSE - Model](https://github.com/Nik-Golthi/Project-2/blob/main/Images/lstm_active_loss.png)

The above MSE seems to be low and acceptable at face value. However, as this MSE is based on data that was scaled by the MinMaxScaler, the value is difficult to interpret on its own. To improve interpretability, we re-scaled our testing data after running it through our model with the inverse_transform function. We then produced performance metrics on the re-scaled data. Below are the additional performance metrics:

##### Active Listings Count LSTM Performance:
![Active Listings LSTM Metrics](https://github.com/Nik-Golthi/Project-2/blob/main/Images/LSTM_active_listings_metrics.png)

The model performs passably on the testing data. The MAPE of roughly 16% seems reasonable, but the RMSE of roughly $110,000 is somewhat high. To further evaluate the model, we plotted the actual median listing prices from the test dataset to the predicted test prices.

#### Test Actual vs. Predicted, Active Listings LSTM
![Test Actual vs. Predicted - Active Listings LSTM](https://github.com/Nik-Golthi/Project-2/blob/main/Images/LSTM_active_listing_hvplot.png)

The actual versus predicted values plot suggests that the monthly active listings counts are only a moderate predictor of the monthly median price. The model appears to predict reasonably at first, but the performance clearly worsens in the later test periods.

### **2) Time on Market LSTM**

#### Data Preparation:
For our second LSTM model, we built a model that uses time on market data to predict the monthly median home listing price. Our data preparation process was identical to the previous active listings count model:

1) Created a combined dataset with all available monthly median days on market data and monthly median listing prices
2) Split combined dataset into features (monthly median days on market) and target (monthly median listing prices) datasets
3) Split features and target datasets into training and testing datasets
4) Scale training and testing datasets with MinMaxScaler from sklearn.preprocessing

#### Model / Run Set-up:
Once again, we used the Sequential model from Keras and LSTM, Dropout, and Dense from Keras.layers to build a model with 3 layers. Our model parameters for time on market LSTM model were almost identical to the previous active listings model parameters. We used the same lookback window of 7 months and dropout fraction of 20%. The only parameter change was to adjust the batch size fit parameter to 30. We experimented with multiple batch sizes and found that a batch size of 30 produced the best results for this model. Below are the full parameters used to build the time on market model:

- Lookback Window: 7 months
- Dropout_fraction: 0.2
- 3 layers

Compile and Fit Parameters:
- Optimizer = “adam”
- Loss = “mean_squared_error”
- Epochs = 100
- Shuffle = False
- Batch_Size = 50

#### Model Results:
The Time on Market LSTM model performed better than the previous active listings count model. For one, the time on market model produced a significantly lower mean squared error of 0.11.

##### Time on Market LSTM MSE - Model Output:
![Time on Market LSTM MSE - Model](https://github.com/Nik-Golthi/Project-2/blob/main/Images/lstm_time_mkt_loss.png)

To produce additional performance metrics, we ran our testing data through the model and re-scaled the results with the inverse_transform function. The additional metrics further confirmed that the time on market LSTM model was a better model for predicting median home listing price. The time on market LSTM model posted a lower RMSE ($79,000), lower mean absolute error ($70,000), and lower MAPE (10.65%):

##### Time on Market LSTM Performance:
![Time on Market LSTM Metrics](https://github.com/Nik-Golthi/Project-2/blob/main/Images/LSTM_time_on_mkt_metrics.png)

Below is a visual representation of the model's performance on the testing data:

#### Test Actual vs. Predicted, Time on Market LSTM
![Test Actual vs. Predicted - Time on Market LSTM](https://github.com/Nik-Golthi/Project-2/blob/main/Images/LSTM_active_listing_hvplot.png)

On the whole, the time on market LSTM model is clearly a better predictor of median listing price than the active listings count model. Given the strength of the performance metrics, time on market is a fairly strong predictor of the median home listing price.

### **3) Median Listing Price LSTM**

#### Data Preparation:
Our last LSTM model used the monthly median home listing prices themselves to predict median listing price. Our data preparation process was very similar to the preparation for the previous two models. The primary distinction was that our input data only consisted of one variable (median listing price) which served as both the predictive feature and target.

1) Created a dataset with all available monthly median listing prices
2) Split combined dataset into features (monthly median listing prices) and target (monthly median listing prices) datasets
3) Split features and target datasets into training and testing datasets
4) Scale training and testing datasets with MinMaxScaler from sklearn.preprocessing

#### Model / Run Set-up:
We built a 3-layer model using the Sequential model from Keras and LSTM, Dropout, and Dense from Keras.layers. Our model parameters for the median listing price LSTM were also very similar to the previous two LSTM models' parameters. One key difference is that we used a 9-month lookback window for the median listing price model; after testing multiple window sizes, we found that the 9-month lookback produced the best results for this model. Additionally, we decided to use a batch size fit parameter of 50 for this model, like we did with the active listings count model. All other parameters were identical to the previous two models' parameters. Below are the full parameters used to build the median listing price LSTM model:

- Lookback Window: 9 months
- Dropout_fraction: 0.2
- 3 layers

Compile and Fit Parameters:
- Optimizer = “adam”
- Loss = “mean_squared_error”
- Epochs = 100
- Shuffle = False
- Batch_Size = 50

#### Model Results:
As expected, the median listings price LSTM model proved to be an excellent predictor of monthly median listing prices. Of the three LSTM models built and tested, the median listings price model produced the lowest mean squared error of 0.05.

##### Median Listing Price LSTM MSE - Model Output:
![Median Listing Price LSTM MSE - Model](https://github.com/Nik-Golthi/Project-2/blob/main/Images/lstm_med_price_loss.png)

Like we did with the previous two models, we ran our testing data through the model and re-scaled the results with the inverse_transform function to better interpret the model's performance. Looking at the additional metrics, we see that the median listing price LSTM model has the lowest RMSE (~$47,000), MAE ($41,000), and MAPE (6.19%) of all three LSTM models we tested:

##### Median Listing Price LSTM Performance:
![Median Listing Price LSTM Metrics](https://github.com/Nik-Golthi/Project-2/blob/main/Images/LSTM_median_list_price_metrics.png)

The plot of the test actual and predicted median prices further highlights the strength of historical median listing prices as a predictor for median listing price.

#### Test Actual vs. Predicted, Median Listing Price LSTM
![Test Actual vs. Predicted - Median Listing Price LSTM](https://github.com/Nik-Golthi/Project-2/blob/main/Images/LSTM_med_list_price_hvplot.png)

In comparison to the previous two LSTM models, the median listing price model's predicted prices are much closer to the actual test values. In addition, the model predictions are following the general trend of the actual median prices over time, which was not necessarily true for the other two LSTM models.

In short, the median listing price LSTM is a great model for our purposes. As expected, historical monthly median prices are strong predictors of monthly median listing prices.

## **Conclusions: LSTM Model Analysis**

The objective of our LSTM Model analysis was to compare the predictive ability of various housing inventory statistics on the monthly median listing price for the Seattle-Tacoma-Bellevue CBSA. Unsurprisingly, we found that historical monthly median listing prices act as the best predictor of median listing prices in our analysis. We also found that the historical median days on market data was a solid predictor of the monthly median listing price as well. Without a doubt, the active listings count LSTM was the poorest model in our comparison.

We believe that there are a few potential limitations to our models and analysis. In our view, the primary limitation in our LSTM model analysis was our input data sample size. Our data spanned the course of 5 years, but given that we worked with monthly data, it only translated to roughly 60 periods of data. While this is a workable sample size, a larger dataset likely would have allowed us to build stronger models and make better predictions and conclusions. As a result, a consideration for the future would be to find larger housing inventory datasets.

Another consideration for next steps would be to switch up our dependent and independent variables. For instance, it would be interesting to use the median days on market as the dependent variable, and test the performance of the median listing price and active listings count as the independent features variables.


