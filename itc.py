import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
df=pd.read_csv('itc.csv', parse_dates=["Date"],index_col="Date")
df = df[['Close']]
#Forecast
forecast_out = 30 
df['Prediction'] = df[['Close']].shift(-forecast_out)
print(df.tail())
X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)
y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)

#Gdp
gdp=pd.read_csv("gdp.csv",parse_dates=["Date"],index_col="Date")
print(gdp)

#Inflation
inflation=pd.read_csv("inflation.csv",parse_dates=["Date"],index_col="Date")
print(inflation)

#Unemployment
unemployment=pd.read_csv("unemployment.csv",parse_dates=["Date"],index_col="Date")
print(unemployment)

#Joining 
data=df.join(gdp)
data=data.join(inflation)
data=data.join(unemployment)
print(data)

#Interpolation
data.interpolate(method="time",inplace=True)
print(data)

#Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

#Correlation
correlation = data.corr(method='pearson')
print(correlation)

#Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print("svr confidence: ", svr_rbf_confidence)
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

#Create and train the Linear Regression  Model
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

#OLS model
from sklearn import linear_model
import statsmodels.api as sm
regr = linear_model.LinearRegression()
regr.fit(X, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# with statsmodels
X = sm.add_constant(X) # adding a constant
model = sm.OLS(y, X).fit()
predictions = model.predict(X)  
print_model = model.summary()
print(print_model)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

#Create and train the Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)
rf_confidence = rf.score(x_test, y_test)
print("rf confidence: ", rf_confidence)
rf_prediction = rf.predict(x_forecast)
print(rf_prediction)

#Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')