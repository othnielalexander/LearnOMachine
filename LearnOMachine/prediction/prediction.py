from numpy import *
import fix_yahoo_finance as yf
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

#   Initial Date for the Data Collection
init_date = '2000-1-1'

#   For setting end date as today
now = datetime.datetime.now()

#   Fromatting end_date as required format
end_date = str(now.year)+'-'+str(now.month)+'-'+str(now.day)

#   Downloading Historical data for GOLD from Yahoo Finances API
frame = yf.download('GLD',init_date, end_date)

#   Only Wanted the Closing Price
frame = frame[['Close']]

#   Drop Rows with NULL Data
frame = frame.dropna()

frame.Close.plot(figsize=(10,5))
plt.ylabel("Gold ETF Prices")
plt.show()

frame['S_3'] = frame['Close'].shift(1).rolling(window=3).mean()
frame['S_9'] = frame['Close'].shift(1).rolling(window=9).mean()

frame= frame.dropna()
X = frame[['S_3','S_9']]
X.head()

# Define dependent variable
y = frame['Close']
y.head()

# Split the data into train and test dataset
t=.8
t = int(t*len(frame))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]

# Create a linear regression model
linear = LinearRegression().fit(X_train,y_train)
print ("Linear Regression equation")
print ("Gold ETF Price (y) =", \
round(linear.coef_[0],2), "* 3 Days Moving Average (x1)", \
round(linear.coef_[1],2), "* 9 Days Moving Average (x2) +", \
round(linear.intercept_,2), "(constant)")

# Predicting the Gold ETF prices
predicted_price = linear.predict(X_test)
print('Predicted Price : ',predicted_price[len(predicted_price)-1])
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])

predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['predicted_price','actual_price'])
plt.ylabel("Gold ETF Price")
plt.xlabel("2019-05", "2019-9", "2020-01", "2020-05", "2020-09", "2021-01", "2021-05", "2021-09", "2022-01", "2022-05")
plt.show()

# R square
r2_score = linear.score(X[t:],y[t:])*100
float("{0:.2f}".format(r2_score))
print(r2_score)