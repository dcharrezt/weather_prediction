from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from math import sqrt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import lstm_rnn as ms

lstm_model = load_model('models/lstm_24.h5')
lstm_model.load_weights('models/lstm_weights_24.h5')

# one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def column(matrix, i):
    return [row[i] for row in matrix]

scaler, train_scaled, test_scaled = ms.get_unshuffled_dataset()

train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)

lstm_model.predict(train_reshaped, batch_size=24)
 
print(test_scaled)

X, y = test_scaled[:, 0:-1], test_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

scores = lstm_model.evaluate(X, y, batch_size=24, verbose=1)
print("\nmse:", scores)
#print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))

predictions = lstm_model.predict(X, batch_size=24, verbose=1)

print(predictions)
print(len(predictions))
print(len(y))
	
print(y[-24:])
print(predictions[-24:])

msra = [x[0] for x in predictions]
print(msra[-24:])

# line plot of observed vs predicted
pyplot.plot(y[-24:])
pyplot.plot(msra[-24:])
pyplot.savefig("images/a.png")
