import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from math import sqrt

import preprocess_data as prep

#################################################################
################## Formating dataset ############################
def format_dataset():
	df = prep.preprocess_aqp_dataset()

	dataset = df[df["date_time"].isin(pd.date_range('2006-12-31', \
		'2017-06-30 23:00:00', freq = 'H'))]

	idx = pd.date_range('2006-12-31', '2017-06-30 23:00:00', \
				freq='H')

	dataset.set_index('date_time', inplace = True)
	dataset.index = pd.DatetimeIndex(dataset.index)
	dataset = dataset.reindex(idx, fill_value = np.nan)

	dataset = dataset.interpolate('linear')
	dataset.drop(['station_id'], axis = 1, inplace = True)
	return dataset	

def get_unshuffled_dataset():
	dataset = format_dataset()
	# transform data to be supervised
	ms = timeseries_to_supervised(dataset, 24)
	print(len(ms))

	# spliting data 20 % to test
	p = 0.79994783516
	temperatures = np.array(ms[['temperature', 'temp_r']])
	train, test = temperatures[:int(len(temperatures)*p)], \
			temperatures[int(len(temperatures)*p):]

	scaler, train_scaled, test_scaled = scale(train, test)
	return scaler, train_scaled, test_scaled



################################################################
##### frame a sequence as a supervised learning problem ########
def timeseries_to_supervised(data, lag):
	df = pd.DataFrame(data)
	columns = [df.shift(lag)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.columns = ['wind_r','temp_r','dew_r', \
			'wind_speed','temperature','dew_point']
	df.dropna(inplace=True)
	return df

###############################################################
################ scaling data from -1 to 1 ####################
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, y):
	inverted_X = scaler.inverse_transform(X)
	inverted_y = scaler.inverse_transform(y)
	return inverted_X, inverted_y

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size,\
					X.shape[1], X.shape[2]),\
						 stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=100, batch_size=batch_size, \
					verbose=2, shuffle=False)
		model.reset_states()
	return model

###################################################################


if __name__ == "__main__":


	#y = np.array(ms['temp_r'])

	batch_size = 24
	n_epochs = 15
	neurons = 4

	lstm_model = fit_lstm(train_scaled, batch_size,\
			 n_epochs, neurons)

	lstm_model.save('models/lstm_24_1500.h5')
	lstm_model.save_weights('models/lstm_weights_24_1500.h5')

	

