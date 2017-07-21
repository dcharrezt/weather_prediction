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
from matplotlib import pyplot
import numpy as np


model = load_model('models/lstm_24.h5')
model_weights = model.load_weights('models/lstm_weights_24.h5')

