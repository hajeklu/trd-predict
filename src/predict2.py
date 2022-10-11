from candle import CandleTime
from data_provider import readCSV
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

file_to_load = ['../data/202208.csv',
                '../data/202207.csv',
                '../data/202206.csv',
                '../data/202205.csv',
                '../data/202204.csv',
                '../data/202203.csv',
                '../data/202202.csv',
                '../data/202201.csv',
                '../data/2021.csv',
                '../data/2020.csv',
                '../data/2019.csv',
                '../data/2018.csv',
                '../data/2017.csv',
                '../data/2016.csv',
                '../data/2015.csv',
                '../data/2014.csv',
                '../data/2013.csv',
                '../data/2012.csv',
                '../data/2011.csv',
                '../data/2010.csv',
                '../data/2009.csv',
                '../data/2008.csv',
                '../data/2007.csv',
                '../data/2006.csv',
                ]

df = pd.read_csv('../data/EURUSD.csv', names=[
    'timestamp', 'open', 'high', 'low', 'close', 'adjclose', 'volume'], sep=",", index_col=0)
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df = df.reset_index(level=0)
print(df.count())
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
df.set_index('timestamp', inplace=True)
df = df.astype(float)

# Add additional features
df['avg_price'] = (df['low'] + df['high']) / 2
# df['range'] = df['high'] - df['low']
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
df['oc_diff'] = df['open'] - df['close']
print(df.size)
df = df.dropna()
print(df.size)
print(df.head())


def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Scale and create datasets
target_index = df.columns.tolist().index('close')
high_index = df.columns.tolist().index('high')
low_index = df.columns.tolist().index('low')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df['close'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)

X, y = create_dataset(dataset, look_back=50)
y = y[:, target_index]

train_size = int(len(X) * 0.80)
trainX = X[:train_size]
trainY = y[:train_size]
testX = X[train_size:]
testY = y[train_size:]

model = Sequential()
model.add(
    Bidirectional(LSTM(30, input_shape=(X.shape[1], X.shape[2]),
                       return_sequences=True),
                  merge_mode='sum',
                  weights=None,
                  input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(10, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4, return_sequences=False))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['mae', 'mse'])
print(model.summary())

model.fit(
    trainX,
    trainY,
    validation_data=(testX, testY),
    epochs=5,
    batch_size=32)

model.save('lstm1')
