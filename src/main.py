# DateTime Stamp;Bar OPEN Bid Quote;Bar HIGH Bid Quote;Bar LOW Bid Quote;Bar CLOSE Bid Quote;Volume


from ast import ListComp
from cProfile import label
from pickletools import optimize
from data_provider import readCSV
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import csv
from candle import Candle, CandleTime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout,  LSTM


def normalize_candles(candles):
    print("Scaling..")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    candles[['open', 'high', 'low', 'close']] = scaler.fit_transform(
        candles[['open', 'high', 'low', 'close']])
    return candles


def trainModel(trainCandles, prediction_minutes=60, model_name='lstm_1m_10_model'):
    tf.keras.backend.clear_session()
    # Prepare Data
    print("Preparing data..")

    x_train = []
    y_train = []
    normalizedCandles = trainCandles[[
        'open', 'high', 'low', 'close', 'avg_price', 'ohlc_price', 'oc_diff']].to_numpy(copy=True)
    for x in range(prediction_minutes, len(normalizedCandles)):
        xdata = normalizedCandles[x-prediction_minutes:x]
        predictionData = []
        for candleX in xdata:
            predictionData.append(
                [candleX[0], candleX[1], candleX[2], candleX[3], candleX[4], candleX[5], candleX[6]])
        candleY = normalizedCandles[x]
        x_train.append(predictionData)
        y_train.append([candleY[0], candleY[1], candleY[2],
                       candleY[3], candleY[4], candleY[5], candleY[6]])

    print("Spliting..")
    # split train and test
    x_toSplit, y_toSplit = x_train, y_train
    sizeOf70percentage = int(len(x_toSplit)/.90)
    x_test = np.array(x_toSplit[sizeOf70percentage:len(x_toSplit)])
    y_test = np.array(y_toSplit[sizeOf70percentage:len(x_toSplit)])
    x_train = np.array(x_toSplit[0: sizeOf70percentage])
    y_train = np.array(y_toSplit[0: sizeOf70percentage])

    print("Total size of samples: " + str(len(x_train)))
    model = None

    if (os.path.isdir(model_name)):  # you won't have a model for first iteration
        print("Loading model..")
        model = load_model(model_name)
    else:
        print("Creatng model..")
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                  input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=7))
        model.compile(optimizer='Adam', loss='mean_squared_error',
                      metrics=['mae', 'mse'])

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32)

    model.save(model_name)


def testModel(name, prediction_minutes=60):
    model = load_model(name)
    candles = readCSV('../data/202209.csv', CandleTime.minute).head(60)
    candles = normalize_candles(candles[['open', 'high', 'low', 'close']])
    test_data = np.array([candles])
    predict_candles = model.predict(test_data)

    print(predict_candles)
    return
    predicted_data = []
    for candle in predict_candles:
        predicted_data.append(candle[0])

    # x axis values
    x = []
    for c in candles:
        x.append(c.date)

    # corresponding y axis values
    y = [o.open for o in candles][prediction_minutes:]
    x = x[prediction_minutes:]

    # plotting the points
    plt.plot(x, predicted_data, label="Predicted")
    plt.plot(x, y, label="Real")
    plt.legend(loc="upper left")
    # naming the x axis
    plt.xlabel('time')
    # naming the y axis
    plt.ylabel('price')

    # giving a title to my graph
    plt.title('Predict!')

    # function to show the plot
    plt.show()


def plotCandleStick(candles):
    import plotly.graph_objects as go

    print(candles)

    fig = go.Figure(data=[go.Candlestick(x=candles.index,
                                         open=candles['open'],
                                         high=candles['high'],
                                         low=candles['low'],
                                         close=candles['close'])])

    fig.show()


def addCountFeature(df):
    # Add additional features
    df['avg_price'] = (df['low'] + df['high']) / 2
    # df['range'] = df['high'] - df['low']
    df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
    df['oc_diff'] = df['open'] - df['close']
    return df


def main():
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

    candles = pd.DataFrame()
    # load all files
    for train_chunk in file_to_load:
        print("Loading: " + train_chunk)
        loadedDataFrame = readCSV(train_chunk, CandleTime.hour)
        candles = pd.concat([candles, loadedDataFrame])

    candles = addCountFeature(candles)
    print(candles.head())
    candles = candles.dropna()
    normalized_all_candles = normalize_candles(candles)
    print("Total count of candles: " + str(len(candles)))
    # taking data by size of file
    count = 0
    trainModel(normalized_all_candles, 30, 'lstm_1d_30_model-5')

    exit()
    for train_chunk in file_to_load:
        size_of_file = len(readCSV(train_chunk, CandleTime.hour))
        candles_to_train = normalized_all_candles[count:count+size_of_file]
        count = count + size_of_file


main()
testModel('lstm_1m_60_model')
