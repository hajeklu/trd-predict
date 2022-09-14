# DateTime Stamp;Bar OPEN Bid Quote;Bar HIGH Bid Quote;Bar LOW Bid Quote;Bar CLOSE Bid Quote;Volume


from ast import ListComp
from pickletools import optimize
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import csv
from candle import Candle, CandleTime, minuteToHour

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout,  LSTM

def csvToCandles(file):
    # Load data
    print("Loading: " + file)
    data = []
    # ../data_202207.csv
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            date = row[0]
            openn = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            data.append(Candle(date, openn, low, high, close, CandleTime.minute))
    return data


def trainModel(trainCandles, prediction_minutes = 60, model_name = 'lstm_1m_10_model'):
    tf.keras.backend.clear_session()
    # scale
    print("Scaling..")
    scaler = MinMaxScaler(feature_range=(0,1))
    opens = scaler.fit_transform([[o.open] for o in trainCandles])
    highs = scaler.fit_transform([[o.high] for o in trainCandles])
    lows = scaler.fit_transform([[o.low] for o in trainCandles])
    closes = scaler.fit_transform([[o.close] for o in trainCandles])
    normalizedCandles = []

    for i in range(0, len(trainCandles)):
        date = trainCandles[i]
        openn = float(opens[i][0])
        high = float(highs[i][0])
        low = float(lows[i][0])
        close = float(closes[i][0])
        normalizedCandles.append(Candle(date, openn, low, high, close, CandleTime.minute))



    #Prepare Data
    print("Preparing data..")
    x_train = []
    y_train = []

    for x in range(prediction_minutes, len(normalizedCandles)):
        xdata = normalizedCandles[x-prediction_minutes:x]
        predictionData = []
        for candleX in xdata:
            predictionData.append([candleX.open, candleX.low, candleX.high, candleX.close])
        candleY = normalizedCandles[x]
        x_train.append(predictionData)
        y_train.append([candleY.open, candleY.low, candleY.high, candleY.close])

    # split train and test
    x_toSplit, y_toSplit = x_train, y_train
    sizeOf70percentage = int(len(x_toSplit)/100*70)
    x_test = np.array(x_toSplit[sizeOf70percentage:len(x_toSplit)])
    y_test = np.array(y_toSplit[sizeOf70percentage:len(x_toSplit)])
    x_train = np.array(x_toSplit[0: sizeOf70percentage])
    y_train = np.array(y_toSplit[0: sizeOf70percentage])


    print("Total size of samples: " + str(len(x_test)))
    model=None

    if (os.path.abspath("./" + model_name)): # you won't have a model for first iteration
        print("Loading model..")
        model = load_model(model_name)
    else:
        print("Creatng model..")
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=4))
        model.compile(optimizer='Adam', loss='mean_squared_error', metrics=["accuracy"])
    
    
    history = model.fit(
        x_train, 
        y_train, 
        validation_data=(x_test, y_test), 
        epochs=8, 
        batch_size=32,
        use_multiprocessing = False,
        max_queue_size=100,
        workers = 0)

    model.save(model_name)


def testModel(name, prediction_minutes = 60,):
    model = load_model(name)
    candles = csvToCandles('../data/202209.csv')

    test_data = []
    for c in range(0, len(candles) - prediction_minutes):
        oneTestCandles = candles[0+c: prediction_minutes+c]
        oneTestCandlesTransformed = []
        for oneCandle in oneTestCandles:
            oneTestCandlesTransformed.append([oneCandle.open, oneCandle.low, oneCandle.high, oneCandle.close])
        test_data.append(oneTestCandlesTransformed)

    test_data = np.array(test_data)

    predict_candles = model.predict(test_data)

    predicted_data = []
    for candle in predict_candles:
        predicted_data.append(candle[0])




    candles = candles[60:70]
    # x axis values
    x = [o.date for o in candles]
    # corresponding y axis values
    y = [o.open for o in candles]
    
    # plotting the points 
    plt.plot(x, predicted_data)
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    
    # giving a title to my graph
    plt.title('My first graph!')
    
    # function to show the plot
    plt.show()


file_to_load = ['../data/202208.csv',
                '../data/2012.csv',
                '../data/2013.csv',
                '../data/2014.csv',
                '../data/2015.csv',
                '../data/2016.csv',
                '../data/2017.csv',
                '../data/2018.csv',
                '../data/2019.csv',
                '../data/2020.csv',
                '../data/2021.csv',
                '../data/202207.csv',
                '../data/202206.csv',
                '../data/202205.csv',
                '../data/202204.csv',
                '../data/202203.csv',
                '../data/202202.csv',
                '../data/202201.csv'
                ]

for train_chunk in file_to_load:
    candles = csvToCandles(train_chunk)
    trainModel(candles, 10, 'lstm_1m_10_model')
    #testModel('lstm_1m_model')