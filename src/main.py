# DateTime Stamp;Bar OPEN Bid Quote;Bar HIGH Bid Quote;Bar LOW Bid Quote;Bar CLOSE Bid Quote;Volume


from ast import ListComp
from cProfile import label
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

def normalize_candles(candles):
    print("Scaling..")
    scaler = MinMaxScaler(feature_range=(0,1))
    opens = scaler.fit_transform([[o.open] for o in candles])
    highs = scaler.fit_transform([[o.high] for o in candles])
    lows = scaler.fit_transform([[o.low] for o in candles])
    closes = scaler.fit_transform([[o.close] for o in candles])
    normalizedCandles = []

    for i in range(0, len(candles)):
        date = candles[i].date
        openn = float(opens[i][0])
        high = float(highs[i][0])
        low = float(lows[i][0])
        close = float(closes[i][0])
        normalizedCandles.append(Candle(date, openn, low, high, close, CandleTime.minute))
    return normalizedCandles


def trainModel(trainCandles, prediction_minutes = 60, model_name = 'lstm_1m_10_model'):
    tf.keras.backend.clear_session()
    normalizedCandles = trainCandles
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

    if (os.path.isdir(model_name)): # you won't have a model for first iteration
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
        epochs=5, 
        batch_size=32)

    model.save(model_name)


def testModel(name, prediction_minutes = 60):
    model = load_model(name)
    candles = csvToCandles('../data/202209.csv')
    candles = normalize_candles(candles)
    candles = candles[:100]

    test_data = []
    for c in range(prediction_minutes, len(candles)):
        oneTestCandles = candles[c-prediction_minutes:c]
        oneTestCandlesTransformed = []
        for oneCandle in oneTestCandles:
            oneTestCandlesTransformed.append([oneCandle.open, oneCandle.low, oneCandle.high, oneCandle.close])
        test_data.append(oneTestCandlesTransformed)

    test_data = np.array(test_data)
    predict_candles = model.predict(test_data)

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



def main():
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
    # normalize all data
    all_candles = []
    for train_chunk in file_to_load:
        candles = csvToCandles(train_chunk)
        all_candles.extend(candles)
    normalized_all_candles = normalize_candles(all_candles)

    # taking data by size of file
    count = 0
    for train_chunk in file_to_load:
        size_of_file = len(csvToCandles(train_chunk))
        candles_to_train = normalized_all_candles[count:count+size_of_file]
        trainModel(candles_to_train, 60, 'lstm_1m_60_model')





main()
    # testModel('lstm_1m_10_model', 10)