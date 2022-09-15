import pandas as pd

from candle import Candle, CandleTime

def readCSV(path, candleTime):
    # Load data
    print("Loading: " + path)
    # ../data_202207.csv
    data = pd.read_csv(path, names=['Date_Time', 'open', 'high', 'low', 'close', 'volume'], sep=";", index_col=0)

    # Convert the index to datetime
    data.index = pd.to_datetime(data.index, format='%Y%m%d %H%M%S%f')

    if(candleTime == CandleTime.hour):
        data = data.resample('1H').agg({'open': 'first', 
                                 'high': 'max', 
                                 'low': 'min', 
                                 'close': 'last'})

    candles = []
    for index, row in data.iterrows():
        date = index
        openn = float(row['open'])
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])
        candles.append(Candle(date, openn, low, high, close, CandleTime.minute))
    return candles

    

    

readCSV('../data/202208.csv', CandleTime.hour)