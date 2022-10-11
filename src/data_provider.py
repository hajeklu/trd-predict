import pandas as pd

from candle import Candle, CandleTime

def readCSV(path, candleTime):
    # Load data
    #print("Loading: " + path)
    data = pd.read_csv(path, names=['Date_Time', 'open', 'high', 'low', 'close', 'volume'], sep=";", index_col=0)

    # Convert the index to datetime
    data.index = pd.to_datetime(data.index, format='%Y%m%d %H%M%S%f')

    if(candleTime == CandleTime.hour):
        data = data.resample('1H').agg({'open': 'first', 
                                 'high': 'max', 
                                 'low': 'min', 
                                 'close': 'last'})
    if(candleTime == CandleTime.day):
        data = data.resample('24H').agg({'open': 'first', 
                            'high': 'max', 
                            'low': 'min', 
                            'close': 'last'})
            
    return data