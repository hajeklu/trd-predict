from candle import CandleTime
from sklearn.preprocessing import MinMaxScaler
from data_provider import readCSV
import pandas as pd
import sys, json, numpy as np
from tensorflow.keras.models import load_model

scaler = MinMaxScaler()
scaler.data_max_ = [1.39621, 1.39927, 1.3948, 1.39623]
scaler.data_min_ = [0.99063, 0.99226, 0.99002, 0.99061]

candlesArray = json.loads(sys.argv[1])

candlesDF = pd.DataFrame(candlesArray, columns = ['open','high','low', 'close'])
candlesDF = scaler.fit_transform(candlesDF)
model = load_model('/Users/luboshajek/Homie/trading/predict/src/lstm_1m_60_model')
predicted_candle = model.predict(np.array([candlesDF]))
predicted_candle = scaler.inverse_transform(predicted_candle)
print(json.dumps({"candle": predicted_candle[0].tolist()}))