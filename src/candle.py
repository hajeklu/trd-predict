from enum import Enum


class Candle:
  def __init__(self, date, open, low, high, close, candleTime):
    self.date = date
    self.open = open
    self.high = high
    self.low = low
    self.close = close
    self.candleTime = candleTime


class CandleTime(Enum):
  minute = 1
  hour = 2


def minuteToHour(data):
  data_1h = {}
  for i in range(0, len(data) - 1):
    candle_m1 = data[i]
    candle_m2 = data[i + 1]
    candle_1h = data_1h[candle_m1.date]
    if candle_1h is None:
      candle_1h = candle_m1
    elif int(candle_1h.date) == int(candle_m1.date):
      mergeCandle(candle_1h, candle_m1)

    if candle_m1.date != candle_m2:
      candle_1h.close = candle_m1.close
    
  return data_1h


def mergeCandle(candle1_base, candle2_new):
  if int(candle1_base.date) != int(candle2_new.date):
    raise Exception('Cannot megre with different date')
  
  candle1_base.high = getBigger(candle1_base.high, candle2_new.high)
  candle1_base.low = getSmaller(candle1_base.low, candle2_new.low)




def getBigger(a, b):
    if a > b:
     return a
    else:
      return b 

def getSmaller(a, b):
    if a < b:
      return a
    else:
      return b 