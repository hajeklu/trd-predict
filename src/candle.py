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
 