from enum import Enum


class Candle:
  def __init__(self, date, open, low, high, close, candleTime):
    self.date = date
    self.open = open
    self.high = high
    self.low = low
    self.close = close
    self.candleTime = candleTime
    
    def __repr__(self):
      return self.date.strftime() + ": " + str(self.open) + ", " + str(self.high) + ", " + str(self.low) + ", " + str(self.close)

    def __str__(self):
      return self.date.strftime() + ": " + str(self.open) + ", " + str(self.high) + ", " + str(self.low) + ", " + str(self.close)


class CandleTime(Enum):
  minute = 1
  hour = 2
  day = 3
 