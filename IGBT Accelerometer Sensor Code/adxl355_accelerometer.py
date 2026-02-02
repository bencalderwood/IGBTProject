import board
import busio
import adafruit_adxl34x

spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs = board.D5  # Chip select pin

accelerometer = adafruit_adxl34x.ADXL355(spi, cs)

accelerometer.range = adafruit_adxl34x.Range.RANGE_2G
