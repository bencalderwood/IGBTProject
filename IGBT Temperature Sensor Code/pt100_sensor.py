 import time
 import board
 import busio
 import digitalio
 import adafruit_max31865


def init_temperature_sensor():
    """Initialise MAX31865 + PT100 sensor"""
    spi = board.SPI()
    cs = digitalio.DigitalInOut(board.D5)  # adjust pin if needed

    sensor = adafruit_max31865.MAX31865(
        spi,
        cs,
        rtd_nominal=100.0,     # PT100
        ref_resistor=430.0
    )

    return sensor
