import numpy as np
## Temperature Feature Extraction of recorded Temperature data from the data logger
def extract_temperature_features(t, T, Tamb = 25): ## extract features from the data logger with ambient temp of 25 deg c
    t = np.array(t)
    T = np.array(T)

    dTdt = np.gradient(T, t)

    features = {
        "T_mean": np.mean(T), ## Mean temperature over the set timeframe
        "T_max": np.max(T), ## Maximum recorded temperature in set timeframe
        "T_std": np.std(T), ## standard deviation of temperature
        "mean_dTdt" : np.mean(dTdt), ## mean rate of change of temperature over timeframe
        "max_dTdt": np.max(dTdt), ## max rate of change of temperature over timeframe (how fast it rises)
        "DeltaT_steady_state": np.mean(T[-5:]) - Tamb, ## steady state temperature
        "time_below_or_equal_75": np.sum(T <= 75) * np.mean(np.diff(t)),## time temperature is within healthy state
        "time_at_75_to_125": np.sum( 75 < T <= 125) * np.mean(np.diff(t)), ## time temperature is in warning state
        "time_above_125": np.sum(T > 125) * np.mean(np.diff(t))  ## time temperature is above warning state
    }

    return features
