def assign_vibration_health_label(vib_rms, vib_kurtosis):
    if vib_rms < 0.05 and vib_kurtosis < 3:
        return "Healthy"
    elif vib_rms < 0.15:
        return "Warning"
    else:
        return "Fault"
