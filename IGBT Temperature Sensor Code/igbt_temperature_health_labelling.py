def assign_temperature_health_label(T_mean):
    if T_mean < 75:
        return "Healthy"
    elif 75 <= T_mean <= 125:
        return "Warning"
    else:
        return "Fault"