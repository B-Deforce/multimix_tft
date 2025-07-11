
def inverse_minmax(y, scaling_max, scaling_min):
    unscaled = ((y + 1) * (scaling_max - scaling_min) / 2) + scaling_min
    return unscaled
