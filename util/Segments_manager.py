
import numpy as np
        
def First2Zero(segments):
    # leva o valor mínimo para zero
    if np.min(segments) < 0:
        segments = segments - np.min(segments)
    init = segments[0, 0]
    if init == 0: return segments
    final = (segments == 0)*init
    initial = (segments == init)*init
    segments = segments - initial + final
    return segments
