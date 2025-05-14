
import numpy as np

from util.timing import timing
        
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

# @timing
def Clusters2segments(segments, labels):
    n_superpixels = len(labels)
    sp_labels = np.unique(segments)
    new_segments = np.zeros_like(segments)
    for i in range(n_superpixels):
        new_segments[segments == sp_labels[i]] = labels[i]
    return new_segments

def KillSmallSegments(segments, threshold):
    # Transforma o label dos segmentos com quantidade de pixels menor ou igual ao threshold em zero
    unique, counts = np.unique(segments, return_counts=True)
    small_segments = unique[counts <= threshold]
    mask = np.isin(segments, small_segments)
    segments = np.where(mask, 0, segments)
    return segments
