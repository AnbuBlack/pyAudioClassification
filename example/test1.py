import numpy as np
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard

features, labels = feature_extraction('C:/Users/luisl/OneDrive/Documentos/GitHub/pyAudioClassification/example/data')
file_name="features"

np.save('%s.npy' % file_name, features)
features = np.load('%s.npy' % file_name)

file_name="label"
np.save('%s.npy' % file_name, labels)
labels = np.load('%s.npy' % file_name)