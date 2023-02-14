import numpy as np
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard

parent_dir = 'C:/Users/luisl/OneDrive/Documentos/GitHub/pyAudioClassification/example'

# step 1: preprocessing
if np.DataSource().exists(f"{parent_dir}/feat.npy") and np.DataSource().exists(f"{parent_dir}/label.npy"):
    features, labels = np.load('example/feat.npy'), np.load('example/label.npy')
else:
    features, labels = feature_extraction(parent_dir)
    np.save('./feat.npy', features)
    np.save('./label.npy', labels)

# step 2: training
if np.DataSource().exists("example/model.h5"):
    from keras.models import load_model
    model = load_model('example/model.h5')
else:
    model = train(features, labels, epochs=100)
    model.save('./model.h5')

# step 3: prediction
pred = predict(model, 'example/cow_test.wav')
print_leaderboard(pred, 'example/data/')
