# !pip list

# !pip install opencv-python matplotlib imageio gdown tensorflow

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio


# tf.config.list_physical_devices('GPU')

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     pass

# tf.__version__

# 1. Build Data Loading Functions

import gdown

# url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
# output = 'data.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('data.zip')

def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)

char_to_num.get_vocabulary()

char_to_num(['n','i','c','k'])

num_to_char([14,  9,  3, 11])

def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# def load_data(path: str):
#     path = bytes.decode(path.numpy())
#     #file_name = path.split('/')[-1].split('.')[0]
#     # File name splitting for windows
#     file_name = path.split('/')[-1].split('.')[0]
#     video_path = os.path.join('data','s1',f'{file_name}.mpg')
#     alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
#     frames = load_video(video_path)
#     alignments = load_alignments(alignment_path)

#     return frames, alignments

test_path = 'bbal6n.mpg'

# print(tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('/')[-1].split('.')[0])

# frames, alignments = load_data(tf.convert_to_tensor(test_path))

frames = load_video('data/s1/bbal6n.mpg')

print("frames.shape: ", frames.shape)

# tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])

# def mappable_function(path:str) ->List[str]:
#     result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
#     return result

# 2. Create Data Pipeline

# from matplotlib import pyplot as plt

# data = tf.data.Dataset.list_files('data/s1/*.mpg')
# data = data.cache() # added caching (Tejas)
# data = data.shuffle(500, reshuffle_each_iteration=False)
# data = data.map(mappable_function)
# data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
# data = data.prefetch(tf.data.AUTOTUNE)
# # Added for split
# train = data.take(450)
# test = data.skip(450)

# len(test), len(train)

# frames, alignments = data.as_numpy_iterator().next()

# len(frames), len(alignments)

# len(frames[0]), len(frames[1]), len(alignments[0]), len(alignments[1])

# plt.imshow(frames[0][40])

# sample = data.as_numpy_iterator()

# val = sample.next();

# # 0:videos, 0: 1st video out of the batch,  0: return the first frame in the video
# plt.imshow(val[0][0][35])

# print(tf.strings.reduce_join([num_to_char(word) for word in val[1][0]]))

# 3. Design the Deep Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# print(data.as_numpy_iterator().next()[0][0].shape)

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

print(model.summary())

# yhat = model.predict(val[0])

# tf.strings.reduce_join([num_to_char(x) for x in tf.argmax(yhat[0],axis=1)])

# tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[0]])

print("model input shape:", model.input_shape)

print("model output shape:", model.output_shape)

# 4. Setup Training Options and Train

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

# model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

#use this cell instead of above when using checkpoints
#we are using previous version of adam on which checkpoints are calculated
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss=CTCLoss)

checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True)

schedule_callback = LearningRateScheduler(scheduler)

# example_callback = ProduceExample(test)

# model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])

# 5. Make a Prediction

url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
output = 'checkpoints.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('checkpoints.zip', 'models')

model.load_weights('models/checkpoint')

# test_data = test.as_numpy_iterator()

# sample = test_data.next()

# plt.imshow(sample[0][0][45])

# sample[1]

sample_frames = load_video('data/s1/bbal6n.mpg')
sample_alignments = load_alignments('data/alignments/s1/bbal6n.align')

print("sample frames", sample_frames.shape)
reshaped_sample_frames = tf.reshape(sample_frames, [-1, 75, 46, 140, 1])
print("reshaped sample frames", reshaped_sample_frames.shape)
print("sample alignments: ", sample_alignments)
print("sample alignments shape: ", sample_alignments.shape)

yhat = model.predict(reshaped_sample_frames)

print(yhat.shape)

print('~'*50, 'REAL TEXT', '~'*50,)
print(tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(sample_alignments.numpy()).numpy()]).numpy())

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

print('~'*50, 'PREDICTIONS', '~'*50)
for sentence in decoded:
    print(tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy())


# print('~'*100, 'PREDICTIONS')
# print(tf.strings.reduce_join([num_to_char(x) for x in tf.argmax(yhat,axis=1)]))


# Test on a Video

# sample_frames = load_video('data/s1/bras9a.mpg')
# sample_alignments = load_alignments('data/alignments/s1/bras9a.align')

# print('~'*50, 'REAL TEXT', '~'*50,)
# print(tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(sample_alignments.numpy()).numpy()]).numpy())

# yhat = model.predict(sample_frames)

# decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

# print('~'*50, 'PREDICTIONS', '~'*50,)
# print(decoded)
# print(tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded)

# Save the model

model.save('models/vispnet')
print("model saved at models/vispnet")

# import pickle
# # Serialize and save the model
# with open('vispnet.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)

# from joblib import dump

# # Save the trained model using joblib
# dump(model, 'models/vispnet.joblib')

# Calculate Accuracy

# !pip install Levenshtein

# import Levenshtein

# def calculate_wer(y_true, y_pred):
#   wer = 0
#   for i in range(len(y_true)):
#     true_words = tf.strings.reduce_join([num_to_char(word) for word in y_true[i]]).numpy().decode('utf-8').split()
#     pred_words = tf.strings.reduce_join([num_to_char(word) for word in y_pred[i]]).numpy().decode('utf-8').split()
#     wer += Levenshtein.distance(true_words, pred_words)
#   return wer / len(y_true)

# cumulative_wer = 0

# for sample in test_data:
#   test_labels = sample[1]
#   test_preds = model.predict(sample[0])
#   cumulative_wer += calculate_wer(test_labels, test_preds)

# test_wer = cumulative_wer/len(test)
# print(f"Test WER: {test_wer}")