import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
import matplotlib.pyplot as plt
import shutil
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

test_path = '/content/brain_mri/Testing'
train_path = '/content/brain_mri/Training'

img_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=5, width_shift_range=0.05, height_shift_range=0.05, rescale=1/255.0)

for lbl in classes:
  req = 2000 - len(os.listdir(os.path.join(train_path, lbl)))
  class_files = os.listdir(os.path.join(train_path, lbl))[:req]
  i = 0

  for f in class_files:
    curr = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(os.path.join(train_path, lbl, f), target_size=(256,256))).reshape(1, 256, 256, 3)
    next(img_gen.flow(curr, batch_size=1, save_to_dir=os.path.join(train_path, lbl), save_format='jpg'))

for lbl in classes:
  print(lbl, ': ', len(os.listdir(os.path.join(train_path, lbl))))

val_path = '/content/brain_mri/Validation'

for c in classes:
  files = os.listdir(os.path.join(test_path, c))[:300]

  for f in files:
    curr = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(os.path.join(test_path, c, f), target_size=(256,256))).reshape(1, 256, 256, 3)
    next(img_gen.flow(curr, batch_size=1, save_to_dir=os.path.join(val_path, c), save_format='jpg'))
  # img_gen.flow_from_directory(os.path.join(test_path, c), batch_size=1, save_to_dir=os.path.join(val_path, c), save_format='jpg'):

for lbl in classes:
  print(lbl, ': ', len(os.listdir(os.path.join(val_path, lbl))))

def normalize(image, label):
  image = tf.cast(image, tf.float32) / 255.0
  return image, label

def one_hot(image, label):
  label = tf.one_hot(label, depth=4)
  return image, label

train_gen = keras.preprocessing.image_dataset_from_directory(train_path)
test_gen = keras.preprocessing.image_dataset_from_directory(test_path)
val_gen = keras.preprocessing.image_dataset_from_directory(val_path)

# One-Hot Encode

train_gen = train_gen.map(one_hot)
test_gen = test_gen.map(one_hot)
val_gen = val_gen.map(one_hot)

# Normalize

train_gen = train_gen.map(normalize)
test_gen = test_gen.map(normalize)
val_gen = val_gen.map(normalize)

base_conv = keras.applications.vgg16.VGG16(include_top=False, input_shape=(256,256,3))

for layer in base_conv.layers[:-5]:
  layer.trainable = False

x = keras.layers.GlobalAveragePooling2D()(base_conv.output)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(4, activation='softmax')(x)

model = keras.models.Model(inputs=base_conv.input, outputs=x)

lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

# model.summary()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

mcp = keras.callbacks.ModelCheckpoint('/content/best_model.keras', save_best_only=True)

history = model.fit(train_gen, batch_size=32, epochs=5, callbacks=[mcp, lr_schedule], validation_data=val_gen)

loaded_model = keras.models.load_model('/content/best_model.keras')

pred = loaded_model.evaluate(test_gen)
print("Loss:", pred[0], "& Accuracy:", pred[1])