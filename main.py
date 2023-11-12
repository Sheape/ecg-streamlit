import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras
import ecg_plot
from scipy.io import loadmat
from scipy.signal import filtfilt, butter
from PIL import Image
import os
import numpy as np

def sepres():
  n_feature_maps = 64
  input_shape = (5000,12)
  input_layer = keras.layers.Input(input_shape)

  # BLOCK 1
  conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same', data_format='channels_last')(input_layer)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)
  conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same', data_format='channels_last')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)
  conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same', data_format='channels_last')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.SeparableConv1D(filters=n_feature_maps, kernel_size=1, padding='same',data_format='channels_last')(input_layer)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
  output_block_1 = keras.layers.add([shortcut_y, conv_z])
  output_block_1 = keras.layers.Activation('relu')(output_block_1)

  # BLOCK 2
  conv_x = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same', data_format='channels_last')(output_block_1)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)
  conv_y = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',data_format='channels_last')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)
  conv_z = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same',data_format='channels_last')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same', data_format='channels_last')(output_block_1)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
  output_block_2 = keras.layers.add([shortcut_y, conv_z])
  output_block_2 = keras.layers.Activation('relu')(output_block_2)

  # BLOCK 3
  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same', data_format='channels_last')(output_block_2)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)
  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same', data_format='channels_last')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)
  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same', data_format='channels_last')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # no need to expand channels because they are equal
  shortcut_y = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same',data_format='channels_last')(output_block_2)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
  output_block_3 = keras.layers.add([shortcut_y, conv_z])
  output_block_3 = keras.layers.Activation('relu')(output_block_3)

  # Block 4
  conv_x = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same',data_format='channels_last', depth_multiplier=12)(output_block_3)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)
  conv_y = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',data_format='channels_last')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)
  conv_z = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same',data_format='channels_last')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # expand channels for the sum
  shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same',data_format='channels_last')(output_block_1)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
  output_block_4 = keras.layers.add([shortcut_y, conv_z])
  output_block_4 = keras.layers.Activation('relu')(output_block_4)

  # BLOCK 5
  conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same', data_format='channels_last')(output_block_4)
  conv_x = keras.layers.BatchNormalization()(conv_x)
  conv_x = keras.layers.Activation('relu')(conv_x)
  conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same', data_format='channels_last')(conv_x)
  conv_y = keras.layers.BatchNormalization()(conv_y)
  conv_y = keras.layers.Activation('relu')(conv_y)
  conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same', data_format='channels_last')(conv_y)
  conv_z = keras.layers.BatchNormalization()(conv_z)

  # no need to expand channels because they are equal
  shortcut_y = keras.layers.SeparableConv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same',data_format='channels_last')(output_block_2)
  shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
  output_block_5 = keras.layers.add([shortcut_y, conv_z])
  output_block_5 = keras.layers.Activation('relu')(output_block_5)

  # FINAL
  gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_5)
  output_layer = keras.layers.Dense(28, activation='sigmoid')(gap_layer)
  model = keras.models.Model(inputs=input_layer, outputs=output_layer)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.BinaryAccuracy(
  name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'),
                  tf.keras.metrics.AUC(
      num_thresholds=200,
      curve="ROC",
      summation_method="interpolation",
      name="AUC",
      dtype=None,
      thresholds=None,
      multi_label=True,
      label_weights=None,
  )])
  #@title Plot model for better visualization
  # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  return model

# Initial page config for streamlit
st.set_page_config(
    page_title='HeartRead',
    layout='wide'
)

# Load the machine learning model for ECG
@st.cache_resource
def load_ml():
    model = load_model('sep_resnet_model.h5')
    return model

@st.cache_resource
def load_resnet():
    model = sepres()
    model.load_weights('sep_resnet_model.h5')
    return model

# model = load_ml()
model = load_resnet()

# Analyzing
def load_mat(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    Wn = 0.1
    b, a = butter(4, Wn, 'low', analog=False)
    smooth_ecg = filtfilt(b, a, data)

    return smooth_ecg

def plot_ecg(filename):
    ecg_data = load_mat(filename)
    ecg_plot.plot_12(ecg_data/1000, sample_rate=500, title="12-Lead ECG Graph")
    return ecg_plot

# Front-end

st.title(':blue[Heart**Read**]')
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

if 'ecg_pred' not in st.session_state:
    st.session_state['ecg-pred'] = ''

def inference_ecg(filename):
    data = load_mat(filename)
    data = data.reshape(1, 5000, 12)
    prediction = model.predict(data)
    st.session_state['ecg_pred'] = prediction

# Initialize Layout
col1, col2 = st.columns([1.5, 3])

with col1:
    st.markdown('#### Upload your digital ECG recording')
    ecg_file = st.file_uploader('',
                                accept_multiple_files=True, type=['mat', 'dat'])
    st.divider()
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        gender = st.radio('Gender', ('Male', 'Female'))
    with subcol2:
        age = st.number_input('Age',
                              min_value=0,
                              max_value=150,
                              format='%d')
    symptoms = st.text_area('Describe the patient',
                            placeholder='What are the symptoms and feelings that your patient experiences?\n\n Description of ECG graph',
                            height=250
                            )
    prediction = st.button('Analyze', type='primary', on_click=inference_ecg, args=[ecg_file[0].name], use_container_width=True)
    st.text(st.session_state['ecg_pred'])

with col2:
    st.header('ECG Graph Analysis')
    st.image(Image.open('ecg-graph.png'))
    st.progress(0.69, text='Atrial Fibrillation')
    st.progress(0.56, text='Right Bundle Branch Block')
    st.progress(0.35, text='T wave change')
