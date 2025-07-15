# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://pni.princeton.edu/sites/g/files/toruqf321/files/styles/freeform_750w/public/2023-10/neuroai_0.jpg?itok=CeLoiBmq')
    st.title('LipiCode')
    st.info('This application is specially designed to aid the deaf community.')

st.title('LipiCode Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # Convert tensor to numpy array and preprocess for visualization
        video_np = video.numpy().squeeze()  # Remove batch/channel dimensions (e.g., from [75, 46, 140, 1] to [75, 46, 140])
        
        # Normalize to 0-255 and convert to uint8
        if video_np.dtype != np.uint8:
            video_np = (video_np - video_np.min()) / (video_np.max() - video_np.min())  # Scale to [0, 1]
            video_np = (video_np * 255).astype(np.uint8)  # Convert to 8-bit
        
        # Save and display the GIF
        imageio.mimsave('animation.gif', video_np, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))  # Add batch dimension
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
