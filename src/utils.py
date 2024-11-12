import matplotlib
matplotlib.use("agg")
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Load the model and classes outside of the function for efficiency
model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/Audio.h5"))
classes = ['Axe', 'BirdChirping', 'Chainsaw', 'Clapping', 'Fire', 'Firework', 'Footsteps', 'Frog', 
           'Generator', 'Gunshot', 'Handsaw', 'Helicopter', 'Insect', 'Lion', 'Rain', 'Silence', 
           'Speaking', 'Squirrel', 'Thunderstorm', 'TreeFalling', 'VehicleEngine', 'WaterDrops', 
           'Whistling', 'Wind', 'WingFlaping', 'WolfHowl', 'WoodChop']

def predict(filepath, output_path):
    try:
        # Load audio file and generate spectrogram
        audio_data, sample_rate = librosa.load(filepath, sr=None)
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plot and save spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sample_rate, hop_length=512, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Preprocess the image
        image = cv2.imread(output_path)
        image = cv2.resize(image, (256, 256))
        input_data = image.reshape((1, 256, 256, 3)).astype('float32') / 255.0

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = classes[int(np.argmax(prediction))]
        
        return predicted_class
    
    except Exception as e:
        raise e
