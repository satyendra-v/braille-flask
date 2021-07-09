# Flask utils
from flask import Flask, render_template, request

# Keras
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from gtts import gTTS
import sys
from skimage import io
import numpy as np
import cv2
import imutils

IMAGE_FOLDER = 'static/images/'
AUDIO_FOLDER = 'static/speech/'
audio_name = "speech.mp3"

# Model saved with Keras model.save()
MODEL_PATH = 'BrailleNet.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Define a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['UPLOAD_FOLDER'] = AUDIO_FOLDER

def model_predict(seg, model):
    seg1 = image.load_img(seg)
    seg1 = seg1.resize((28,28))
    x=np.asarray(seg1)
    x=np.expand_dims(x,axis=0)
    a=np.argmax(model.predict(x), axis=1)
    return a[0]

def image_segmentation(img):
    width_of_braille_cell = 55
    print(img.shape)
    height_of_braille_image, width_of_braille_image, ch = img.shape
    no_of_braille_cells = width_of_braille_image // width_of_braille_cell
    segments = []
    segment = 1
    x, y, w, h = 10, 15, 45, 70
    for i in range(no_of_braille_cells):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        new_img = img[y:y+h, x:x+w]
        cv2.imwrite("segments/" + str(segment) + ".png", new_img)
        segments.append("segments/" + str(segment) + ".png")
        segment = segment+1
        x = x+55
    return segments

def get_text(segments):
    final_text = ""
    for seg in segments:
        category = model_predict(seg, model)
        character = chr(ord('a') + category)
        if(category == 26):
            final_text = final_text + " "
        else:
            final_text = final_text + character
    return final_text

def convert_to_speech(final_text):
    speech_object = gTTS(text = final_text, lang='en', slow=False)
    audio_path = AUDIO_FOLDER + audio_name
    speech_object.save(audio_path)

@app.route('/', methods = ['GET'])
def index_view():
    # Main page
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def load():
    # Get the file from post request
    imageFile = request.files['image']
    image_path = "static/images/" + imageFile.filename
    imageFile.save(image_path)
    url = image_path 
    img = image.load_img(url)
    img = io.imread(url)
    img = imutils.resize(img, 1500)    
    
    # Image Segmentation
    segments = image_segmentation(img)
    # Predict every segment and get text
    final_text = get_text(segments)
    # Conversion of text to speech
    convert_to_speech(final_text)  

    full_filename = IMAGE_FOLDER + imageFile.filename
    audio_file = AUDIO_FOLDER + audio_name
    return render_template('data.html', final_text=final_text, usr_img = full_filename, usr_audio = audio_file)

if __name__ == '__main__':
    app.run(debug=True, port=8000)