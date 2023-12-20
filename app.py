from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gtts import gTTS
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

featuress = np.load('features.npy', allow_pickle=True).item()
tokenizerss = joblib.load("tokenizer.pkl")
model_path = 'myModel.h5'
# image_path_to_predict = sys.argv[1]

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text

def text_to_speech(text, language='en', filename='output.mp3'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filename)
    os.system(f"start {filename}") 

def generate_and_display_caption(model_path, image_path, tokenizer, max_length, features):
    model = load_model(model_path)
    img = load_img(image_path, target_size=(229, 229))
    img_array = img_to_array(img)
    img_array = img_array / 255.

    caption = predict_caption(model, os.path.basename(image_path), tokenizer, max_length, features)

    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()

    words = caption.split()[1:-1]
    text_without_first_last = ' '.join(words)
    text_to_speech(text_without_first_last)

generate_and_display_caption(model_path, '1299459562_ed0e064aee.jpg', tokenizerss, 34, featuress)