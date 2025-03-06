import os
import io
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

custom_objects = {"LSTM": lambda **kwargs: LSTM(**{k: v for k, v in kwargs.items() if k != "time_major"})}
model_path = r"models\model_weights_25.h5"

try:
    model = load_model(model_path, custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model_resnet = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
model_resnet.trainable = False
model_final = Model(model_resnet.input, model_resnet.layers[-2].output)

try:
    with open(r"models\word_to_idx.pkl", "rb") as w2i:
        word_to_idx = pickle.load(w2i)
    with open(r"models\idx_to_word.pkl", "rb") as i2w:
        idx_to_word = pickle.load(i2w)
except Exception as e:
    print(f"Error loading word mappings: {e}")
    exit()

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_final.predict(img, verbose=0)
    return feature_vector

def predict_caption_using_greedySearch(photo):
    inp_text = "startseq"
    max_len = 33
    for _ in range(max_len):
        sequence = [word_to_idx[word] for word in inp_text.split() if word in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding="post")
        pred_label = model.predict([photo, sequence], verbose=0)
        pred_label = pred_label.argmax()
        if pred_label not in idx_to_word:
            break
        pred_word = idx_to_word[pred_label]
        inp_text += " " + pred_word
        if pred_word == "endseq":
            break
    final_caption = inp_text.split(" ")[1:-1]
    return " ".join(final_caption)

def generate_captions_for_folder(folder_path):
    if not os.path.exists(folder_path):
        print("Invalid path! Please enter a correct folder path.")
        return
    
    output_file = os.path.join(folder_path, "captions.txt")
    with open(output_file, "w") as f:
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    image = Image.open(img_path)
                    encoded_img = encode_image(image)
                    caption = predict_caption_using_greedySearch(encoded_img)
                    f.write(f"{img_name}: {caption}\n")
                    print(f"Processed: {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    print(f"Captions saved to {output_file}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ")
    generate_captions_for_folder(folder_path)
