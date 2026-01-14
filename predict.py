import pickle
from preprocess import clean_text

model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_emotion(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]
