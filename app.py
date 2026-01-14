from src.predict import predict_emotion

text = input("Enter your text: ")
emotion = predict_emotion(text)

print("Predicted Emotion:", emotion)
