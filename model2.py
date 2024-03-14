import cv2
import numpy as np
import tensorflow 
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Load the pre-trained emotion detection model
model_path = "path/to/your/emotion_model.h5"
emotion_model = load_model(model_path)

# Load an image for emotion detection
image_path = "path/to/your/image.jpg"
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize the image to fit the model's input size
img_gray = cv2.resize(img_gray, (48, 48))
img_array = image.img_to_array(img_gray)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Perform emotion prediction
emotion_prediction = emotion_model.predict(img_array)

# Map prediction to emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

# Display the result
cv2.putText(img, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()