import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.applications.blazeface.BlazeFaceModel(weights="blazeface_tf.h5")

image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_size = (128, 128)
resized_image = cv2.resize(image, input_size)
normalized_image = resized_image / 255.0
input_tensor = np.expand_dims(normalized_image, axis=0)

predictions = model.predict(input_tensor)

boxes = predictions[0]['boxes']
for box in boxes:
    box = [int(coord) for coord in box]
    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)


cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()