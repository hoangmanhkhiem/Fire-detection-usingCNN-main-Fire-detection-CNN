import cv2
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('fire.h5')

# Function to predict fire in a frame
def predict_frame(frame):
    # Resize and preprocess the frame
    frame = cv2.resize(frame, (256, 256))
    img = frame / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)

    # Determine the label text and color
    if prediction[0][0] < 0.5:
        label_text = "No Fire Detection"  # Change the label text here
        color = (0, 255, 0)  # Green for no fire
    else:
        label_text = "Fire Detection"  # Change the label text here
        color = (0, 0, 255)  # Red for fire

    # Display the result on the frame
    cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

# Open the laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict and display the frame
    result_frame = predict_frame(frame)

    # Display the frame with the prediction result
    cv2.imshow('Fire Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
