# Required imports
from collections import deque
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog as fd
import time

filename = 0

class Parameters:
    def __init__(self):
        self.CLASSES = (
            open("C:/Users/phada/Downloads/human-activity-recognition-with-gui/human-activity-recognition-with-gui/model/action_recognition_kinetics.txt").read().strip().split("\n")
        )
        self.ACTION_RESNET = "C:/Users/phada/Downloads/human-activity-recognition-with-gui/human-activity-recognition-with-gui/model/resnet-34_kinetics.onnx"
        self.VIDEO_PATH = filename
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

param = Parameters()

captures = deque(maxlen=param.SAMPLE_DURATION)

print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)
print("[INFO] accessing video stream...")

vs = cv2.VideoCapture(param.VIDEO_PATH)

# Create a Tkinter window for the alert message
alert_window = tk.Tk()
alert_window.title("Alert")

# Flag to track if an interested action has been detected
action_detected = False

# Function to display an alert message in the Tkinter window and stop the code
def display_alert_message_and_stop(message):
    global alert_window  # Declare alert_window as global

    # Check if the window is still valid
    if alert_window is not None and tk._default_root is not None:
        alert_label = tk.Label(alert_window, text=message, font=("Helvetica", 16))
        alert_label.pack()
        alert_window.update()

        def destroy_window():
            alert_window.destroy()
            global action_detected
            action_detected = True  # Set the global flag to True after destroying the window

        alert_window.after(1000, destroy_window)  # Destroy the window after 1000 milliseconds (1 second)

# Define the array of interested actions
interested_actions = ["punching person (boxing)", "sword fighting", "smoking", "smoking hookah", "punching person"]  # Add your specific actions here

while True:
    (grabbed, capture) = vs.read()

    if not grabbed:
        print("[INFO] no capture read from stream - exiting")
        break

    capture = cv2.resize(capture, dsize=(600, 350))
    captures.append(capture)

    if len(captures) < param.SAMPLE_DURATION:
        continue

    imageBlob = cv2.dnn.blobFromImages(
        captures,
        1.0,
        (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
        (114.7748, 107.7354, 99.4750),
        swapRB=True,
        crop=True,
    )

    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    net.setInput(imageBlob)
    outputs = net.forward()
    label = param.CLASSES[np.argmax(outputs)]

    if label in interested_actions and not action_detected:
        alert_message = f"Alert: {label} detected!"
        display_alert_message_and_stop(alert_message)
        time.sleep(3)  # Stop the code execution after displaying the alert message for 1 second

    cv2.rectangle(capture, (0, 500), (380, 300), (255, 240, 245), -1)
    cv2.putText(capture, label, (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Human Activity Recognition", capture)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()

