# Report: Face Lock Application Analysis

![WhatsApp Image 2025-07-31 at 20 46 25_f9bfc974](https://github.com/user-attachments/assets/e8e7738b-64af-40e6-bcf2-be1825a83adb)
<br>
<br><br>

![WhatsApp Image 2025-07-31 at 20 46 55_db13f876](https://github.com/user-attachments/assets/b73e87bf-5869-4bb8-af81-a36c262a38e1)

## 1. Project Overview

This project is a simple "Face Lock" web application. It uses your computer's webcam to either register a new person's face or verify person from the database.

The application is built using Python and interface created with Streamlit.

The main goal is to demonstrate a complete face recognition system:
*   *Capturing* an image from a live camera.
*   *Analyzing* the face in the image to create a unique signature.
*   *Storing* this signature for new users.
*   *Comparing* a new signature against the stored ones to find a match.

## 2. How It Works 

1.  When you run the script, a web page opens up showing a live feed from your webcam.
2.  On the left side, you can choose one of two modes:
    *   *New Face*: This mode is for adding a new person to the system.
    *   *Verify Identity*: This mode is for checking if a person is already registered.
3.  You position your face in front of the camera and click the button labeled "Round Click".


## 3. Impored Modules

import streamlit as ui
import cv2
import pickle
import os
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity


## 4. Code Breakdown 


#### a. Setup and Helper Functions

*   extract_face_vec(img):
    *   It takes an image as input.
    *   It uses DeepFace.represent() to find a face and convert it into a numerical vector (the signature).
    *   If no face is found in the image, it returns None.

*   store_faces(data) and fetch_faces():
    *   These functions manage the database file, faces.pkl.
    *   fetch_faces() loads the saved faces from the file. If the file doesn't exist, it starts with an empty database.
    *   store_faces() saves the updated database (with any new faces).

```py
# This function turns a face in an image into a list of numbers.
def extract_face_vec(img, model="Facenet"):
    try:
        result = DeepFace.represent(img_path=img, model_name=model, enforce_detection=True, detector_backend='opencv')
        return result[0]["embedding"]
    except:
        return None
```

```py
# These functions save and load the face data from a file.
def store_faces(data):
    with open(vault, "wb") as file:
        pickle.dump(data, file)

def fetch_faces():
    return pickle.load(open(vault, "rb")) if os.path.exists(vault) else {}
```

#### b. Application Flow and Logic

The main part of the script sets up the user interface and handles the logic for the two modes.

1.  The app sets a title, creates the mode selection in the sidebar, starts the webcam, and loads any previously saved faces into the database.

2.  The while loop runs continuously to show the live video feed.
    *   It reads a frame from the camera.
    *   It flips the image so it looks like a mirror.
    *   It displays the image on the screen.
    *   It waits for the user to click button.

3.  It calls extract_face_vec(snapshot) to get the face signature from the captured image.

4.  *Mode-Specific Actions*:
    *   *If Mode is "New Face"*:
        *   It checks if a face vector was successfully created.
        *   If yes, it takes the label (name) you entered, adds the [name, vector] pair to the database, and saves it.
        *   If no face was found, it shows a warning message.

    *   *If Mode is "Verify Identity"*:
        *   It checks if a face vector was created.
        *   If yes, it compares this new vector to *every* stored vector in the database using cosine_similarity.
        *   It finds the person with the *highest similarity score*.
        *   It uses a *threshold of 0.6* and displays a success message with the person's name.
        *   If the score is below 0.6, it concludes the person is not a match and displays an error.
        *   If no face was found in the snapshot, it shows a warning.

```py
# This is the logic for verifying a face
elif mode == "Verify Identity":
    if vector: # If a face was found in the snapshot
        highest = -1
        who = None
        # Loop through every person in the database
        for person, stored in database.items():
            # Compare the new face with the stored face
            score = cosine_similarity(np.array(vector).reshape(1, -1), np.array(stored).reshape(1, -1))[0][0]
            # Keep track of the best match
            if score > highest:
                highest = score
                who = person
        # If the best match score is above 0.6, it's a success!
        if highest > 0.6:
            ui.success(f"{who} Detected Succesfully.  Similarity: {highest:.2f}")
        else:
            ui.error(f"Wrong face. Similarity: {highest:.2f}")
```


## 5. Conclusion

This Face Lock application is an good example of how modern tools can be combined to build a face recognition system. It demonstrates the entire pipeline from capturing and verifying, making it a great project for learning about computer vision and machine learning applications.
