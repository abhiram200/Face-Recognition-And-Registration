from time import sleep
import face_recognition
import cv2
import os
import pyautogui
import numpy as np

def FaceDetection():

    # Load the known images and encode them
    known_images_dir = 'known_images'
    known_encodings = []
    known_names = []
    for image_name in os.listdir(known_images_dir):
        image_path = os.path.join(known_images_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(image_name)[0])

    # Open the camera
    video_capture = cv2.VideoCapture(0)

    # Capture a frame from the camera
    ret, frame = video_capture.read()

    # Check if the input is a still image
    if not is_live(frame):
        # pyautogui.alert('Please provide a live capture instead of a still image!')
        
        # Release the camera and close the window
        video_capture.release()
        cv2.destroyAllWindows()
        
        return False
    
    else:

        # Find all the faces and their encodings in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    
        # Identify the faces in the frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the face encoding to the known encodings
            results = face_recognition.compare_faces(known_encodings, face_encoding)
    
            # Check if any face matches the known faces
            match = None
            for i, result in enumerate(results):
                if result:
                    match = known_names[i]
                    break
    
            # If no known face matches, prompt the user to register a new face
            if match is None:
                
                pyautogui.alert('Unknown face detected! Please register this face.')
                name = pyautogui.prompt('Enter your name:')
                if name is None:
                    continue
    
                # Save the image with the inputted name
                new_image_path = os.path.join(known_images_dir, name + '.jpg')
                cv2.imwrite(new_image_path, frame)
                print(f"New face {name} registered!")
    
                # Update the known encodings and names
                image = face_recognition.load_image_file(new_image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_names.append(name)
    
            # Draw a rectangle and label the face in the frame
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, match, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
    
            # Show the frame
            cv2.imshow('Video', frame)
            
            
            print(match)
            
            sleep(2)
        
        # Release the camera and close the window
        video_capture.release()
        cv2.destroyAllWindows()
        
        return True

def is_live(frame):
    """
    Determines whether the input frame is a live capture or a still image.

    Args:
        frame: A numpy array representing the input frame.

    Returns:
        A boolean indicating whether the input frame is a live capture (True) or a still image (False).
    """
    # Convert the input frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute the Laplacian of the blurred image
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # print(variance)
    
    # If the variance is below a certain threshold, assume it is a still image
    if variance < 0.1:
        return False

    # Otherwise, assume it is a live capture
    else:
        return True

FaceDetection()
