# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2022
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Face Recognition
# 01. Acquisition-module stub.
# Language: Python 3
# Needed libraries: OpenCV (https://opencv.org/)
# Quick install (with PyPI - https://pypi.org/):
# execute "pip3 install opencv-python" on command shell.

import cv2

# Connected camera configuration.
global CAM_ID
global CAM
CAM_ID = -1


# Function to acquire an image from the available and connected camera.
# The image is acquired with three color channels (BGR).
# Parameters
# cam_id: ID (0 or above) of the connected camera to be used; single camera is usually ZERO.
# view: TRUE if loaded image must be shown in a proper window, FALSE otherwise.
def acquire_from_camera(cam_id=0, view=False):
    global CAM_ID
    global CAM

    if cam_id != CAM_ID:
        CAM_ID = cam_id
        CAM = cv2.VideoCapture(CAM_ID)

    while True:
        _, frame = CAM.read()

        if not view:
            return frame

        else:
            cv2.imshow('Press any key to capture image.', frame)
            key = cv2.waitKey(1)
            if key != -1:
                return frame


# Stub function to acquire an image that might contain a face from the given file path.
# The image is acquired with three color channels (BGR).
# Parameters
# file_path: The path to the image file containing the face.
# view: TRUE if loaded image must be shown in a proper window, FALSE otherwise.
def acquire_from_file(file_path, view=False):
    # reads the image from the given file path
    # and returns it
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # shows the read image, if it is the case
    if view:
        cv2.imshow('Press any key.', image)
        cv2.waitKey(0)

    print('[INFO] Acquired image from file.')
    return image
