# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2022
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Face Recognition
# 02. Module to enhance face samples, aiming at further feature extraction.
# 2nd version, based on Dlib (http://dlib.net/).
# Language: Python 3
# Needed libraries: NumPy (https://numpy.org/), OpenCV (https://opencv.org/), abd Dlib (http://dlib.net/).
# A local CMake installation is also necessary for DLib.
# Quick install (with PyPI - https://pypi.org/): execute, on command shell (each line at a time):
# "pip3 install numpy";
# "pip3 install opencv-contrib-python";
# "pip3 install dlib" (CMake is needed).

import math
import cv2
import numpy
import dlib

# Configuration parameters.
# Width of the acquired image depicting a face after normalization; the original aspect ratio is kept.
IMAGE_WIDTH = 640

# Width and height of the extracted face after normalization.
FACE_SIZE = 256

# Dlib CNN face and face shape (68 points) detectors.
DLIB_FACE_DETECTOR = dlib.cnn_face_detection_model_v1('data/dlib_face_detector.dat')
DLIB_LANDMARK_DETECTOR = dlib.shape_predictor('data/dlib_shape_predictor_68_landmarks.dat')


# Rotates the given <image> and reference face rectangle <face_rect> CCW obeying the given <rad_angle>.
# Returns the rotated image and new face rectangle (x, y, w, h).
def __rotate_face(image, face_rect, rad_angle):
    # rotation matrix
    sine = math.sin(rad_angle)
    cosine = math.cos(rad_angle)

    rot_mat = numpy.zeros((3, 3))
    rot_mat[0, 0] = cosine
    rot_mat[0, 1] = -sine
    rot_mat[1, 0] = sine
    rot_mat[1, 1] = cosine
    rot_mat[2, 2] = 1.0

    # rotates the image borders
    rot_border = numpy.array([(0, 0), (0, image.shape[0]), (image.shape[1], 0), (image.shape[1], image.shape[0])])
    rot_border = cv2.perspectiveTransform(numpy.float32([rot_border]), rot_mat)[0]
    rot_w = int(round(numpy.max(rot_border[:, 0]) - numpy.min(rot_border[:, 0])))
    rot_h = int(round(numpy.max(rot_border[:, 1]) - numpy.min(rot_border[:, 1])))

    # translation added to the rotation matrix to compensate for negative points
    rot_mat[0, 2] = rot_mat[0, 2] - numpy.min(rot_border[:, 0])
    rot_mat[1, 2] = rot_mat[1, 2] - numpy.min(rot_border[:, 1])

    # rotates the given image
    rot_image = cv2.warpPerspective(image, rot_mat, (rot_w, rot_h))

    # rotates the given face rectangle
    x, y, w, h = face_rect
    rot_face_rect = numpy.array([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
    rot_face_rect = cv2.perspectiveTransform(numpy.float32([rot_face_rect]), rot_mat)[0]

    # computes a new non-rotated face rectangle containing the rotated one
    new_face_rect = numpy.min(rot_face_rect[:, 0]), numpy.min(rot_face_rect[:, 1]), numpy.max(
        rot_face_rect[:, 0]), numpy.max(rot_face_rect[:, 1])

    new_face_rect = int(round(new_face_rect[0])), int(round(new_face_rect[1])), int(
        round(new_face_rect[2] - new_face_rect[0])), int(round(new_face_rect[3] - new_face_rect[1]))

    # returns the rotated image and new face rectangle
    return rot_image, new_face_rect


# Preprocesses the given <image> for further face detection.
# Provide <view> as True if you want to see the result of computations. Will ask and wait for key press.
# Returns the preprocessed image.
def _01_preprocess(image, view=False):
    # makes the image grayscale, if it is still colored
    if len(image.shape) > 2 and image.shape[2] > 1:  # more than one channel?
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resizes the image to present a width of IMAGE_WIDTH pixels, keeping original aspect ratio
    aspect_ratio = float(image.shape[1]) / image.shape[0]
    height = int(round(IMAGE_WIDTH / aspect_ratio))
    image = cv2.resize(image, (IMAGE_WIDTH, height))

    # shows the obtained image, if it is the case
    if view:
        cv2.imshow('Preprocessing, press any key.', image)
        cv2.waitKey(0)

    print('[INFO] Preprocessed image.')
    return image


# Detects the largest face over the given gray-scaled image.
# Provide <view> as True if you want to see the result of computations. Will ask and wait for key press.
# Returns the rectangle (x, y, w, h) of the detected face, as well as the Dlib face_rect object
def _02_detect_face(gs_image, view=False):
    # detects faces on the given image with the CNN detector
    faces_cnn = DLIB_FACE_DETECTOR(gs_image, 1)

    # if there are no faces, returns None
    if len(faces_cnn) == 0:
        return None

    # else...
    # takes the largest face among the detected ones
    # TODO detecting more faces can be added here
    x, y, w, h = 0, 0, 0, 0
    face_rect = None
    size = 0

    for face in faces_cnn:
        ix = face.rect.left()
        iy = face.rect.top()
        iw = face.rect.right() - x
        ih = face.rect.bottom() - y

        if iw * ih > size:
            x, y, w, h = ix, iy, iw, ih
            face_rect = face
            size = iw * ih

    # show the obtained face, if it is the case
    if view:
        view_image = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(view_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Detected face, press any key.', view_image)
        cv2.waitKey(0)

    print('[INFO] Detected face.')
    return x, y, w, h, face_rect


# Transforms the given gray-scaled image <gs_image> according to the given face position <face_rect>,
# and Dlib face rectangle <dlib_face_rect>.
# making it up-face aligned with the horizontal line.
# Provide <view> as True if you want to see the result of computations. Will ask and wait for key press.
# Returns the transformed image and new rectangle (x, y, w, h) of the aligned face.
def _03_align_face(gs_image, face_rect, dlib_face_rect, view=False):
    # detects the facial landmarks using Dlib
    shape = DLIB_LANDMARK_DETECTOR(gs_image, dlib_face_rect.rect)

    # if a shape was detected
    if shape is not None:
        np_shape = numpy.zeros((68, 2), int)
        for i in range(68):
            np_shape[i] = (shape.part(i).x, shape.part(i).y)

        # eye centers
        xc1 = int(round((np_shape[36, 0] + np_shape[40, 0]) / 2.0))
        yc1 = int(round((np_shape[36, 1] + np_shape[40, 1]) / 2.0))

        # left eye, mirrored on the right42 45
        xc2 = int(round((np_shape[42, 0] + np_shape[45, 0]) / 2.0))
        yc2 = int(round((np_shape[42, 1] + np_shape[45, 1]) / 2.0))

        # show the landmarks, if is the case
        if view:
            view_image = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
            for (x, y) in np_shape:
                cv2.circle(view_image, (x, y), 3, (0, 255, 0), -1)

            cv2.circle(view_image, (xc1, yc1), 3, (0, 0, 255), -1)
            cv2.circle(view_image, (xc2, yc2), 3, (0, 0, 255), -1)

            cv2.imshow('Facial landmarks, press any key.', view_image)
            cv2.waitKey(0)

        # angle between eyes
        angle = math.atan2(yc2 - yc1, xc2 - xc1)

        # obtains the aligned image and new face rectangle
        gs_image, face_rect = __rotate_face(gs_image, face_rect, -angle)

    # shows the aligned face, if it is the case
    if view:
        x, y, w, h = face_rect
        view_image = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(view_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Aligned face, press any key.', view_image)
        cv2.waitKey(0)

    print('[INFO] Aligned the face.')
    return gs_image, face_rect


# Crops and normalizes the face contained in the given gray-scaled image <gs_image>,
# following the provided face rectangle <face_rect>.
# Provide <view> as True if you want to see the result of computations. Will ask and wait for key press.
# Returns the extracted face, ready for description (feature extraction).
def _04_extract_face(gs_image, face_rect, view=False):
    x, y, w, h = face_rect
    cx = int(round(x + w / 2.0))
    cy = int(round(y + h / 2.0))
    r = int(round(max(w, h) / 2.0))

    face_image = gs_image[
                 max(0, cy - r):min(cy + r + 1, gs_image.shape[0]),
                 max(0, cx - r):min(cx + r + 1, gs_image.shape[1])]  # squared face
    if len(face_image) > 0:
        face_image = cv2.resize(face_image, (FACE_SIZE, FACE_SIZE))  # face in normalized size
        face_image = cv2.equalizeHist(face_image)  # color histogram normalization
    else:
        return None

    if view:
        cv2.imshow('[INFO] Normalized face, press any key.', face_image)
        cv2.waitKey(0)

    print('[INFO] Extracted the face.')
    return face_image


# Enhances the given image, returning the normalized version of the largest face depicted within it.
# Provide <view> as True if you want to see the results of computations. Will ask and wait for many key presses.
# Returns the normalized face image, useful for description (feature extraction), or None, if no face was detected.
def enhance(image, view=False):
    # pre-processes the given image
    pp_image = _01_preprocess(image, view=view)

    # detects a face in the given image
    face_rect = _02_detect_face(pp_image, view=view)

    if face_rect is not None:
        # aligns the obtained face
        aligned_image, aligned_face = _03_align_face(pp_image, face_rect[0:4], face_rect[4], view=view)

        # extracts and returns the detected face
        return _04_extract_face(aligned_image, aligned_face, view=view)

    # no face was found
    return None
