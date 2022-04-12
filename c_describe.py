# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2022
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Face Recognition
# 03. Module to describe faces.
# Language: Python 3
# Needed library: ArcFace (https://github.com/mobilesec/arcface-tensorflowlite).
# Quick install (with PyPI - https://pypi.org/): execute, on command shell:
# "pip3 install arcface".

from arcface import ArcFace

global FACE_DESCRIPTOR
FACE_DESCRIPTOR = None


# Describes the given normalized face image <norm_face> with ArcFace.
# Returns the obtained feature vector.
def describe(norm_face):
    global FACE_DESCRIPTOR
    if FACE_DESCRIPTOR is None:
        FACE_DESCRIPTOR = ArcFace.ArcFace()

    feature_vector = FACE_DESCRIPTOR.calc_emb(norm_face)
    print('[INFO] Described face with feature vector of size:', feature_vector.shape)
    return FACE_DESCRIPTOR.calc_emb(norm_face)
