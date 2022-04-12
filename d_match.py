# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2022
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Face Recognition
# 04. Module to match face descriptions.
# Language: Python 3
# Language: Python 3
# Needed library: ArcFace (https://github.com/mobilesec/arcface-tensorflowlite).
# Quick install (with PyPI - https://pypi.org/): execute, on command shell:
# "pip3 install arcface".

from arcface import ArcFace

global FACE_DESCRIPTOR
FACE_DESCRIPTOR = None


# Matches the given feature vectors <description_1> and <description_2>.
# Returns the distance between them, expressing how likely they are of representing the same face.
# The distance is a positive real number.
def match(description_1, description_2):
    global FACE_DESCRIPTOR
    if FACE_DESCRIPTOR is None:
        FACE_DESCRIPTOR = ArcFace.ArcFace()

    distance = FACE_DESCRIPTOR.get_distance_embeddings(description_1, description_2)
    print('[INFO] Computed distance:', distance)

    return distance
