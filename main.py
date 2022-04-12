# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2022
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Face Recognition
# Language: Python 3
# Needed libraries: NumPy (https://numpy.org/), OpenCV (https://opencv.org/), Astropy (https://www.astropy.org/),
# and ArcFace (https://github.com/mobilesec/arcface-tensorflowlite).
# Quick install (with PyPI - https://pypi.org/): execute, on command shell (each line at a time):
# "pip3 install numpy";
# "pip3 install opencv-contrib-python";
# "pip3 install astropy";
# "pip3 install arcface".

import a_acquire
import b_enhance
import b_enhance_2
import c_describe
import d_match
import os
import itertools
import random
import metrics
import cv2
import numpy

# Some hardcoded global variables that will come handy later :)
QUERY_PATH = 'data/queries/'
DATASET_PATH = 'data/dataset/'
VIEW = False
ALGO_TYPE = 'vj'
SUBJECTS = ['subject0' + str(j) for j in range(1, 10)] + ['subject10']

# I did some math to figure this out and then confirmed it in code
# Math is: There are 10 different images for 9 subjects. Subject 1 has 9 images
# Total Pairs for n items = 1 + 2 + ... + (n-1) = (n*(n-1))/2 = (10*9)/2 = 45
# Subject 1 has 9 items --> (9*8)/2 = 36
# Total Possible Pairs = 36 + (9 * 45) = 441
TOTAL_GENUINE_PAIRS = 441


def test_run():
    # Test script.
    img1 = a_acquire.acquire_from_file('data/test_data/face_1_1.jpg', view=VIEW)
    face1 = b_enhance.enhance(img1, view=VIEW)
    # face1 = b_enhance_2.enhance(img1, view=VIEW) #CNN face detector instead of classical Viola-Jones
    desc_1 = c_describe.describe(face1)

    img2 = a_acquire.acquire_from_file('data/test_data/face_1_2.jpg', view=VIEW)
    face2 = b_enhance.enhance(img2, view=VIEW)
    # face2 = b_enhance_2.enhance(img2, view=VIEW) #CNN face detector instead of classical Viola-Jones
    desc_2 = c_describe.describe(face2)

    img3 = a_acquire.acquire_from_file('data/test_data/face_2_1.jpg', view=VIEW)
    face3 = b_enhance.enhance(img3, view=VIEW)
    # face3 = b_enhance_2.enhance(img3, view=VIEW) #CNN face detector instead of classical Viola-Jones
    desc_3 = c_describe.describe(face3)

    img4 = a_acquire.acquire_from_file('data/test_data/face_2_2.jpg', view=VIEW)
    face4 = b_enhance.enhance(img4, view=VIEW)
    # face4 = b_enhance_2.enhance(img4, view=VIEW) #CNN face detector instead of classical Viola-Jones
    desc_4 = c_describe.describe(face4)

    img5 = a_acquire.acquire_from_file('data/test_data/face_3_1.jpg', view=VIEW)
    face5 = b_enhance.enhance(img5, view=VIEW)
    # face5 = b_enhance_2.enhance(img5, view=VIEW) #CNN face detector instead of classical Viola-Jones
    desc_5 = c_describe.describe(face5)

    img6 = a_acquire.acquire_from_file('data/test_data/face_3_2.jpg', view=VIEW)
    face6 = b_enhance.enhance(img6, view=VIEW)
    # face6 = b_enhance_2.enhance(img6, view=VIEW) #CNN face detector instead of classical Viola-Jones
    desc_6 = c_describe.describe(face6)

    dist_12 = d_match.match(desc_1, desc_2)
    dist_34 = d_match.match(desc_3, desc_4)
    dist_56 = d_match.match(desc_5, desc_6)
    print('')

    dist_13 = d_match.match(desc_1, desc_3)
    dist_24 = d_match.match(desc_2, desc_4)
    dist_15 = d_match.match(desc_1, desc_5)
    dist_26 = d_match.match(desc_2, desc_6)
    dist_35 = d_match.match(desc_3, desc_5)
    dist_46 = d_match.match(desc_4, desc_6)

    return None


# This function just calls the different enhance functions based on what argument you'd like to use
def enhance_face(img):
    if ALGO_TYPE not in ('vj', 'cnn'):
        raise ValueError('Algorithm used for enhancement must be either \'vj\' or \'cnn\'')
    elif ALGO_TYPE == 'vj':
        return b_enhance.enhance(img)
    elif ALGO_TYPE == 'cnn':
        return b_enhance_2.enhance(img)


# This function loads the filenames from the data directory (the endswith check is to make sure we only get images)
def load_filenames(dir_name):
    dir_contents = os.listdir(dir_name)
    return sorted([filename for filename in dir_contents if filename.endswith('png')])


# Generates every possible genuine pair from the given dataset of 99 images
# Returns a list of (file1, file2) tuples, with each tuple being unique and containing a genuine pair
def generate_genuine_pairs():
    filenames = load_filenames('data/dataset')
    genuine_pairs = []

    for subject_id in SUBJECTS:
        # Get the other files that have the same subject (i.e. different photos of the same person)
        subject_files = [file for file in filenames if file.split('.')[0].startswith(subject_id)]

        # Get all unique pairs of images for this particular subject
        # Returns a list of (file1, file2) tuples
        subject_pairs = list(itertools.combinations(subject_files, 2))

        # Add this particular subject's pairs to the overall list of genuine pairs
        genuine_pairs += subject_pairs

    # Returns a list of tuples containing all genuine pairs from this particular dataset
    return genuine_pairs


def generate_impostor_pairs():
    filenames = load_filenames('data/dataset')
    impostor_pairs = []

    # We're going to generate the same number of impostor pairs as there are genuine pairs (441)
    i = 0
    while i < TOTAL_GENUINE_PAIRS:
        # Generate two random indices
        idx1, idx2 = random.randint(0, len(filenames)-1), random.randint(0, len(filenames)-1)
        # Grab the files at those indices
        file1, file2 = filenames[idx1], filenames[idx2]

        # If the subjects are not the same (i.e. they're different people)...
        if file1.split('.')[0] != file2.split('.')[0]:
            # ... make the pair into a set. It's important we make it a set so that the "in" comparison works
            pair = {file1, file2}

            # If the current pair isn't already in the list...
            if pair not in impostor_pairs:
                # ... append it and add to the counter
                impostor_pairs.append(pair)
                i += 1

    return impostor_pairs


# I only ran this function a few times to get the files in place for reading later and so this system could be
# ... replicated. Running this will get new impostor pairs so only do so if you need to generate new pairs
# I essentially created these files such that we can see where each score is coming from in the results file later
def create_files():
    yes = input("Creating new impostor_pairs.csv that will overwrite existing ones. Proceed? (y/n) ").lower().rstrip()
    if yes == 'y':
        with open('data/genuine_pairs.csv', 'w') as f:
            gen_pairs = generate_genuine_pairs()
            f.write('# filename1, filename2\n')
            for file1, file2 in gen_pairs:
                f.write(f'{file1},{file2}\n')

        with open('data/impostor_pairs.csv', 'w') as f:
            imp_pairs = generate_impostor_pairs()
            f.write('# filename1, filename2\n')
            for file1, file2 in imp_pairs:
                f.write(f'{file1},{file2}\n')
        return True
    else:
        return False


def read_file(filename):
    pairs = []
    with open(filename, 'r') as f:
        f.readline()  # Skip header
        pairs += [tuple(line.rstrip().split(',')) for line in f.readlines()]
    return pairs


def get_distance(file1, file2):
    path1, path2 = DATASET_PATH + file1, DATASET_PATH + file2
    print(path1, path2)
    img1, img2 = a_acquire.acquire_from_file(path1), a_acquire.acquire_from_file(path2)
    face1, face2 = enhance_face(img1), enhance_face(img2)
    desc1, desc2 = c_describe.describe(face1), c_describe.describe(face2)
    dist = d_match.match(desc1, desc2)
    return dist


# This function actually runs pretty fast for how much stuff goes on in it
# Basically, every pair is iterated over, has the images read in, enhanced, and described, and then a distance
# is calculated and stored in a CSV file along with a label
def calculate_distances(genuine_pairs, impostor_pairs):
    with open('data/results.csv', 'w') as f:
        i = 0
        f.write('# [0:impostor 1:genuine], distance\n')
        for file1, file2 in genuine_pairs:
            print(f"Comparison #{i + 1}")
            dist = get_distance(file1, file2)
            f.write(f'1,{dist}\n')
            i += 1
        for file1, file2 in impostor_pairs:
            print(f"Comparison #{i + 1}")
            dist = get_distance(file1, file2)
            f.write(f'0,{dist}\n')
            i += 1


def get_metrics(filename):
    # Load the results
    observations = metrics.load_data(filename)

    # Get the ideal threshold and other metrics
    fnmr, fmr, eer = metrics.compute_fmr_fnmr_eer(observations, 'distance')
    print(fnmr, fmr)

    # Get the AUC and plot the ROC curve
    auc, fmr, tmr = metrics.compute_fmr_tmr_auc(observations, 'distance')
    metrics.plot_all_fmr_tmr_auc(fmr, tmr, auc)

    return eer, auc


def capture_face():
    while True:
        img1 = a_acquire.acquire_from_camera(cam_id=0, view=False)
        face1 = enhance_face(img1)

        view_image = img1.copy()
        if face1 is not None:
            view_image = numpy.zeros((max(img1.shape[0], face1.shape[0]), img1.shape[1] + face1.shape[1], 3),
                                     numpy.uint8)
            view_image[:img1.shape[0], :img1.shape[1], :] = img1[:, :, :]
            view_image[:face1.shape[0], img1.shape[1]:, :] = cv2.cvtColor(face1, cv2.COLOR_GRAY2BGR)[:, :, :]

            cv2.imshow('Detected face 1, press any key', view_image)
            key = cv2.waitKey(1)
            if key != -1:
                break

    return face1


def find_similar_face():
    # Capture my own face and get a description
    my_face = capture_face()
    my_face_desc = c_describe.describe(my_face)

    # Find the file/face with the smallest distance when compared to my own
    min_dist = 1000
    best_img = None
    for filename in load_filenames('data/dataset'):
        img = a_acquire.acquire_from_file(DATASET_PATH + filename)
        other_face = enhance_face(img)
        other_face_desc = c_describe.describe(other_face)
        dist = d_match.match(my_face_desc, other_face_desc)

        # If this new distance is less than the previously found minimum, indicate that this file is the best match
        if dist < min_dist:
            min_dist = dist
            best_img = filename

    return min_dist, best_img


def find_query_results():
    query_files = load_filenames('data/queries')
    results = {f: {'min_dist': 100, 'best_file': ''} for f in query_files}
    for query_file in query_files:
        query_img = a_acquire.acquire_from_file(QUERY_PATH + query_file)
        query_face = enhance_face(query_img)
        query_desc = c_describe.describe(query_face)
        for filename in load_filenames('data/dataset'):
            img = a_acquire.acquire_from_file(DATASET_PATH + filename)
            face = enhance_face(img)
            desc = c_describe.describe(face)
            dist = d_match.match(desc, query_desc)

            # During testing, I noticed that pictures of the same people had scores of 0, indicating that we're
            # comparing a photo to itself. If this happens we can break this loop because we know who this photo is of
            if dist < results[query_file]['min_dist']:
                results[query_file]['min_dist'] = dist
                results[query_file]['best_file'] = filename
                if dist == 0:
                    break

    return results


if __name__ == '__main__':
    # Get the metrics of the system (also plots ROC curve)
    eer, auc = get_metrics('data/results.csv')

    # Find the best-matching face to my own and display it
    best_dist, best_match = find_similar_face()
    a_acquire.acquire_from_file(DATASET_PATH + best_match, view=True)

    # Determine who the images in the query sets are (if they are in the system)
    query_results = find_query_results()

    print(f"QUESTION 2.1: EER = {eer}")
    print(f"QUESTION 2.2: AUC = {auc}")
    print(f'QUESTION 2.3: Closest Face = {best_match}; Distance = {best_dist}')
    print(f'QUESTION 2.4:')
    for q_file in query_results.keys():
        if query_results[q_file]["min_dist"] > eer:
            print(f'\tQuery File: {q_file}. NO MATCH. Minimum Distance is {query_results[q_file]["min_dist"]}:.2f')
        else:
            print(f'\tqQuery File: {q_file}. Best Match: {query_results[q_file]["best_file"]}. Distance: {query_results[q_file]["min_dist"]}')
