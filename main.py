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
SUBJECTS = ['subject0' + str(j) for j in range(1, 10)] + ['subject10']

# I did some math to figure this out and then confirmed it in code
# Math is: There are 10 different images for 9 subjects. Subject 1 has 9 images
# Total Pairs for n items = 1 + 2 + ... + (n-1) = (n*(n-1))/2 = (10*9)/2 = 45
# Subject 1 has 9 items --> (9*8)/2 = 36
# Total Possible Pairs = 36 + (9 * 45) = 441
TOTAL_GENUINE_PAIRS = 441


# This function just calls the different enhance functions based on what argument you'd like to use
def enhance_face(img, algo_type):
    if algo_type not in ('vj', 'cnn'):
        raise ValueError('Algorithm used for enhancement must be either \'vj\' or \'cnn\'')
    elif algo_type == 'vj':
        return b_enhance.enhance(img)
    elif algo_type == 'cnn':
        return b_enhance_2.enhance(img)


# This function loads the filenames from the data directory (the endswith check is to make sure we only get images)
def load_filenames(dir_name):
    dir_contents = os.listdir(dir_name)
    return sorted([filename for filename in dir_contents if filename.endswith('png')])


# Generates every possible genuine pair from the given dataset of 99 images
# Returns a list of (file1, file2) tuples, with each tuple being unique and containing a genuine pair
def generate_pairs():
    filenames = load_filenames('data/dataset')
    gen_pairs = []
    imp_pairs = []

    for subject_id in SUBJECTS:
        # Get the other files that have the same subject (i.e. different photos of the same person)
        subject_files = [file for file in filenames if file.split('.')[0].startswith(subject_id)]

        # Get all unique pairs of images for this particular subject
        # Returns a list of (file1, file2) tuples
        subject_pairs = list(itertools.combinations(subject_files, 2))

        # Add this particular subject's pairs to the overall list of genuine pairs
        gen_pairs += subject_pairs

    # Returns a list of tuples containing all genuine pairs from this particular dataset
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
            if pair not in imp_pairs:
                # ... append it and add to the counter
                imp_pairs.append(pair)
                i += 1

    return gen_pairs, imp_pairs


# ONE-OFF FUNCTION
# I only ran this function a few times to get the files in place for reading later and so this system could be ...
# ... replicated. Running this will get new impostor pairs so only do so if you need to generate new pairs.
# I essentially created these files such that we can see where each score is coming from in the results file later
def create_pairs_files():
    gen_pairs, imp_pairs = generate_pairs()
    yes = input("Creating new impostor_pairs.csv that will overwrite existing ones. Proceed? (y/n) ").lower().rstrip()
    if yes == 'y':
        with open('data/genuine_pairs.csv', 'w') as f:
            f.write('# filename1, filename2\n')
            for file1, file2 in gen_pairs:
                f.write(f'{file1},{file2}\n')

        with open('data/impostor_pairs.csv', 'w') as f:
            f.write('# filename1, filename2\n')
            for file1, file2 in imp_pairs:
                f.write(f'{file1},{file2}\n')
        return True
    else:
        return False


def read_pairs_files(genuine_pairs_file, impostor_pairs_file):
    gen_pairs = []
    imp_pairs = []

    with open(genuine_pairs_file, 'r') as f:
        f.readline()
        gen_pairs += [line.rstrip().split(',') for line in f.readlines()]
    with open(impostor_pairs_file, 'r') as f:
        f.readline()
        imp_pairs += [line.rstrip().split(',') for line in f.readlines()]

    return gen_pairs, imp_pairs


# Finds the angular distance between two different faces given the filepaths to their images
def get_distance(file1, file2, algo_type):
    path1, path2 = DATASET_PATH + file1, DATASET_PATH + file2
    print(path1, path2)
    img1, img2 = a_acquire.acquire_from_file(path1), a_acquire.acquire_from_file(path2)
    face1, face2 = enhance_face(img1, algo_type), enhance_face(img2, algo_type)
    desc1, desc2 = c_describe.describe(face1), c_describe.describe(face2)
    dist = d_match.match(desc1, desc2)
    return dist


# This also is a one-off function. The results are stored in data/results_{algo_type}.csv
# This function actually runs pretty fast for how much stuff goes on in it
# Basically, every pair is iterated over, has the images read in, enhanced, and described, and then a distance
# is calculated and stored in a CSV file along with a label (1 for genuine, 0 for impostor
def calculate_distances(gen_pairs, imp_pairs, algo_type):
    dists = []
    with open(f'data/results_{algo_type}.csv', 'w') as f:
        i = 0
        f.write('# [0:impostor 1:genuine], distance\n')
        for file1, file2 in gen_pairs:
            print(f"Comparison #{i + 1}")
            dist = get_distance(file1, file2, algo_type)
            f.write(f'1,{dist}\n')
            dists.append((1, dist))
            i += 1
        for file1, file2 in imp_pairs:
            print(f"Comparison #{i + 1}")
            dist = get_distance(file1, file2, algo_type)
            f.write(f'0,{dist}\n')
            dists.append((0, dist))
            i += 1

    return dists


# Uses the metrics.py methods to return the EER and AUC of a given system, and prints them to the user
# Also will plot the ROC curve of the system
def get_metrics(results_filename):
    # Load the results
    observations = metrics.load_data(results_filename)

    # Get the ideal threshold and FNMR and FMR
    fnmr, fmr, eer = metrics.compute_fmr_fnmr_eer(observations, 'distance')
    print(fnmr, fmr)

    # Get the AUC and plot the ROC curve
    auc, fmr, tmr = metrics.compute_fmr_tmr_auc(observations, 'distance')
    metrics.plot_all_fmr_tmr_auc(fmr, tmr, auc)

    return eer, auc


# Get an image of my own face for comparison to other faces in the dataset
def capture_face(algo_type):
    while True:
        img1 = a_acquire.acquire_from_camera(cam_id=0, view=False)
        face1 = enhance_face(img1, algo_type)

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


# Get an image of my own face and then iterate over the system to find the face in the dataset that is the closest
# match to me.
# Returns the smallest distance found, and the filename of the image that had that distance
def find_similar_face(algo_type):
    # Capture my own face and get a description
    my_face = capture_face(algo_type)
    my_face_desc = c_describe.describe(my_face)

    # Find the file/face with the smallest distance when compared to my own
    min_dist = 1000
    best_img = None
    for filename in load_filenames('data/dataset'):
        img = a_acquire.acquire_from_file(DATASET_PATH + filename)
        other_face = enhance_face(img, algo_type)
        other_face_desc = c_describe.describe(other_face)
        dist = d_match.match(my_face_desc, other_face_desc)

        # If this new distance is less than the previously found minimum, indicate that this file is the best match
        if dist < min_dist:
            min_dist = dist
            best_img = filename

    return min_dist, best_img


# This function goes through each of the query files in data/queries and finds the best match for each of them
# "Best Match" here means the subject who has the image with the smallest distance in the entire dataset when compared
# with the query
def find_query_results(algo_type):
    query_files = load_filenames('data/queries')
    results = {f: {'min_dist': 100, 'best_file': ''} for f in query_files}
    for query_file in query_files:
        query_img = a_acquire.acquire_from_file(QUERY_PATH + query_file)
        query_face = enhance_face(query_img, algo_type)
        query_desc = c_describe.describe(query_face)
        for filename in load_filenames('data/dataset'):
            img = a_acquire.acquire_from_file(DATASET_PATH + filename)
            face = enhance_face(img, algo_type)
            desc = c_describe.describe(face)
            dist = d_match.match(desc, query_desc)

            # During testing, I noticed that pictures of the same people had scores of 0, indicating that we're
            # comparing a photo to itself. If this happens we can break this loop because we know who this photo is of
            if dist < results[query_file]['min_dist']:
                results[query_file]['min_dist'] = float(dist)
                results[query_file]['best_file'] = filename
                if dist == 0:
                    break

    # Return a dictionary where each query file is a key, and its value is another dictionary containing the minimum
    # distance, and the name of the image file with that distance
    # i.e. { 'example1.png': { 'min_dist': 0.25, 'best_file': 'subject01.glasses.png' }, 'example2.png': ... }
    with open(f'data/query_results_{algo_type}.csv', 'w') as f:
        f.write("#query_file,best_match,distance\n")
        for file in query_files:
            match = results[file]['best_file']
            distance = results[file]['min_dist']
            f.write(f'{file},{match},{distance}\n')

    return results


def read_query_results(algo_type):
    results = {}
    with open(f'data/query_results_{algo_type}.csv', 'r') as f:
        f.readline()
        for line in f.readlines():
            query_filename, match_filename, distance = line.rstrip().split(',')
            results[query_filename] = {'best_file': match_filename, 'min_dist': float(distance)}

    return results


if __name__ == '__main__':
    # Set the algorithm we are going to use
    # If creating a new system, it will use that algorithm to find the distances for each pair
    # If analyzing an existing system, it will examine the results that were generated by the selected algorithm
    ALGO_TYPE = 'vj'  # 'cnn' or 'vj'

    # Running the code below creates a set of genuine and impostor pairs, and then finds the distances for them.
    # It essentially creates a new system to be analyzed
    # WARNING: It will overwrite existing results files and can take a pretty good amount of time to run so be careful
    # create_pairs_files()
    # genuine_pairs, impostor_pairs = read_pairs_files('data/genuine_pairs.csv', 'data/impostor_pairs.csv')
    # distances = calculate_distances(genuine_pairs, impostor_pairs, ALGO_TYPE)

    # Get the metrics of the system (also plots ROC curve)
    eer, auc = get_metrics(f'data/results_{ALGO_TYPE}.csv')

    # Find the best-matching face to my own and display it
    best_dist, best_matching_file = find_similar_face(ALGO_TYPE)
    a_acquire.acquire_from_file(DATASET_PATH + best_matching_file, view=True)

    # Determine who the images in the query sets are (if they are in the system)
    # WARNING: This function takes the longest of any of them at well over 30 mins for the CNN
    # Run the Method below it only to read in results that have already been calculated
    # query_results = find_query_results(ALGO_TYPE)

    # The following method should be run if you want to read in existing results for the queries
    query_results = read_query_results(ALGO_TYPE)
    print(f"QUESTION 2.1: EER = {eer}")
    print(f"QUESTION 2.2: AUC = {auc}")
    print(f'QUESTION 2.3: Closest Face = {best_matching_file}; Distance = {best_dist}')
    print(f'QUESTION 2.4:')

    # Print the results of the different queries. If the minimum distance > threshold, say there's no match
    for q_file in query_results.keys():
        if query_results[q_file]["min_dist"] > eer:
            print(f'\tQuery File: {q_file}. NO MATCH. Minimum Distance is {query_results[q_file]["min_dist"]}')
        else:
            print(f'\tQuery File: {q_file}. Best Match: {query_results[q_file]["best_file"]}. Distance: {query_results[q_file]["min_dist"]}')
