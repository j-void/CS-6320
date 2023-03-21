import numpy as np
import math
from least_squares_fundamental_matrix import solve_F
import two_view_data
import fundamental_matrix


def calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct):
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    
    num_samples = math.log(1-prob_success) / math.log(1-(ind_prob_correct**sample_size))
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return int(num_samples)


def find_inliers(x_0s, F, x_1s, threshold):
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    if x_0s.shape[1] == 2:
        ones = np.ones((x_0s.shape[0], 1))
        x_0s = np.concatenate((x_0s, ones), axis = 1)
    if x_1s.shape[1] == 2:  
        ones = np.ones((x_1s.shape[0], 1))
        x_1s = np.concatenate((x_1s, ones), axis = 1)
        
    inliers = []

    for i in range(len(x_0s)):
        e1 = fundamental_matrix.point_line_distance(F @ x_1s[i], x_0s[i])
        e2 = fundamental_matrix.point_line_distance(F.T @ x_0s[i], x_1s[i])
        if abs((e1+e2)/2) <= threshold and abs(e1) <= threshold and abs(e2) <= threshold:
            inliers.append(i)
    inliers = np.array(inliers)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return inliers


def ransac_fundamental_matrix(x_0s, x_1s):
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    prob_success = 0.999
    sample_size = 9 #int(x_0s.shape[0]*0.05)
    ind_prob_correct = 0.9
    n = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)
    print(f'P={prob_success}, k={sample_size}, p={ind_prob_correct} and N={n}')
    
    best_F = np.eye(3)
    inliers_x_0 = np.zeros((x_0s.shape))
    inliers_x_1 = np.zeros((x_1s.shape))
    best_inlier_count = -1
    for i in range(n):
        indices = np.random.choice(x_0s.shape[0], sample_size, replace = False)
        x_0, x_1 = x_0s[indices], x_1s[indices]
        F = solve_F(x_0s=x_0, x_1s=x_1)
        inliers = find_inliers(x_0s=x_0, F=F, x_1s=x_1, threshold=2)
        if len(inliers) > best_inlier_count:
            best_F = F.copy()
            inliers_x_0 = x_0[inliers]
            inliers_x_1 = x_1[inliers]
            best_inlier_count = len(inliers)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return best_F, inliers_x_0, inliers_x_1


def test_with_epipolar_lines():
    """Unit test you will create for your RANSAC implementation.

    It should take no arguments and it does not need to return anything,
    but it **must** display the images when run.

    Use the code in the jupyter notebook as an example for how to open the
    image files and perform the necessary operations on them in our workflow.
    Remember the steps are Harris, SIFT, match features, RANSAC fundamental matrix.

    Display the proposed correspondences, the true inlier correspondences
    found by RANSAC, and most importantly the epipolar lines in both of your images.
    It should be clear that the epipolar lines intersect where the second image
    was taken, and the true point correspondences should indeed be good matches.

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################

    from feature_matching.SIFTNet import get_siftnet_features
    from feature_matching.utils import load_image, PIL_resize, rgb2gray
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # Rushmore
    image1 = load_image('../data/kit1.jpg')
    image2 = load_image('../data/kit3.jpg')

    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1]*scale_factor), int(image1.shape[0]*scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1]*scale_factor), int(image2.shape[0]*scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    #convert images to tensor
    tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    to_tensor = transforms.ToTensor()
    image_input1 = to_tensor(image1_bw).unsqueeze(0)
    image_input2 = to_tensor(image2_bw).unsqueeze(0)

    from feature_matching.HarrisNet import get_interest_points
    from feature_matching.utils import show_interest_points
    x1, y1, _ = get_interest_points(image_input1.float())
    x2, y2, _ = get_interest_points(image_input2.float())

    x1, x2 = x1.detach().numpy(), x2.detach().numpy()
    y1, y2 = y1.detach().numpy(), y2.detach().numpy()
    print('{:d} corners in image 1, {:d} corners in image 2'.format(len(x1), len(x2)))
    image1_features = get_siftnet_features(image_input1, x1, y1)
    image2_features = get_siftnet_features(image_input2, x2, y2)

    from feature_matching.student_feature_matching import match_features
    matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2)
    print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))

    from feature_matching.utils import show_correspondence_circles, show_correspondence_lines
    # num_pts_to_visualize = len(matches)
    num_pts_to_visualize = 100
    c2 = show_correspondence_lines(image1, image2,
                        x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                        x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
    plt.figure(); plt.title('Proposed Matches'); plt.imshow(c2)

    from ransac import ransac_fundamental_matrix
    # print(image1_features.shape, image2_features.shape)
    num_features = min([len(image1_features), len(image2_features)])
    x0s = np.zeros((len(matches), 2))
    x1s = np.zeros((len(matches), 2))
    x0s[:,0] = x1[matches[:, 0]]
    x0s[:,1] = y1[matches[:, 0]]
    x1s[:,0] = x2[matches[:, 1]]
    x1s[:,1] = y2[matches[:, 1]]
    # print(image1_pts.shape)
    F, matches_x0, matches_x1 = ransac_fundamental_matrix(x0s, x1s)
    print(F)
    # print(matches_x0)
    # print(matches_x1)

    from utils import draw_epipolar_lines
    # Draw the epipolar lines on the images and corresponding matches
    match_image = show_correspondence_lines(image1, image2,
                                    matches_x0[:num_pts_to_visualize, 0], matches_x0[:num_pts_to_visualize, 1],
                                    matches_x1[:num_pts_to_visualize, 0], matches_x1[:num_pts_to_visualize, 1])
    plt.figure(); plt.title('True Matches'); plt.imshow(match_image)
    draw_epipolar_lines(F, image1, image2, matches_x0, matches_x1)

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
