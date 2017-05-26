import math
import sys
import cv2
import numpy as np
from sklearn.decomposition import PCA
from preprocessing import load_radiographs, preprocess_radiograph

def pca(X, nb_components = 0):
    '''
            Do a PCA analysis on X
        Args:
            X: np.array containing the samples
                shape = (nb samples, nb dimensions of each sample)
            nb_components: the nb components we're interested in
        Returns:
            The ``nb_components`` largest evals and evecs of the covariance matrix and
            the average sample.
    '''
    [n, d] = X.shape
    if (nb_components <= 0) or (nb_components > n):
        nb_components = n

    mu = np.average(X, axis=0)
    X -= mu.transpose()

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
    eigenvectors = np.dot(np.transpose(X), eigenvectors)

    eig = zip(eigenvalues, np.transpose(eigenvectors))
    eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                         x[1] / np.linalg.norm(x[1])), eig)

    eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
    eig = eig[:nb_components]

    eigenvalues, eigenvectors = map(np.array, zip(*eig))

    return eigenvalues, np.transpose(eigenvectors), mu
    
    
def normalize_image(img):
    """Normalize an image such that it min=0 , max=255 and type is np.uint8
    """
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)
 
 
def project(W, X, mu):
    """Project X on the space spanned by the vectors in W.
    mu is the average image.
    """
    return np.dot(X - mu.T, W)


def reconstruct(W, Y, mu):
    """Reconstruct an image based on its PCA-coefficients Y, the evecs W
    and the average mu.
    """
    return np.dot(Y, W.T) + mu.T
          

def sliding_window(image, search_region, step_size, window_size):
    """
    Slides a window across a given image.
    Args:
        image: the image that we are going to loop over.
        search_region: the region of the image to search in [(xLT, yLT), (xRB, yRB)]
        step_size: the number of pixels to skip in both the (x,y) direction.
        window_size: (width, height) of the extracted window.
    Yields:
        A tuple containing the x  and y  coordinates of the sliding window,
        along with the window itself.
    """
    for y in range(search_region[0][1], search_region[1][1] - window_size[1], step_size) + [search_region[1][1] - window_size[1]]:
        for x in range(search_region[0][0], search_region[1][0] - window_size[0], step_size) + [search_region[1][0] - window_size[0]]:
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def sharpe_boxes(mean_pca, evecs_pca, image, width, height, is_upper, mean_boxes, show=False):
    """Finds a bounding box around the four upper or lower incisors.
    A sliding window is moved over the given image. The window which matches best
    with the given appearance model is returned.

    Args:
        mean: PCA mean.
        evecs: PCA eigen vectors.
        image: The dental radiograph on which the incisors should be located.
        width (int): The default width of the search window.
        height (int): The default height of the search window.
        is_upper (bool): Wheter to look for the upper (True) or lower (False) incisors.
        jaw_split (Path): The jaw split.

    Returns:
        A bounding box around what looks like four incisors.
        The region of the image selected by the bounding box.
    """
    h, w = image.shape
    
    # [b1, a1]---------------
    # -----------------------
    # -----------------------
    # -----------------------
    # ---------------[b2, a2]
    
    if is_upper:
        b1 = int(w/2 - w/10)
        b2 = int(w/2 + w/10)
        a1 = int(mean_boxes[1]) - 350
        a2 = int(mean_boxes[3]) + 100
    else:
        b1 = int(w/2 - w/12)
        b2 = int(w/2 + w/12)
        a1 = int(mean_boxes[5])
        a2 = int(mean_boxes[7]) + 350
    search_region = [(b1, a1), (b2, a2)]
    
    best_score = float("inf")
    best_score_bbox = [(-1, -1), (-1, -1)]
    best_score_img = np.zeros(image.shape)
    for wscale in np.arange(0.7, 1.3, 0.1):
        for hscale in np.arange(0.7, 1.3, 0.1):
            winW = int(width * wscale)
            winH = int(height * hscale)
            for (x, y, window) in sliding_window(image, search_region, step_size=36, window_size=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                reCut = cv2.resize(window, (width, height))

                X = reCut.flatten()
                Y = project(evecs_pca, X, mean_pca)
                Xacc = reconstruct(evecs_pca, Y, mean_pca)

                score = np.linalg.norm(Xacc - X)
                if score < best_score:
                    best_score = score
                    best_score_bbox = [(x, y), (x + winW, y + winH)]
                    best_score_img = reCut

                #if show:
                #    #window = [(x, y), (x + winW, y + winH)]
                #    #Plotter.plot_autoinit(image, window, score, jaw_split, search_region, best_score_bbox,
                #    #                      title="wscale="+str(wscale)+" hscale="+str(hscale))
                #    img = image.copy()
                #    cv2.rectangle(img, , 1)
                #    cv2.imshow(img)

    return (best_score_bbox, best_score_img)


def fit_template(template, model, img):
    """
        Try to improve the fit of a shape by trying different configurations for
        position, scale and rotation and returning the configuration with the best
        fit for the grey-level model.
    
        Args:
            template (Landmarks): The initial fit of an incisor.
            model (GreyLevelModel): The grey-level model of the incisor.
            img: The dental radiograph on which the shape should be fitted.
    
        Returns:
        Landmarks: The estimated location of the shape.
    """
    gimg = rg.togradient_sobel(img)

    dmin, best = np.inf, None
    for t_x in xrange(-5, 50, 10):
        for t_y in xrange(-50, 50, 10):
            for s in np.arange(0.8, 1.2, 0.1):
                for theta in np.arange(-math.pi/16, math.pi/16, math.pi/16):
                    dists = []
                    X = template.T([t_x, t_y], s, theta)
                    for ind in list(range(15)) + list(range(25,40)):
                        profile = Profile(img, gimg, X, ind, model.k)
                        dist = model.glms[0][ind].quality_of_fit(profile.samples)
                        dists.append(dist)
                    avg_dist = np.mean(np.array(dists))
                    if avg_dist < dmin:
                        dmin = avg_dist
                        best = X

                    Plotter.plot_landmarks_on_image([template, best, X], img, wait=False)

    return best
    

def init_fitting(edges, is_upper):
    """Find an initial estimate for the model in the given image.
    Args:
        model (Landmarks): The shape which should be fitted.
        img: The dental radiograph on which the shape should be fitted.
    Returns:
        Landmarks: An initial estimate for the position of the model in the given image.
    """
    if is_upper:
        width = 380
        height = 315
    else:
        width = 280
        height = 260
    
    [evalues, evectors, mean] = pca(edges, 5)
    #
    #[(a, b), (c, d)], _ = sharpe_boxes(mean, evecs, img, width, height, is_upper, jaw_split, show=show)
    #
    
