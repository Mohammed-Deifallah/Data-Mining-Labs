import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def segment(img_path, k, iterations=5):
    # Reading image as (width x height x 3) ndarray #
    img = cv2.imread(img_path)
    
    # Converting the (width x height x 3) image into an (n x 3) matrix; where #
    # n = (width x height) and each row is now a pixel vector in the 3-D space of RGB #
    pixs_rgb = img.reshape(-1, 3)
    
    # Initializing a vector that holds which cluster a pixel is currently in #
    pixs_cluster = np.ndarray(pixs_rgb.shape[0], dtype=int)
    
    # Initializing random centroids #
    min_val = np.min(pixs_rgb) # inclisive
    max_val = np.max(pixs_rgb) + 1 # exclusive
    centroids = np.array([np.random.randint(low=min_val, high=max_val, size=3) for i in range(k)])

    # Logic #
    for iteration in range(iterations):
        # Clustering
        for pix_index, pix in enumerate(pixs_rgb):
            distances = [euclidean_distances(pix.reshape(1, -1), centroid.reshape(1, -1)) for centroid in centroids]
            pixs_cluster[pix_index] = np.argmin(distances)
            
        # Updating centroids
        for cluster in range(k):
            cur_cluster_pixs = [pixs_rgb[pix_index] for pix_index, pix_cluster in enumerate(pixs_cluster) if pix_cluster == cluster]
            centroids[cluster] = np.mean(cur_cluster_pixs, axis=0).round().astype(int)
        
    # Creating new segmented image #
    pixs_cluster_centroid = centroids[pixs_cluster.flatten()]
    segmented_img = pixs_cluster_centroid.reshape((img.shape))
    
    # Saving the segmented image to target #
    # Checking target direectory
    target_dir = 'output/k-means/' + str(k)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    # Checking old targets
    segmented_img_path = os.path.join(target_dir , os.path.basename(img_path))
    if os.path.exists(segmented_img_path):
        os.remove(segmented_img_path)
    
    # Saving
    cv2.imwrite(segmented_img_path, segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Returning a (width x height) array of pixel clusters (labels), #
    # along with a (width x height x 3) segmented image #
    return pixs_cluster.reshape(img.shape[0], img.shape[1]), segmented_img