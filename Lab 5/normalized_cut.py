# import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.cluster import SpectralClustering

def segment(img_path, n_clusters=3, gamma=1, k=7, is_rbf=True):
    
    affinity = 'rbf'
    if not is_rbf:
        affinity='nearest_neighbors'
    
    #Read image using matplotlib package
    img = mpimg.imread(img_path)
    
    #Save original dimensions before resizing
    x_old=img.shape[0]
    y_old=img.shape[1]
    
    #Resize image to fit in memory
    img = resize(image=img, output_shape=(60, 60))
    
    #Save dimensions of the matrix before reshaping
    x = img.shape[0]
    y = img.shape[1]
    z = img.shape[2]
    
    #Reshape the matrix
    img = img.reshape(x * y, z)
    
    #Using SpectralClustering from sklearn package
    clusterer = SpectralClustering(n_clusters=n_clusters, gamma=gamma, n_neighbors=k, affinity=affinity, random_state=8, assign_labels='discretize', n_jobs=-1)
    
    #Fit and predict the image
    res_img = clusterer.fit_predict(img)
    
    #Reshape the matrix again to get the image
    res_img = res_img.reshape(x, y)
    
    #Resize the image again to its original dimensions to compare with ground truth one
    res_img = resize(res_img, output_shape=(x_old, y_old))

#    # Saving the segmented image to target #
#    target_dir = 'output/normalized-cut'
#    
#    # Checking old targets
#    segmented_img_path = os.path.join(target_dir , str(n_clusters) + "-clusters, " + str(gamma) + "-gamma, "
#                                      + str(k) + "-k, "+ str(is_rbf) + "-rbf, " + os.path.basename(img_path))
#    if os.path.exists(segmented_img_path):
#        os.remove(segmented_img_path)
#    
#    # Saving
#    mpimg.imsave(segmented_img_path, )                                         
    return res_img