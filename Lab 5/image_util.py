import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat

def show_images(img_path, seg_path):
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(50,50))
    show_image(img_path, axes.flat[0])
    show_segmented(seg_path, axes.flat[1], axes.flat[2])
    plt.show()
    return

def show_image(img_path, ax):
    img = cv2.imread(img_path)
    im_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    #cv2 loads images in BGR mode, it needs to be converted to RGB before it's shown with matplotlib
    ax.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    ax.set_xlabel('Image', fontsize=50)
    ax.legend(loc='upper right')
    return

def show_segmented(seg_path, ax1, ax2):
    mat = loadmat(seg_path)
    groundTruth = mat.get('groundTruth')
    label_num = groundTruth.size
    for i in range(label_num):
        boundary = groundTruth[0][i]['Boundaries'][0][0]
        segmentation = groundTruth[0][i]['Segmentation'][0][0]
        
        ax1.imshow(boundary)
        ax1.set_xlabel('Ground Truth Boundary', fontsize=50)
        ax1.legend(loc='upper right')
        
        ax2.imshow(segmentation)
        ax2.set_xlabel('Ground Truth Segmentation', fontsize=50)
        ax2.legend(loc='upper right')
    return