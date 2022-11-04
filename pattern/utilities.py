import os
import torch
import cv2
import numpy as np


# Load the model
FIL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = torch.hub.load('ultralytics/yolov5', 'custom', FIL_DIR+'/pattern/best.pt')


def read_image(img_path):
    return cv2.imread(img_path)


def to_gray(img_arr):
    return cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)


def resize_image(img_arr, dim=(224, 224)):
    img_arr = cv2.resize(img_arr, dim)
    return img_arr


def get_root_area(img_path, mdl=model_path):
    results = mdl(img_path)
    return results.xyxy[0].numpy().astype(int)[0][:4]


def cluster_image_new(img_arr, max_iter=100, eps=0.2, k=2, ini=10, f=cv2.KMEANS_RANDOM_CENTERS):
    # Reshape_image and convert to numply float32
    img_arr = np.float32(img_arr.reshape((-1,3)))
    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + f, max_iter, eps)
    # Get Clusters
    ret, labels, centers = cv2.kmeans(img_arr, k, None, criteria, ini,
                                      cv2.KMEANS_RANDOM_CENTERS)
    return ret, labels, (centers)


def process_clustering_results(cent, lbls, img_shape):
    #  Convert back to numpy uint8
    centers = np.uint8(cent)
    # flatten the labels array
    labels = lbls.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    segmented_image = segmented_image.reshape(img_shape)
    return segmented_image


def process_clustering_results_v1(cent, lbls, img_arr, mask_tuple):
    #     Create black mask
    black_seg = np.zeros_like(img_arr)
    #     print(black_seg.shape)

    #  Convert back to numpy uint8
    centers = np.uint8(cent)
    centers_root = centers.copy()

    # Sort centers by R channel and get root pixel center
    non_root_pixels = centers[np.argsort(centers[:, 1])][0]

    # Convert all cluster centers to zero apart from root cluster
    centers_root[np.where(np.all(centers_root == non_root_pixels, axis=1))] = [0, 0, 0]

    # flatten the labels array
    labels = lbls.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    segmented_image_black = centers_root[labels.flatten()]

    segmented_image = segmented_image.reshape(img_arr[mask_tuple[2]:mask_tuple[3],
                                              mask_tuple[0]:mask_tuple[1]].shape)
    segmented_image_black = segmented_image_black.reshape(img_arr[mask_tuple[2]:mask_tuple[3],
                                                          mask_tuple[0]:mask_tuple[1]].shape)

    # Convert for root pixels
    img_arr[mask_tuple[2]:mask_tuple[3], mask_tuple[0]:mask_tuple[1]] = segmented_image
    black_seg[mask_tuple[2]:mask_tuple[3], mask_tuple[0]:mask_tuple[1]] = segmented_image_black

    return img_arr, black_seg


def get_root_pixels(img_arr, mask_img):
    return cv2.bitwise_and(img_arr, img_arr, mask=to_gray(mask_img))


def remove_background(img_path):
    img_rgb = read_image(img_path)
    img_x1, img_y1, img_x2, img_y2 = get_root_area(img_path)
    img_ret, img_labels, img_centers = cluster_image_new(img_rgb[img_y1:img_y2, img_x1:img_x2], max_iter=50, ini=2, k=2)
    img_seg1, img_seg_black = process_clustering_results_v1(img_centers, img_labels, img_rgb.copy(),
                                                            (img_x1, img_x2, img_y1, img_y2))
    img_no_bg = get_root_pixels(img_rgb, img_seg_black)
    return cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2RGB)

