from scipy.ndimage import gaussian_filter
from skimage import io, measure, img_as_float, color
from skimage.morphology import reconstruction
import numpy as np
import cv2
import csv
import os


def read_img(filename):
    return cv2.imread(filename)


def convert_pixels(rgb_img):
    root_cpy = rgb_img.copy()
    nec_cpy = rgb_img.copy()
    root_cpy[np.where((rgb_img[:, :, 0] == 0) & (rgb_img[:, :, 1] == 0) & (rgb_img[:, :, 2] == 128))] = (0, 128, 0)
    nec_cpy[np.where((rgb_img[:, :, 0] == 0) & (rgb_img[:, :, 1] == 128) & (rgb_img[:, :, 2] == 0))] = (0, 0, 0)
    return root_cpy, nec_cpy


def to_gray(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)


def blur_image(gray_img, kern=(5, 5)):
    return cv2.GaussianBlur(gray_img, kern, 0)


def binary_threshold(in_img, thresh=10):
    return cv2.threshold(in_img, thresh, 255, cv2.THRESH_BINARY)[1]


def get_contours(in_img):
    contours, hierarchy = cv2.findContours(in_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_contour_center(c):
    m = cv2.moments(c)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


# noinspection PyPep8Naming
def get_extreme_points(cnt):
    if cnt.shape[0] > 2:
        cent = get_contour_center(cnt)
    else:
        cent = "Too small (Less than 3 pixels found)"
    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return cent, extLeft, extRight, extBot, extTop


def draw_contour_center(cont, in_img, labl, font=cv2.FONT_HERSHEY_SIMPLEX, text_color=(0, 255, 255), font_size=0.3,
                        line_thick=1):
    centX, centY = get_contour_center(cont)
    # x, y, w, h = cv2.boundingRect(cont)
    cv2.circle(in_img, (centX, centY), 1, (255, 255, 255), -1)
    cv2.putText(in_img, labl, (centX, centY), font, font_size, text_color, line_thick)


def draw_root_centroid(root_center, in_img):
    cv2.circle(in_img, root_center, 3, (255, 255, 255), -1)


def draw_contours_and_centroid(orig_img, nec_cont, root_cont, col=(128, 0, 0), thick=1):
    orig_img_copy = orig_img.copy()
    root_cn = get_contour_center(root_cont[0])
    for idx in range(len(nec_cont)):
        cv2.drawContours(orig_img_copy, nec_cont[idx], -1, col, thick)
        if nec_cont[idx].shape[0] > 2:  # Modified to draw center if more than 2 pixels found in contour
            draw_contour_center(nec_cont[idx], orig_img_copy, str(idx + 1))
        # draw_contour_center(nec_cont[idx], orig_img_copy, str(idx+1))
        draw_root_centroid(root_cn, orig_img_copy)
    return orig_img_copy

# def draw_countours_and_centroid(orig_img, nec_cont, col=(128, 0, 0), thick=3):
#     orig_img_copy = orig_img.copy()
#     for idx in range(len(nec_cont)):
#         cv2.drawContours(orig_img_copy, nec_cont[idx], -1, col, thick)
#         draw_contour_center(nec_cont[idx], orig_img_copy, str(idx+1))
#     return orig_img_copy


def dilate_image(filename):
    # img_orig = io.imread(filename)
    imggray = io.imread(filename, as_gray=True)
    img_orig = cv2.imread(filename)
    image = img_as_float(imggray)
    image = gaussian_filter(image, 5)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')
    outimg = image - dilated

    return img_orig, image, dilated, outimg


def find_contours(dilated_image):
    contours = measure.find_contours(dilated_image, 0.2, fully_connected='low',
                                     positive_orientation='low')
    return contours


def convert_and_save_image(image_path, image):
    # image = cv2.convertScaleAbs(image, alpha=255.0)
    # cv2.imread(input_path)
    # cv2.imwrite(image_path, image,  [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(image_path, image)


def draw_contours(contours, image, cl, thick):
    save_image = ""
    for c in contours[1:]:
        pts = c.astype(np.int64)
        pts[:, [0, 1]] = pts[:, [1, 0]]
        save_image = cv2.polylines(image, [pts], True, cl, thick)
    return save_image


def create_and_update_csv(fname, img_name, clone, les_count, date):
    if os.path.exists(fname):
        try:
            with open(fname, "a", newline="\n") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([img_name, clone, les_count, date])
        except FileNotFoundError as e:
            print(e)
    else:
        with open(fname, "w", newline="\n") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Image Name", "Clone", "Number of Lesions", "Date"])
            csvwriter.writerow([img_name, clone, les_count, date])


# noinspection DuplicatedCode
def create_and_update_csv_updated(fname, img_name, clone, les_count, les_res, rcn, date):
    if os.path.exists(fname):
        try:
            with open(fname, "a", newline="\n") as csvfile:
                csvwriter = csv.writer(csvfile)
                for idx, r in enumerate(les_res):
                    # print(f'idx: {idx}')
                    csvwriter.writerow([img_name, clone, les_count, idx+1, rcn, r[0], r[1], r[2], r[3], r[4], date])
        except FileNotFoundError as e:
            print(e)
    else:
        with open(fname, "w", newline="\n") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Image Name", "Clone", "#Lesions", "LesionID", "RootCent", "LesionCent", "Left",
                                "Right",  "Bottom", "Top", "Date"])
            for idx, r in enumerate(les_res):
                print("New files")
                print(f'idx: {idx}')
                csvwriter.writerow([img_name, clone, les_count, idx+1, rcn, r[0], r[1], r[2], r[3], r[4], date])
