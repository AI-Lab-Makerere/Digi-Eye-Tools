import cv2
import os
import csv
import numpy as np
import tensorflow as tf


def load_image(img_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw)
    return img


def image_preprocessing(img_path, im_shape=(256, 256, 3)):
    # Reading and pre-processing
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw)
    img = tf.image.resize(img, im_shape[:2], method='nearest')
    return img


def create_dataset_new(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    # Apply preprocessing across the dataset
    dataset = dataset.map(image_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def process_output(out_tensor, pixel_mapping, img_shape):
    out_tensor = tf.squeeze(out_tensor, axis=0)  # Remove extra-dimension
    out_array = tf.argmax(out_tensor, axis=-1).numpy()  # Get predicted classes and convert to numpy array
    # print(type(out_array))
    # print(img_shape)
    out_mask = np.zeros(img_shape).astype(np.uint8)
    for c in np.unique(out_array):
        out_mask[np.where(out_array == c)] = pixel_mapping[c]
    return out_mask


def get_necrosis_pixels_and_percentage(in_mask):
    # Get necrosis pixels
    nec = np.where((in_mask[:, :, 0] == 128) & (in_mask[:, :, 1] == 0) & (in_mask[:, :, 2] == 0))
    # Get necrosis and root percentage
    nec_percentage = 0.00
    root_percentage = 100.00
    print(f'Unique pixels: {np.unique(in_mask.reshape(-1, 3), axis=0, return_counts=True)}')
    # counts = np.unique(in_array.reshape(-1, 1), axis=0, return_counts=True)[1]
    counts = np.unique(in_mask.reshape(-1, 3), axis=0, return_counts=True)[1]
    # print(f'Counts: {counts}')
    if len(counts) > 2:
        nec_percentage = counts[2] / (counts[2] + counts[1])
        root_percentage = counts[1] / (counts[2] + counts[1])
    else:
        nec_percentage = 0.00
    return nec_percentage, root_percentage, nec


def annotate_and_save_image_and_mask(orig_image, nec_mask, nec_pixels, save_path, mask_color=(255, 255, 255)):
    # print(f'Mask shape: {nec_mask.shape}')
    # print(f'Mask pixels: {np.unique(nec_mask)}')
    # nec_mask = cv2.convertScaleAbs(nec_mask, alpha=255.0)
    ann_image = orig_image.copy()
    if len(nec_pixels) > 0:
        ann_image[nec_pixels] = mask_color
        combined = np.hstack((orig_image, ann_image, nec_mask))
        cv2.imwrite(save_path+"_results.png", cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        cv2.imwrite(save_path+"_mask.png", cv2.cvtColor(nec_mask, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(save_path + "_results.png", cv2.convertScaleAbs(combined, alpha=255.0))
        # cv2.imwrite(save_path + "_mask.png", nec_mask)
    else:
        combined = np.hstack((orig_image, orig_image, nec_mask))
        cv2.imwrite(save_path+"_results.png", cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        cv2.imwrite(save_path+"_mask.png", cv2.cvtColor(nec_mask, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(save_path + "_results.png", cv2.convertScaleAbs(combined, alpha=255.0))
        # cv2.imwrite(save_path + "_mask.png", nec_mask)


def make_predictions(ds, image_list, mdl, p_map, img_save_path, in_shape):
    for idx, d in enumerate(ds):
        pred = mdl.predict(tf.expand_dims(d, axis=0))
        pred_mask = process_output(pred, p_map, in_shape)
        print(f'Mask shape: {pred_mask.shape}')
        pred_nec_per, pred_root_per, pixel_nec = get_necrosis_pixels_and_percentage(pred_mask)
        print(f'Root {pred_root_per:.2f} Necrosis {pred_nec_per:.2f}')
        annotate_and_save_image_and_mask(d.numpy().astype("uint8"), pred_mask, pixel_nec,
                                         img_save_path+os.path.basename(image_list[idx]).split(".")[0])
        yield pred_nec_per, pred_root_per, img_save_path+os.path.basename(image_list[idx]).split(".")[0]+"_mask.png"


def create_and_update_necrosis_csv(fname, img_name, nec_per, date):
    if os.path.exists(fname):
        try:
            with open(fname, "a", newline="\n") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([img_name, nec_per, date])
        except FileNotFoundError as e:
            print(e)
    else:
        with open(fname, "w", newline="\n") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Image Name", "Root Necrosis Score", "Date"])
            csvwriter.writerow([img_name, nec_per, date])
