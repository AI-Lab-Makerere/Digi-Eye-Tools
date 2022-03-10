from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import Http404
from .models import Image
from .utilities import (
    read_img,
    to_gray,
    convert_pixels,
    binary_threshold,
    get_contours,
    get_contour_center,
    get_extreme_points,
    draw_contours_and_centroid,
    convert_and_save_image,
    create_and_update_csv_updated)
from .unet import build_unet_model
from .unet_old import get_unet
from .unet_utilities import (
    create_dataset_new,
    make_predictions,
    create_and_update_necrosis_csv
)
import os
import cv2
import json
import time
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# weights_path = 'media/weights/unet_model_28_02_22.h5'
weights_path = 'media/weights/unet-weights-18.hdf5'
image_shape = (256, 256, 3)
pixel_class_map = {2: (0, 0, 0),  # background
                   1: (0, 128, 0),  # root
                   0: (128, 0, 0)  # necrosis
                   }
# model = build_unet_model(image_shape)
model = get_unet(3, image_shape)
model.load_weights(weights_path)
# Create your views here.


def index(request):
    return render(request, "pattern/index.html", {})


def upload_images(request):
    if request.method == 'POST':
        start = time.time()
        time_arr = []
        # noinspection PyPep8Naming
        initialPreview = []
        # noinspection PyPep8Naming
        initialPreviewConfig = []

        data_list = []
        for f in request.FILES.getlist('images'):
            instance = Image(images=f)
            instance.save()
            # Batch images for necrosis detection
            data_list.append(BASE_DIR + instance.images.url)

        mask_save_path = 'media/unet_results/'
        csv_save_path = 'media/unet_csv/results.csv'
        dataset = create_dataset_new(data_list)
        gen = make_predictions(dataset, data_list, model, pixel_class_map, mask_save_path, image_shape)
        for d in data_list:
            try:
                n_per, r_per, res_path = next(gen)
                print(res_path)
                create_and_update_necrosis_csv(csv_save_path, os.path.basename(d), f'{(n_per*100):2f}',
                                               datetime.date.today())

                sample_rgb = read_img(res_path)
                sample_root, sample_nec = convert_pixels(sample_rgb)
                sample_root_gray = to_gray(sample_root)
                sample_nec_gray = to_gray(sample_nec)
                sample_root_thresh = binary_threshold(sample_root_gray)
                sample_nec_thresh = binary_threshold(sample_nec_gray)
                sample_root_cont = get_contours(sample_root_thresh)
                sample_root_cont = sorted(sample_root_cont, key=cv2.contourArea, reverse=True)[:1]
                sample_nec_cont = get_contours(sample_nec_thresh)
                center_of_root = get_contour_center(sample_root_cont[0])
                res_list = [get_extreme_points(s) for s in sample_nec_cont]

                # Draw contours and centroid
                sample_out_labeled = draw_contours_and_centroid(sample_rgb, sample_nec_cont, sample_root_cont)

                print(f'Number of lesions {len(res_list)}')

                fpath = os.path.join('media/pattern_results',
                                     os.path.basename(instance.images.url).split(".")[0] + ".png")
                csv_path = os.path.join('media/pattern_csv', "results.csv")

                convert_and_save_image(fpath, sample_out_labeled)
                create_and_update_csv_updated(csv_path, os.path.basename(instance.images.url), "Clone", len(res_list),
                                              res_list, center_of_root, datetime.date.today())

                statinfo = os.stat(res_path)  # Get file size

                imgpath = "<img src=' /" + fpath + "' style='height:260px' class='kv-preview-data krajee-init-preview file-preview-image' alt='RootResult' title='Root Result'>"
                imgconfig = {"type": "image", "caption": "Lesion count:" + str(len(res_list)) + "",
                             "size": statinfo.st_size}

                initialPreview.append(imgpath)
                initialPreviewConfig.append(imgconfig)
            except StopIteration:
                break

            # sample_rgb = read_img(BASE_DIR+instance.images.url)
            # sample_root, sample_nec = convert_pixels(sample_rgb)
            # sample_root_gray = to_gray(sample_root)
            # sample_nec_gray = to_gray(sample_nec)
            # sample_root_thresh = binary_threshold(sample_root_gray)
            # sample_nec_thresh = binary_threshold(sample_nec_gray)
            # sample_root_cont = get_contours(sample_root_thresh)
            # sample_nec_cont = get_contours(sample_nec_thresh)
            # center_of_root = get_contour_center(sample_root_cont[0])
            # res_list = [get_extreme_points(s) for s in sample_nec_cont]
            #
            # # Draw contours and centroid
            # sample_out_labeled = draw_countours_and_centroid(sample_rgb, sample_nec_cont, sample_root_cont)
            #
            # print(len(res_list))
            #
            # fpath = os.path.join('media/pattern_results', os.path.basename(instance.images.url).split(".")[0] + ".png")
            # csv_path = os.path.join('media/pattern_csv', "results.csv")
            #
            # convert_and_save_image(fpath, sample_out_labeled)
            # create_and_update_csv_updated(csv_path, os.path.basename(instance.images.url), "Clone", len(res_list),
            #                               res_list, center_of_root, datetime.date.today())
            #
            # statinfo = os.stat(fpath)  # Get file size
            #
            # imgpath = "<img src=' /" + fpath + "' style='height:260px' class='kv-preview-data krajee-init-preview file-preview-image' alt='RootResult' title='Root Result'>"
            # imgconfig = {"type": "image", "caption": "Lesion count:" + str(len(res_list)) + "",
            #              "size": statinfo.st_size}
            #
            # initialPreview.append(imgpath)
            # initialPreviewConfig.append(imgconfig)
        print(f'Runtime: {time.time()-start}')
        resu = {'initialPreview': initialPreview, 'initialPreviewAsData': 'false',
                'initialPreviewConfig': initialPreviewConfig}
    return HttpResponse(json.dumps(resu))


def download_csv(request):
    path = os.path.join(BASE_DIR, "media/pattern_csv/results.csv")
    try:
        my_file = open(path, "r")
        response = HttpResponse(my_file, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename= Pattern_analysis_results.csv'
        return response
    except Exception as e:
        raise Http404

