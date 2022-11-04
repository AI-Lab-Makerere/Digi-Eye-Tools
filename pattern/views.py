from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import Http404
from .models import Image
import os
import cv2
import json
import datetime
import numpy as np
from sklearn.cluster import KMeans
import pickle
import warnings
import glob
from tensorflow.keras.applications import ResNet50, VGG19, EfficientNetB5
from tensorflow.keras.utils import load_img #For tf >= 2.9
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import csv
import torch
from .utilities import remove_background, resize_image

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Create your views here.


def index(request):
    return render(request, "pattern/index.html", {})


def home(request):
    return render(request, "pattern/landing.html", {})


def rgb_page(request):
    return render(request, "pattern/sample_index.html", {})


def mealiness_page(request):
    return render(request, "pattern/mealiness_index.html", {})


def upload_images(request):
    fname = os.path.join('media/pattern_csv/', request.get_host()+".csv")
    with open(fname, "w", newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image Name", "Prediction type","Root Score", "Date"])
    if request.method == 'POST':
        # noinspection PyPep8Naming
        initialPreview = []
        # noinspection PyPep8Naming
        initialPreviewConfig = []

        for f in request.FILES.getlist('images'):
            instance = Image(images=f)
            instance.save()
            # Batch images for necrosis detection
            # data_list.append(BASE_DIR + instance.images.url)

            def visualize_Dominant_colors(cluster, C_centroids):
                C_labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
                (C_hist, _) = np.histogram(cluster.labels_, bins = C_labels)
                C_hist = C_hist.astype("float")
                C_hist /= C_hist.sum()

                rect_color = np.zeros((50, 300, 3), dtype=np.uint8)
                img_colors = sorted([(percent, color) for (percent, color) in zip(C_hist, C_centroids)])
                start = 0
                for (percent, color) in img_colors:
                    print(color, "{:0.2f}%".format(percent * 100))
                    end = start + (percent * 300)
                    cv2.rectangle(rect_color, (int(start), 0), (int(end), 50), \
                                color.astype("uint8").tolist(), -1)
                    start = end
                return rect_color

            # Load image
            src_image = cv2.imread('.'+instance.images.url)
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
            reshape_img = src_image.reshape((src_image.shape[0] * src_image.shape[1], 3))

            # Display dominant colors Present in the image
            KM_cluster = KMeans(n_clusters=4).fit(reshape_img)
            print(type(KM_cluster))

            visualize_color = visualize_Dominant_colors(KM_cluster, KM_cluster.cluster_centers_)
            visualize_color1 = cv2.cvtColor(visualize_color, cv2.COLOR_RGB2BGR)
            visualize_color2 = cv2.cvtColor(visualize_color1, cv2.COLOR_BGR2RGB)
            print("I am here ...")
            print(visualize_color2[2])
            #cv2.imshow('visualize_Color', visualize_color)
            # cv2.waitKey()
            count =0
            for arr in visualize_color2[0]:
                print(arr)
                print(arr[0])
                if arr[0]<=120:
                    count = count +1
                    print(f'count:{count}')
                elif arr[0]> 190 and arr[1]>199:
                    count = count +1
                    continue
                elif arr[0]> 120:
                    break
            list_result = visualize_color2[0][count]   
            result=np.array2string(visualize_color2[0][count], formatter={'float_kind':lambda x: "%.2f" % x})

            # #cv2.imshow('Source image',src_img)
            # cv2.imshow('Average Color',d_img)
            # cv2.waitKey(0)
            rgb_value = "893, 90, 33"
            warnings.filterwarnings("ignore")
            print(BASE_DIR)
            print(type(list_result))
            print(list_result)
            loaded_model_r = pickle.load(open(BASE_DIR+'/models/color_measurement_LR_regressmodel2.sav', 'rb'))
            model_result = loaded_model_r.predict([list_result]).item()
            name_img= instance.images.name
            print(name_img)
            name_img =name_img.split("/")
            print(name_img) 
            final_result = "RGB:"+result+" "+"Score:"+ str(model_result)+ " Image:" + name_img[1]+ " Size:"
            
            #create result csv
            cpath = os.path.join('media/pattern_csv/', request.get_host()+".csv")
            create_and_update_csv(cpath, os.path.basename('.'+instance.images.url), "Color evalution 3 roots",str(model_result), datetime.datetime.today())


            statinfo = os.stat('.'+instance.images.url)  # Get file size
            print(statinfo.st_size)

            imgpath = "<img src=' "+ instance.images.url + "' style='height:260px' class='kv-preview-data krajee-init-preview file-preview-image' alt='RootResult' title='Analysis Result'>"
            print(imgpath)
            imgconfig = {"type": "image", "caption": final_result,
                         "size": statinfo.st_size}

            initialPreview.append(imgpath)
            initialPreviewConfig.append(imgconfig)
            
            

        resu = {'initialPreview': initialPreview, 'initialPreviewAsData': 'false',
                'initialPreviewConfig': initialPreviewConfig}
    return HttpResponse(json.dumps(resu))


def upload_sample(request):
    fname = os.path.join('media/pattern_csv/', request.get_host()+".csv")
    with open(fname, "w", newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image Name", "Prediction type","Root Score", "Date"])
    if request.method == 'POST':
        # noinspection PyPep8Naming
        initialPreview = []
        # noinspection PyPep8Naming
        initialPreviewConfig = []

        for f in request.FILES.getlist('images'):
            instance = Image(images=f)
            instance.save()
            # Batch images for necrosis detection
            # data_list.append(BASE_DIR + instance.images.url)

            print(instance.images.url)
            print(request.get_host())

        
        

            def visualize_Dominant_colors(cluster, C_centroids):
                C_labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
                (C_hist, _) = np.histogram(cluster.labels_, bins = C_labels)
                C_hist = C_hist.astype("float")
                C_hist /= C_hist.sum()

                rect_color = np.zeros((50, 300, 3), dtype=np.uint8)
                img_colors = sorted([(percent, color) for (percent, color) in zip(C_hist, C_centroids)])
                start = 0
                for (percent, color) in img_colors:
                    print(color, "{:0.2f}%".format(percent * 100))
                    end = start + (percent * 300)
                    cv2.rectangle(rect_color, (int(start), 0), (int(end), 50), \
                                color.astype("uint8").tolist(), -1)
                    start = end
                return rect_color

            # Load image
            src_image = cv2.imread('.'+instance.images.url)
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
            reshape_img = src_image.reshape((src_image.shape[0] * src_image.shape[1], 3))

            # Display dominant colors Present in the image
            KM_cluster = KMeans(n_clusters=2).fit(reshape_img)
            print(type(KM_cluster))

            visualize_color = visualize_Dominant_colors(KM_cluster, KM_cluster.cluster_centers_)
            visualize_color = cv2.cvtColor(visualize_color, cv2.COLOR_RGB2BGR)
            visualize_color = cv2.cvtColor(visualize_color, cv2.COLOR_BGR2RGB)
            print("I am here ...")
            print(visualize_color)
            #cv2.imshow('visualize_Color', visualize_color)
            # cv2.waitKey()
            count =0
            for arr in visualize_color[0]:
                print(arr[0])
                if arr[0]<=120:
                    count = count +1
                elif arr[0]>120:
                    break
            list_result = visualize_color[0][count]
            print(list_result)   
            result=np.array2string(visualize_color[0][count], formatter={'float_kind':lambda x: "%.2f" % x})

            # #cv2.imshow('Source image',src_img)
            # cv2.imshow('Average Color',d_img)
            # cv2.waitKey(0)
            rgb_value = "893, 90, 33"
            warnings.filterwarnings("ignore")
            print(BASE_DIR)
            print(type(list_result))
            print(list_result)
            loaded_model_r = pickle.load(open(BASE_DIR+'/models/color_measurement_RF_regressmodel.sav', 'rb'))
            model_result = loaded_model_r.predict([list_result]).item()
            name_img= instance.images.name
            print(name_img)
            name_img =name_img.split("/")
            print(name_img) 
            final_result = "RGB:"+result+" "+"Score:"+ str(model_result)+ " Image:"+ name_img[1]+ " Size"
            
            #create result csv
            cpath = os.path.join('media/pattern_csv/', request.get_host()+".csv")
            create_and_update_csv(cpath, os.path.basename('.'+instance.images.url), "Color evalution single root",str(model_result), datetime.datetime.today())


            statinfo = os.stat('.'+instance.images.url)  # Get file size
            print(statinfo.st_size)

            imgpath = "<img src=' "+ instance.images.url + "' style='height:260px' class='kv-preview-data krajee-init-preview file-preview-image' alt='RootResult' title='Analysis Result'>"
            print(imgpath)
            imgconfig = {"type": "image", "caption": final_result,
                         "size": statinfo.st_size}
            

            initialPreview.append(imgpath)
            initialPreviewConfig.append(imgconfig)
            
            

        resu = {'initialPreview': initialPreview, 'initialPreviewAsData': 'false',
                'initialPreviewConfig': initialPreviewConfig}
    return HttpResponse(json.dumps(resu))


def predict_mealiness(request):
    fname = os.path.join('media/pattern_csv/', request.get_host()+".csv")
    with open(fname, "w", newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image Name", "Prediction type", "Root Score", "Date"])
    if request.method == 'POST':
        # noinspection PyPep8Naming
        initialPreview = []
        # noinspection PyPep8Naming
        initialPreviewConfig = []

        # Path to trained model
        model_file = BASE_DIR+'/models/log_reg.pickle'


        # Load mealiness prediction model
        with open(model_file, "rb") as p_file:
            model = pickle.load(p_file)

        # Using EfficientNetB5 as feature extractor
        feature_extractor = EfficientNetB5(weights="imagenet", include_top=False)

                # Extract image features.
        def prepare_image(img_path, backbone, tgt_size=(224, 224)):
            """
                Function takes a path to an image, the model to be used for feature extraction and
                the image input shape to the feature extractor.
                It returns a numpy array of extracted features.
            """
            # image = load_img(img_path, target_size=tgt_size) # Default target size assumes VGG19 input. Edit as required.
            # image = img_to_array(image)
            image = remove_background(img_path)
            image = resize_image(image)
            image = np.expand_dims(image, axis=0)
            features = backbone.predict(image) # Extract features using backbone
            return features.flatten().reshape(1, -1) # Reshape to prepare input for model


        # Mealiness prediction
        def predict_mealiness(X, estimator):
            """ Takes an array and a trained model as parameters.
                Returns an array of predictions for mealiness.
            """
            return estimator.predict(X)      
        
        for f in request.FILES.getlist('images'):
            instance = Image(images=f)
            instance.save()
            # Batch images for necrosis detection
            # data_list.append(BASE_DIR + instance.images.url)

            print(instance.images.url)
            print(request.get_host())


            # Load images in list
            image_files = glob.glob('.'+instance.images.url)
            print(image_files)
            print('.'+instance.images.url)



            # Image shape
            img_shape = (224,224)


            # Get features for a single image
            single_image_features = prepare_image('.'+instance.images.url, feature_extractor)

            # Optional for extracting and predicting for a batch of images (uncomment to run)
            # multi_image_features = np.array([prepare_image(i, feature_extractor) for i in image_files])

            # Predictions is an array of float values for the mealiness
            mealiness = predict_mealiness(single_image_features, model)
            # mealiness = predict_mealiness(multi_image_features, model)

            print(mealiness)
            name_img= instance.images.name
            print(name_img)
            name_img =name_img.split("/")
            print(name_img) 

            final_result = "<br/>"+"Mealiness Score:"+ str(mealiness) + " Image: "+name_img[1]+ "size"
            statinfo = os.stat('.'+instance.images.url)  # Get file size
            print(statinfo.st_size)

            #create result csv
            cpath = os.path.join('media/pattern_csv/', request.get_host()+".csv")
            create_and_update_csv(cpath, os.path.basename('.'+instance.images.url), "Mealiness evalution",str(mealiness[0]), datetime.datetime.today())

            imgpath = "<img src=' "+ instance.images.url + "' style='height:260px' class='kv-preview-data krajee-init-preview file-preview-image' alt='RootResult' title='Analysis Result'>"
            print(imgpath)
            imgconfig = {"type": "image", "caption": final_result,
                         "size": statinfo.st_size}
    

            initialPreview.append(imgpath)
            initialPreviewConfig.append(imgconfig)

            # remove_img(path=BASE_DIR,img_name='.'+instance.images.url)

            

        resu = {'initialPreview': initialPreview,
                'initialPreviewConfig': initialPreviewConfig}
    return HttpResponse(json.dumps(resu))

def download_csv(request):
    path = os.path.join(BASE_DIR, "media/pattern_csv/" , request.get_host()+".csv")
    try:
        my_file = open(path, "r")
        response = HttpResponse(my_file, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename= Pattern_analysis_results.csv'
        return response
    except Exception as e:
        print(e)
        raise Http404

def create_and_update_csv(fname, img_name, prediction_type,score, date):
    if os.path.exists(fname):
        try:
            with open(fname, "a", newline="\n") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([img_name,prediction_type, score, date])
        except FileNotFoundError as e:
            print(e)
    else:
        with open(fname, "w", newline="\n") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Image Name", "Prediction type","Root Score", "Date"])
            csvwriter.writerow([img_name, prediction_type, score, date])


def remove_img(path, img_name):
    os.remove(path + '/' + img_name)
# check if file exists or not
    if os.path.exists(path + '/' + img_name) is False:
        # file did not exists
        return True