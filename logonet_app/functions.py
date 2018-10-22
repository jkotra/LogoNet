import skimage.io
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from keras.models import load_model
import pickle
import argparse


def pp_nd_ss(image_dir):
    ss_arr = []

    img = skimage.io.imread(image_dir)
    img = Image.fromarray(img).resize((640, 480))
    img = np.array(img)

    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0, min_size=500)

    candidates = []

    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if h is 0 or w is 0:
            continue
        if w / h > 2 or h / w > 2:
            continue
        candidates.append(r['rect'])

        image = Image.fromarray(img).crop((x, y, x + w, h + y)).resize((64, 64))
        ss_arr.append(np.array(image))

    ss_arr = np.array(ss_arr) / 255
    return ss_arr,candidates


def load_k_model(model_dir):
    return load_model(model_dir)

def load_labelenc(pickle_dir):
    labenc = open(pickle_dir,'rb')
    labenc = pickle.load(labenc)
    return labenc

def predict(model,img_array):
    # print('Input Shape',img_array.shape) #uncomment for debug
    return model.predict_proba(img_array,10)


def max_predict(predictions,cand,label_encoder,target_list,api=False):
    prediction_result = []
    prediction_prob = []

    target_flag = False

    if target_list is not None:
        target_flag = True

    for pred in predictions:
        if target_flag:
            if label_encoder.inverse_transform([np.argmax(pred,axis=0)])[0] in target_list:
                prediction_prob.append(pred.max())
                prediction_result.append(label_encoder.inverse_transform([np.argmax(pred,axis=0)]))
        if target_flag == False:
            prediction_prob.append(pred.max())
            prediction_result.append(label_encoder.inverse_transform([np.argmax(pred,axis=0)]))

    max_prob = prediction_prob.index(max(prediction_prob))
    if api is True:
        x,y,w,h = cand[max_prob]
        return {
        'prediction': prediction_result[max_prob][0],
        'probability': str(max(prediction_prob)),

        'bbox': {'resize_canvas': '640x480','xywh': {
                'x': str(x),
                'y': str(y),
                'w': str(w),
                'h': str(h),
                 }
                 }
                 }
    else:
        return prediction_result,prediction_prob,max_prob