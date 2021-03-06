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

"""  
    github = @jkotra
 """

parser = argparse.ArgumentParser()

parser.add_argument('--i', action="store", required=True)
parser.add_argument('--o', action="store", required=False)
parser.add_argument('--model', action="store", required=True)
parser.add_argument('--label', action="store", required=True)
parser.add_argument('--target', action="store", required=False)

ap = parser.parse_args()

if ap.target:
    arg_target = ap.target
else:
    arg_target = None


def pp_nd_ss(image_dir):
    global img
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

    return ss_arr,candidates


def load_k_model(model_dir):
    return load_model(model_dir)

def load_labelenc(pickle_dir):
    labenc = open(pickle_dir,'rb')
    labenc = pickle.load(labenc)
    return labenc

def predict(model,img_array):
    print('Input Shape',img_array.shape)
    return model.predict_proba(img_array,10)


model = load_k_model(ap.model)
print("Model loaded from",ap.model)
label_encoder = load_labelenc(ap.label)
print("LabelEncoder Unpickle'd from",ap.label)
ssr,cand = pp_nd_ss(ap.i)
ssr = np.array(ssr) / 255
prediction = predict(model,ssr)


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
        return prediction_result,prediction_prob,max_prob,cand


prediction_result,prediction_prob,max_prob,cand = max_predict(prediction,cand,label_encoder,arg_target)

print(prediction_result[max_prob],'=>',max(prediction_prob))

x,y,h,w = cand[max_prob]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
rect = mpatches.Rectangle((x,y), w, h, fill=False, edgecolor='red', linewidth=1)
ax.add_patch(rect)
ax.text(
            x,
            y,
    "{} - {}".format(prediction_result[max_prob],max(prediction_prob)),
            fontsize=13,
bbox=dict(facecolor='blue', alpha=0.7))

ax.imshow(img)
plt.show()
