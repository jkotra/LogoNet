from flask import Flask, jsonify, request
import json
import numpy as np
from functions import load_k_model,load_labelenc,predict,max_predict,pp_nd_ss

app = Flask(__name__)

CONFIG = json.loads(open('config.json','rb').read())

def load_resources():
    global model,label_encoder
    model = load_k_model(CONFIG['Model_Path'])
    model._make_predict_function()
    label_encoder = load_labelenc(CONFIG['Label_Encoder_Path'])

load_resources()
image_vault = "cache/image.jpg"


@app.route("/logonet",methods=['POST'])
def hello():
    c = 0
    ret_res = {}
    for key in request.files.keys():
        print(key)
        request.files["{}".format(key)].save(image_vault)
        arr,cand = pp_nd_ss(image_vault)
        predictions = predict(model,arr)
        res = max_predict(predictions,label_encoder,api=True)
        ret_res[c] = res
        c = c+1

    return jsonify(ret_res)

if __name__ == '__main__':
    app.run(debug=True)