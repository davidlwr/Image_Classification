import cv2
import numpy as np
import requests
import base64
import json
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--img_path", "-p", default="", help="Path to the image file to send to model")
arg_parser.add_argument("--url", "-u", default="http://localhost:8501/v1/models/VGG", help="url of REST API")
args                    = arg_parser.parse_args()
restore_path            = args.restorepath

# Img to array
img_path = args.img_path
np_img = cv2.imread(img_path)
np_img_float = np_img.astype('float32')
# print(np_img_float)
    
# Payload
payload = {}
payload['instances'] = [np_img_float.tolist()]
json_payload = json.dumps(payload)
# print(json_payload)

print("starting prediction ================================================")
url_predict  = args.url + ":predict"
response =  requests.post(url=url_predict, data=json_payload)
print(json.dumps(response.json()), indent=2)
print("ending prediction ==================================================")

print("starting classification ============================================")
url_classify = args.url + ":classify"
response =  requests.post(url=url_classify, data=json_payload)
print(json.dumps(response.json()), indent=2)
print("Ending classification ==============================================")

