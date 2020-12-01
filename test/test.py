import requests
import os
import json
# Two Observation:
# Training should be done with random rotation applied to it
# Apart from that, in loss function, we can have a term which can also penalize when prediction length and output length re not of same length
directory = "../testsvhn/"
predicted_outputs =[]
for test_sample in os.listdir(directory):
    if test_sample.endswith(".png"):
        resp = requests.post(" http://127.0.0.1:5000/predict", files={'file': open(directory+test_sample, 'rb')})
        temp_dict = {}
        temp_dict['file'] = test_sample
        temp_dict['response'] = resp.json()
        predicted_outputs.append(temp_dict)
with open(directory+"outputs/predicted_output.json", "w") as fout:
    json.dump(predicted_outputs, fout)