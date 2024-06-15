import argparse

import json

import Model_Helper
import Data_Processing_Helper

parser = argparse.ArgumentParser()

parser.add_argument('image_path', help='Filepath to image to be predicted')
parser.add_argument('checkpoint', help='Path to checkpoint file')

parser.add_argument('--top_k', help='Return top k most probable names')
parser.add_argument('--category_names', help='Map of category numbers to names')
parser.add_argument('--gpu', help='GPU Use', action='store_true')

print("Parsing arguments")
args = parser.parse_args()
top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

model = Model_Helper.load(args.checkpoint)
processed_image = Data_Processing_Helper.process_image(args.image_path)

probabilities, predict_classes = Model_Helper.predict(model, processed_image, top_k, args.gpu)

classes = []
    
for predict_class in predict_classes:
    classes.append(cat_to_name[predict_class])

print(probabilities)
print(classes)