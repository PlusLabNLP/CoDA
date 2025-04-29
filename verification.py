import os
import json
import base64
import random
import numpy as np
import torch
from openai import OpenAI
from dotenv import load_dotenv
from utils.llava_vllm import LLaVA_VLLM
from PIL import Image

from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

"""
utils
"""

def gpt_call(prompt, system="You are a helpful assistant.", model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].message.content.strip().strip()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_call_w_img(prompt, image_url, system="You are a helpful assistant.", model="gpt-4o-mini"):
    image_url = encode_image(image_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                  {"type": "text",
                   "text": prompt},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{image_url}", },
                  },
              ]},
        ],
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].message.content.strip().strip()

def extract_attributes_from_image(model, img):
    prompt = "Extract visual attributes from the image as much as you can, return in a python list. Here is example response format: [attribute1, attribute2]. No verbose output."
    if model != gpt_call_w_img:
      img = Image.open(img)
    response = model(prompt, img)
    try:
      attributes = response.split('[')[1].split(']')[0]
    except Exception as e:
      response = model(prompt, img)
      attributes = response.split('[')[1].split(']')[0]
    return [attribute.strip() for attribute in attributes.split(",")]
  
"""
Attribute Extraction
"""

def visual_text(img, main_class, confusing_class):
    attributes = extract_attributes_from_image(gpt_call_w_img, img)
    attributes = [attribute.strip('"') for attribute in attributes]
    return attributes

def contrastive_visual_text(img, main_class, confusing_class):
    prompt = f"What attributes of this {main_class} makes it distinct from a {confusing_class}? Extract visual attributes from the image as much as you can, return in a python list. Here is example response format: [attribute1, attribute2]. No verbose output."
    response = gpt_call_w_img(prompt, img)
    try:
      attributes = response.split('[')[1].split(']')[0]
    except Exception as e:
      response = gpt_call_w_img(prompt, img)
      attributes = response.split('[')[1].split(']')[0]
    attributes = [attribute.strip().strip('"') for attribute in attributes.split(",")]
    return attributes

def contrastive_text(img, main_class, confusing_class):
    prompt = f"What visual attributes of a {main_class} makes it distinct from a {confusing_class}? Write visual attributes as much as you can, return in a python list. Here is example response format: [attribute1, attribute2]"
    response = gpt_call(prompt)
    try:
      attributes = response.split('[')[1].split(']')[0]
    except Exception as e:
      response = gpt_call(prompt)
      attributes = response.split('[')[1].split(']')[0]
    attributes = [attribute.strip().strip('"') for attribute in attributes.split(",")]
    return attributes

def text(img, main_class, confusing_class):
    prompt = f"Write visual attributes of a {main_class} as much as you can, return in a python list. Here is example response format: [attribute1, attribute2]"
    response = gpt_call(prompt)
    try:
      attributes = response.split('[')[1].split(']')[0]
    except Exception as e:
      response = gpt_call(prompt)
      attributes = response.split('[')[1].split(']')[0]
    attributes = [attribute.strip().strip('"') for attribute in attributes.split(",")]
    return attributes


"""
Verification
"""

def verify_attributes(attributes, img, model):
  prompt = f"You are an image verification specialist. Your task is to meticulously assess the image for specific attributes and confirm their presence. For each attribute in the list, carefully check the image, examine visual elements such as color, shape, texture, position, and context clues that might indicate whether the attribute is present. Provide a binary python output list, where each element is either 1 (attribute is present) or 0 (attribute is absent), corresponding exactly to the order of attributes provided. \n\nAttributes to Verify:{attributes} \n\nExpected Output: A list of 0s and 1s indicating the presence or absence of each attribute, in the same order as listed. Here is an example output: [0, 1, 1]."
  if model != gpt_call_w_img:
    img = Image.open(img)
  response = model(prompt, img)
  try:
    response = response.split('[')[1].split(']')[0]
  except Exception as e:
    response = model(prompt, img)
    response = response.split('[')[1].split(']')[0]
  return [int(i) for i in response.split(",") if i.strip() != ""]

def calculate_match_score(target_attributes, extracted_attributes):
    target_attributes = set(target_attributes)
    extracted_attributes = set(extracted_attributes)
    return len(target_attributes.intersection(extracted_attributes)) / len(target_attributes)

def calculate_hard_string_match_score(target_attributes, extracted_attributes):
    target_attributes = [attribute.lower() for attribute in target_attributes]
    extracted_attributes = " ".join(extracted_attributes).lower()
    count = 0
    for attribute in target_attributes:
        if attribute in extracted_attributes:
            count += 1
    return count / len(target_attributes)
  
def calculate_overlap_score(target_attributes, extracted_attributes):
    target_attributes = " ".join(target_attributes).lower()
    extracted_attributes = " ".join(extracted_attributes).lower()
    target_attributes = set(target_attributes.split())
    extracted_attributes = set(extracted_attributes.split())
    return len(target_attributes.intersection(extracted_attributes)) / len(target_attributes.union(extracted_attributes))

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract and verify visual attributes from images.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to the JSON file containing data configuration.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--attributes_prompts", nargs='+', required=True, help="List of attribute extraction methods to use.")
    return parser.parse_args()
  
def main(data_config_path, output_path, attributes_prompts):
    print("Loading LLaVA Model...")  
    llava_vllm = LLaVA_VLLM("liuhaotian/llava-v1.6-34b")

    with open(data_config_path, "r") as f:
        data_config = json.load(f)

    pairs = data_config["pairs"]
    imgs_path = data_config["synthetic_images_path"] + "/34b"
    scores = {}

    pair_idx = 0
    for pair in pairs:
        pair_idx += 1
        print(f"\nPair {pair_idx}/{len(pairs)}\n")

        main_class = pair["ground_truth"]
        main_class_name = pair["ground_truth_full_name"]
        confusing_class = pair["confusing_class"]
        confusing_class_name = pair["confusing_class_full_name"]
        
        for attributes_prompt in attributes_prompts:
            print(f"\nAttributes Prompt: {attributes_prompt}\n")
            if attributes_prompt not in scores:
                scores[attributes_prompt] = {}
            
            main_class_imgs = [img for img in os.listdir(f"{imgs_path}/{attributes_prompt}/{main_class_name}") 
                               if img.endswith((".jpg", ".jpeg", ".png"))]
            scores[attributes_prompt][main_class] = []
            
            real_imgs_path = f"{data_config['real_images_path']}/{main_class_name}"
            real_img = os.listdir(real_imgs_path)[0]
            img_path = f"{real_imgs_path}/{real_img}"
            try:
                target_attributes = globals()[attributes_prompt](img_path, main_class, confusing_class)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            for img in tqdm(main_class_imgs, desc=f"Main Class: {main_class}"):
                img_path = f"{imgs_path}/{attributes_prompt}/{main_class_name}/{img}"
                res = verify_attributes(str(target_attributes), img_path, llava_vllm)
                result = {target_attributes[i]: res[i] for i in range(len(target_attributes))} if res else {}
                scores[attributes_prompt][main_class].append({"img": img, "target_attributes": target_attributes, "result": result, "score": np.mean(res)})
                with open(output_path, "w") as f:
                    json.dump(scores, f)
            
            confusing_class_imgs = [img for img in os.listdir(f"{imgs_path}/{attributes_prompt}/{confusing_class_name}") 
                                    if img.endswith((".jpg", ".jpeg", ".png"))]
            scores[attributes_prompt][confusing_class] = []
            
            real_imgs_path = f"{data_config['real_images_path']}/{confusing_class_name}"
            real_img = os.listdir(real_imgs_path)[0]
            img_path = f"{real_imgs_path}/{real_img}"
            try:
                target_attributes = globals()[attributes_prompt](img_path, confusing_class, main_class)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            for img in tqdm(confusing_class_imgs, desc=f"Confusing Class: {confusing_class}"):
                img_path = f"{imgs_path}/{attributes_prompt}/{confusing_class_name}/{img}"
                res = verify_attributes(str(target_attributes), img_path, llava_vllm)
                result = {target_attributes[i]: res[i] for i in range(len(target_attributes))} if res else {}
                scores[attributes_prompt][confusing_class].append({"img": img, "target_attributes": target_attributes, "result": result, "score": np.mean(res)})
                with open(output_path, "w") as f:
                    json.dump(scores, f)
    
    print("Verification Done!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.data_config, args.output_path, args.attributes_prompts)
