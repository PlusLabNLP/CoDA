import re
import os
import json
import torch
import random
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


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


def run_classification(pair, pair_dirs, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None, num_val = 20):
    def image_parser(args):
        out = args.image_file.split(args.sep)
        return out

    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(image_files):
        out = []
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
        return out
    
    args = type('Args', (), {
        "model_path": finetuned_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
        "model_base": base_model_path,
        "model_name": None,
        "query": "What are the things I should be cautious about when I visit here?",
        "conv_mode": None,
        "image_file": "https://llava-vl.github.io/static/images/view.jpg",
        "sep": ",",
        "temperature": 0.7,
        "top_p": 1.0,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    if not finetuned_model_path:
        args = type('Args', (), {
            "model_path": base_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": None,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0.7,
            "top_p": 1.0,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
    if not finetuned_model_path:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = "llava-v1.6-34b-lora"


    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Run llava
    all_species = pair
    class_1 = pair[0]
    class_2 = pair[1]
    misclassification_matrix = {class_1: {class_1: 0, class_2: 0}, class_2: {class_1: 0, class_2: 0}}
    for i in range(2):
        common_name = pair[i]
        species_dir_name = pair_dirs[i]
        images = os.listdir(species_dir_name)
        images = [img for img in images if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
        images = [os.path.join(species_dir_name, img) for img in images]
        images = images[-num_val:]

        for file_name in tqdm(images):
            provided_options = random.sample(all_species, len(all_species))
            provided_options_capitalized = [p.capitalize() for p in provided_options]
            args.query = "Is this " + ", ".join(provided_options_capitalized[:-1])+", or " + provided_options_capitalized[-1] + "? Directly answer the question using exactly one of the options."
            args.image_file = file_name
            qs = args.query
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs        
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            image_files = image_parser(args)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
            
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs_lower = outputs.lower()
            valid_output = False
            for option in provided_options:
                option_lower = option.lower()
                if outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) ) or (outputs_lower.startswith("this is a " + option_lower) ) or (outputs_lower.startswith("this is " + option_lower)) or  (outputs_lower.startswith("this is an " + option_lower) ) or (outputs_lower.startswith("the image shows a " + option_lower)) or (outputs_lower.startswith("the image shows an " + option_lower)) or (outputs_lower.startswith("the image shows " + option_lower)):
                    misclassification_matrix[common_name][option] = misclassification_matrix[common_name][option]  + 1/num_val
                    valid_output = True
            if not valid_output:
                print("Error Output: " + outputs)
                print("Prompt: " + prompt)
                
    del model
    import gc
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return misclassification_matrix

def run_classification_w_text_features(pair, pair_dirs, features,base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None, num_val = 20):
    def image_parser(args):
        out = args.image_file.split(args.sep)
        return out

    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(image_files):
        out = []
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
        return out
    
    args = type('Args', (), {
        "model_path": finetuned_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
        "model_base": base_model_path,
        "model_name": None,
        "query": "What are the things I should be cautious about when I visit here?",
        "conv_mode": None,
        "image_file": "https://llava-vl.github.io/static/images/view.jpg",
        "sep": ",",
        "temperature": 0.7,
        "top_p": 1.0,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    if not finetuned_model_path:
        args = type('Args', (), {
            "model_path": base_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": None,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0.7,
            "top_p": 1.0,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
    if not finetuned_model_path:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = "llava-v1.6-34b-lora"


    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Run llava
    all_species = pair
    class_1 = pair[0]
    class_2 = pair[1]
    misclassification_matrix = {class_1: {class_1: 0, class_2: 0}, class_2: {class_1: 0, class_2: 0}}
    for i in range(2):
        common_name = pair[i]
        species_dir_name = pair_dirs[i]
        images = os.listdir(species_dir_name)
        images = [img for img in images if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
        images = [os.path.join(species_dir_name, img) for img in images]
        images = images[-num_val:]

        for file_name in tqdm(images):
            feature_str = f"Backgourd Context:\n\nThe {pair[0]} image contains following features: {', '.join(features[0])}.\nThe {pair[1]} image contains following features: {', '.join(features[1])}.\n\n"
            provided_options = random.sample(all_species, len(all_species))
            provided_options_capitalized = [p.capitalize() for p in provided_options]
            question = f"Task:\n\nYou are an image classification specialist with expertise in categorizing images into specific groups. Given an image, identify its category from the following options: " + ", ".join(provided_options_capitalized[:-1]) + ", or " + provided_options_capitalized[-1] + ". Provide your answer as only one category name for precise classification. Please response with the category name only."
            args.query = feature_str + question
            args.image_file = file_name
            qs = args.query
    
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs        
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            image_files = image_parser(args)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
            
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs_lower = outputs.lower()
            valid_output = False
            for option in provided_options:
                option_lower = option.lower()
                if outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) ) or (outputs_lower.startswith("this is a " + option_lower) ) or (outputs_lower.startswith("this is " + option_lower)) or  (outputs_lower.startswith("this is an " + option_lower) ) or (outputs_lower.startswith("the image shows a " + option_lower)) or (outputs_lower.startswith("the image shows an " + option_lower)) or (outputs_lower.startswith("the image shows " + option_lower)):
                    misclassification_matrix[common_name][option] = misclassification_matrix[common_name][option]  + 1/num_val
                    valid_output = True
            if not valid_output:
                print("Error Output: " + outputs)
                print("Prompt: " + prompt)
                
    del model
    import gc
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return misclassification_matrix


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="YOUR_PATH")
    parser.add_argument("--workspace", type=str, default="YOUR_PATH")
    parser.add_argument("--feature_extraction_approachs", type=str, default="contrastive_visual")
    parser.add_argument("--model", type=str, default="34b")
    parser.add_argument("--num_val", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--zeroshot", type=bool, default=True)
    args = parser.parse_args()
    
    data_path = args.data_path
    working_space = args.workspace
    feature_extraction_approachs = args.feature_extraction_approachs.split(",")
    model = args.model
    num_epcoh = args.num_epochs
    batch_size = args.batch_size
    zeroshot = args.zeroshot
    num_val = args.num_val
    
    config_path = os.path.join(data_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    pairs = config["pairs"]
    validation_imgs_path = config["val_images_path"]
    synthetic_imgs_path = config["synthetic_images_path"]
    
    os.makedirs(working_space + "/results", exist_ok=True)
        
    for feature_extraction_approach in feature_extraction_approachs:
        print("##############################################\n")
        print(f"Feature Extraction Approach: {feature_extraction_approach}")
        print(f"Zeroshot: {zeroshot}")
        print("##############################################\n")
        
        if zeroshot:
            if feature_extraction_approach != feature_extraction_approachs[0]:
                exit(1)
            finetuned_model_path = None
        else:
            if "new" in working_space:
                finetuned_model_path = working_space + "/ckpts/" + model + "_" + feature_extraction_approach + "_"+ str(ratio) +"_0"
            else:
                finetuned_model_path = working_space + "/ckpts/" + model + "_" + feature_extraction_approach + "_"+ str(ratio) +"_0" + "/"+ str(batch_size) + "_" + str(num_epcoh)
                
        results = {}
        for item in pairs:
            pair = [item["ground_truth"], item["confusing_class"]]
            pair_dirct = [os.path.join(validation_imgs_path, item["ground_truth_full_name"]), os.path.join(validation_imgs_path, item["confusing_class_full_name"])]
        
                            
            if zeroshot or feature_extraction_approach == "crop" or feature_extraction_approach == "flip" or feature_extraction_approach == "armanda":
                res = run_classification(pair, pair_dirct, finetuned_model_path= finetuned_model_path)
            else:
                ground_truth_features_path = synthetic_imgs_path + "/" + model + "/" + feature_extraction_approach + "/" + item["ground_truth_full_name"] +"/attributes.json"
                with open(ground_truth_features_path, "r") as f:
                    ground_truth_features = json.load(f)
                confusing_class_features_path = synthetic_imgs_path + "/" + model + "/" + feature_extraction_approach + "/" + item["confusing_class_full_name"] +"/attributes.json"
                with open(confusing_class_features_path, "r") as f:
                    confusing_class_features = json.load(f)
                features = [ground_truth_features, confusing_class_features]
                res = run_classification_w_text_features(pair, pair_dirct, features, finetuned_model_path= finetuned_model_path)

            
            results["_VS_".join(pair)] = res
        
            if zeroshot:
                with open("results_zeroshot.json", "w") as f:
                    json.dump(results, f)
            else:
                with open(f"{working_space}/results/results_{feature_extraction_approach}.json", "w") as f:
                    json.dump(results, f)
