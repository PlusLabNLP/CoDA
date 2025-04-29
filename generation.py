import os
import json
import numpy as np
from dotenv import load_dotenv

from filelock import FileLock
from datetime import datetime
from PIL import Image
from filter import filter_one_return_single_path, print_contrastiveness
import random
import torch
from diffusers import StableDiffusion3Pipeline
version = "stabilityai/stable-diffusion-3.5-large-turbo"
sd_ppl = StableDiffusion3Pipeline.from_pretrained(
    version, torch_dtype=torch.bfloat16).to("cuda")

def generate_image(sd_ppl, prompt, negative_prompt, save_path = None, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, guidance_scale=0.0):

    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)

    generator = torch.Generator().manual_seed(seed)
    image = sd_ppl(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        max_sequence_length=512,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    if save_path:
        image.save(save_path, "PNG")

    return image, seed


if __name__ == "__main__":
    load_dotenv()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="contrastive_visual_text")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--working_dir", type=str, default="YOUR_PATH")
    
    args = parser.parse_args()
    prompts = args.prompts.split(",")
    working_dir = args.working_dir
    num_images = args.num_images
    num_test = args.num_test
    output_path = working_dir + "/synthetic_improved/34b"
    os.makedirs(output_path, exist_ok=True)

    configs_path = working_dir + "/config.json"
    with open(configs_path, "r") as f:
        configs = json.load(f)
        
    pairs = configs["pairs"]
    failed_pairs = []
    prompts = ["contrastive_visual_text"]
    for prompt_ in prompts:
        if os.path.exists(output_path + "/failed_pairs.json"): 
            with open(output_path + "/failed_pairs.json", "r") as f:
                failed_pairs = json.load(f)

        for j, pair in enumerate(pairs):
            if j in failed_pairs:
                continue
            for mvc in ['mvc', 'cvm']:
                if mvc == 'mvc':
                    main_class = pair["ground_truth"]
                    main_class_full_name = pair["ground_truth_full_name"]
                    confusing_class = pair["confusing_class"]
                    confusing_class_full_name = pair["confusing_class_full_name"]
                else:
                    main_class = pair["confusing_class"]
                    main_class_full_name = pair["confusing_class_full_name"]
                    confusing_class = pair["ground_truth"]
                    confusing_class_full_name = pair["ground_truth_full_name"]
            
                print("=====================================")
                print(f"{main_class} vs {confusing_class}")
                print("=====================================")
                
                img_parent_dir = os.path.join(working_dir , "train/" + main_class_full_name)
                image_files = os.listdir(img_parent_dir)
                image_paths = filter_one_return_single_path(img_parent_dir, main_class, top_n = 5)# os.path.join(img_parent_dir, image_files[0])
                
                confusing_img_parent_dir = os.path.join(working_dir,"train/" + confusing_class_full_name)
                confusing_image_files = os.listdir(confusing_img_parent_dir)
                confusing_image_paths = filter_one_return_single_path(confusing_img_parent_dir, confusing_class, top_n = 5)

                individual_attributes= []
                with open(output_path + f"/{prompt_}/{main_class_full_name}/attributes.json", "r") as f:
                    attributes = json.load(f)
                negative_prompt = "Do not include any human-made objects or structures. Avoid showing other unnatural elements. Ensure the depiction is realistic and not cartoonish"
                if prompt_ == "contrastive_visual" or prompt_ == "contrastive_visual_text":
                    summary = print_contrastiveness(image_paths, confusing_image_paths, attributes)

                    attributes = [attribute.strip().strip('"').strip('\'') for attribute in attributes]
                    attributes = [attr for attr in attributes if summary[attr] > 0.6]
                    generated_images = [output_path + f"/{prompt_}/{main_class_full_name}/{i}.png" for i in range(num_test)]

                    generable_attributes = []
                    all_attributes = []
                    for start in range(0, len(attributes), 4):
                        subgroup = attributes[start:min(start+4, len(attributes))]
                        prompt = f"Generate a 4K realistic image of {main_class} that contains the following attributes: {', '.join(subgroup)}"
                        print(prompt)
                        for it in range(num_test):
                            img, seed = generate_image(sd_ppl, prompt, negative_prompt, randomize_seed=True)
                            img.save(output_path + f"/{prompt_}/{main_class_full_name}/{it}.png")

                        summary_gen = print_contrastiveness(generated_images, confusing_image_paths, subgroup)
                        all_for_this_group = [(attr, summary_gen[attr]) for attr in summary_gen]
                        all_attributes.extend(all_for_this_group)
                    generable_attributes = sorted(all_attributes, key=lambda x: x[1], reverse=True)[:5]

                    st = {}
                    for key, acc in generable_attributes:
                        st[key] = {
                            "real": summary[key],
                            "synthetic": acc
                        }
                    save_json = {
                        "combined": st,
                    }
                    with open(output_path + f"/{prompt_}/{main_class_full_name}/attributes_contrastiveness_statistics.json", "w") as f:
                        json.dump(save_json, f)
                    generable_attributes = [x[0] for x in generable_attributes]
                else:
                    generable_attributes = attributes
                for k in range(num_images):
                    prompt = f"Generate a 4K realistic image of {main_class} that contains the following attributes: {', '.join(generable_attributes)}"
                    img, seed = generate_image(sd_ppl, prompt, negative_prompt, randomize_seed=True)
                    img.save(output_path + f"/{prompt_}/{main_class_full_name}/{k}.png")

