import os
import tqdm
import json
from PIL import Image
import random

def flip_image(image_path):
    img = Image.open(image_path)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img
  

def crop_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    new_width = random.randint(int(width * 0.3), width)
    new_height = random.randint(int(height * 0.3), height)
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = min(left + new_width, width)
    bottom = min(top + new_height, height)
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--output_path", type=str, default="YOUR_OUTPUT_PATH")
    parser.add_argument("--working_dir", type=str, default="YOUR_WORKING_DIR")
    parser.add_argument("--prompts", type=str, default="flip,crop")
    parser.add_argument("--SUN", type=bool, default=False, help="Whether the dataset is SUN")
    args = parser.parse_args()
    prompts = args.prompts.split(",")
    output_path = args.output_path
    working_dir = args.working_dir
    num_images = args.num_images
    SUN = args.SUN
    os.makedirs(output_path, exist_ok=True)

    for prompt in prompts:
        os.makedirs(output_path + f"/{prompt}", exist_ok=True)

    configs_path = working_dir + "/config.json"
    with open(configs_path, "r") as f:
        configs = json.load(f)
        
    pairs = configs["pairs"]
    failed_pairs = {}
    pair_idx = 0
    for pair in pairs:
        main_class = pair["ground_truth"]
        main_class_full_name = pair["ground_truth_full_name"]
        confusing_class = pair["confusing_class"]
        confusing_class_full_name = pair["confusing_class_full_name"]
        
        pair_idx += 1
        print("=====================================")
        print(f"Pair {pair_idx}/{len(pairs)}: {main_class} vs {confusing_class}")
        print("=====================================")
        
        img_parent_dir = os.path.join(working_dir, "train", main_class_full_name)
        if SUN:
            img_parent_dir = working_dir + "/train" + main_class_full_name
        image_files = os.listdir(img_parent_dir)
        image_files = [x for x in image_files if x.endswith(".jpg") or x.endswith(".png") or x.endswith(".jpeg")]
        
        confusing_img_parent_dir = os.path.join(working_dir,"train", confusing_class_full_name)
        if SUN:
            confusing_img_parent_dir = working_dir + "/train" + confusing_class_full_name
        confusing_image_files = os.listdir(confusing_img_parent_dir)
        confusing_image_files = [x for x in confusing_image_files if x.endswith(".jpg") or x.endswith(".png") or x.endswith(".jpeg")]
        for prompt_ in prompts:
            print(f"Prompt: {prompt_}")
            try:
                print("Generating images for main class")
                for i in tqdm.tqdm(range(len(image_files))):
                    img_path = os.path.join(img_parent_dir, image_files[i])
                    if prompt_ == "flip":
                        img = flip_image(img_path)
                    elif prompt_ == "crop":
                        img = crop_image(img_path)
                    os.makedirs(output_path + f"/{prompt_}/{main_class_full_name}", exist_ok=True)
                    img.save(output_path + f"/{prompt_}/{main_class_full_name}/{i}.png")
            except Exception as e:
                print("+++++++++++++++++++++++++++++++++++++++++++")
                print(f"Failed to generate image for {main_class} vs {confusing_class} ({prompt_})")
                print(f"Error: {e}")
                print("+++++++++++++++++++++++++++++++++++++++++++")
                if f"{main_class} vs {confusing_class} ({prompt_})" not in failed_pairs:
                    failed_pairs[f"{main_class} vs {confusing_class} ({prompt_})"] = 0
                failed_pairs[f"{main_class} vs {confusing_class} ({prompt_})"] += 1

            try:
                print("Generating images for confusing class")
                for i in tqdm.tqdm(range(len(confusing_image_files))):
                    img_path = os.path.join(confusing_img_parent_dir, confusing_image_files[i])
                    if prompt_ == "flip":
                        img = flip_image(img_path)
                    elif prompt_ == "crop":
                        img = crop_image(img_path)
                    os.makedirs(output_path + f"/{prompt_}/{confusing_class_full_name}", exist_ok=True)
                    img.save(output_path + f"/{prompt_}/{confusing_class_full_name}/{i}.png")
            except Exception as e:
                print("+++++++++++++++++++++++++++++++++++++++++++")
                print(f"Failed to generate image for {confusing_class} vs {main_class} ({prompt_})")
                print(f"Error: {e}")
                print("Prompt: ", prompt)
                print("+++++++++++++++++++++++++++++++++++++++++++")
                if f"{confusing_class} vs {main_class} ({prompt_})" not in failed_pairs:
                    failed_pairs[f"{confusing_class} vs {main_class} ({prompt_})"] = 0
                failed_pairs[f"{confusing_class} vs {main_class} ({prompt_})"] += 1
            
    with open(output_path + "/failed_pairs.json", "w") as f:
        json.dump(failed_pairs, f)
    
        