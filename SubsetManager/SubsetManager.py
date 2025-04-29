import os
import shutil
import random
import json
import torch
import copy
import pandas as pd
import csv
import warnings
warnings.filterwarnings("ignore")
from torch.nn.utils.rnn import pad_sequence

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

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
# from generation import generate_image

class SubsetManager:
    def __init__(self):
        file_path = "subsets_info/paths.json"
        try:
            with open(file_path, 'r') as file:
                paths = json.load(file)
                self.root_path = paths["whole_dataset_path"]
                self.generated_images_path = paths["generated_images_path"]
                self.filtered_images_path = paths["filtered_images_path"]
                self.subsets_info_dir = "subsets_info"
                print(f"Using data from {self.root_path}")
        except FileNotFoundError:
            return f"Error: The file '{file_path}' was not found."


    def create_subset(self, supercategory, name = None, size = 15, sampling_method = "Random", sampling_args = None):
        if not name:
            # Generate a name using the sampling method and the largest unseen number
            index = 1
            while os.path.exists(os.path.join(self.subsets_info_dir, f"{supercategory}_{sampling_method}_{index}")):
                index += 1
            name = f"{supercategory}_{sampling_method}_{index}"
        else:
            name = f"{supercategory}_{name}"
        # Create the directory under subsets_info/
        subset_path = os.path.join(self.subsets_info_dir, name)
        os.makedirs(subset_path, exist_ok=True)
        if sampling_method == "Random":
            with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
            # Load the data from the file
                data = json.load(file)
            all_species = list({category["common_name"] for category in data["categories"] if category["supercategory"] == supercategory})
            all_species_sample = random.sample(all_species, size)
            all_species_dirs = {}
            for common_name in all_species_sample:
                for category in data["categories"]:
                    if category["supercategory"] == supercategory:
                        if category["common_name"] == common_name:
                            all_species_dirs[common_name] = (category["image_dir_name"])
                            break
            sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')
            with open(sample_file_path, 'w') as sample_file:
                json.dump(all_species_dirs, sample_file, indent=4)

        else:
            print(f"Error: no such sampling method: {sampling_method}")
        return name

    def remove_subset(self, name):
        """
        Remove a subset directory and all its contents.

        :param name: The name of the subset directory to be removed.
        :return: A message indicating success or failure.
        """
        subset_path = os.path.join(self.subsets_info_dir, name)

        if os.path.exists(subset_path):
            try:
                shutil.rmtree(subset_path)
                return f"Subset '{name}' removed successfully."
            except Exception as e:
                return f"An error occurred while removing subset '{name}': {e}"
        else:
            return f"Subset '{name}' does not exist."
        
    def get_all_species_in_subset_and_directories(self, name, prefix = True):
        """
        Get the species in a subset and the corresponding directories. 

        :return: a dictionary of species and their directories
        """
        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')

        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    all_species_dirs = json.load(sample_file)
                    real_image_dirs = {}
                    for key in all_species_dirs:
                        if prefix:
                            real_image_dirs[key] =  os.path.join(self.root_path , "train_mini" , all_species_dirs[key])
                        else:
                            real_image_dirs[key] = (all_species_dirs[key])
                return real_image_dirs
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return []
        else:
            print(f"Subset '{name}' does not exist or 'sampled_species.json' is missing.")
            return []

    def get_all_species_in_subset(self, name):
        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')

        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    species_list = json.load(sample_file)
                return list(species_list.keys())
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return []
        else:
            print(f"Subset '{name}' does not exist or 'sampled_species.json' is missing.")
            return []

    def get_list_of_subset_names(self):
        """
        Get a list of all subset directory names.

        :return: A list of subset names.
        """
        try:
            subset_names = [name for name in os.listdir(self.subsets_info_dir) if os.path.isdir(os.path.join(self.subsets_info_dir, name))]
            return subset_names
        except Exception as e:
            print(f"An error occurred while listing subset names: {e}")
            return []
    

    def run_classification_for_subset(self, name, num_samples = 50, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None):
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

        def is_sub_species(species, higher_rank_classification):
            for rank in higher_rank_classification:
                if not higher_rank_classification[rank] == None:
                    if not higher_rank_classification[rank] == species[rank]:
                        return False
            return True

        def sampling_from_species(species, n_samples):
            target_level = 'common_name'
            for i in range(len(heirachy_of_classification)):
                if species[heirachy_of_classification[i]] == None:
                    target_level = heirachy_of_classification[i-1]
                    break
            count = 0
            index = 0
            samples = []
            indices = random.sample(range(len( data['annotations'])), len( data['annotations']))
            while count < n_samples:
                index_original = indices[index]
                annotation = data['annotations'][index_original]
                category_id = annotation["category_id"]
                category = data["categories"][category_id]
                if is_sub_species(category, species):
                    samples.append(index_original)
                    count = count + 1
                index = index + 1
            return samples

        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')
        size_of_llava = base_model_path.split("-")[-1]
        
        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    all_species_dirs = json.load(sample_file)
                    all_species = list(all_species_dirs.keys())
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return
        if size_of_llava == "34b":
            if not finetuned_model_path:
                csv_path = (os.path.join(subset_path, 'misclassification_matrix.csv'))
            else: 
                csv_path = (os.path.join(subset_path, f'misclassification_matrix_{finetuned_model_path}.csv'))
        else:
            if not finetuned_model_path:
                csv_path = (os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix.csv'))
            else: 
                (os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix_{finetuned_model_path}.csv'))            
        if os.path.exists(csv_path):
            return get_misclassification_matrix_pd(self, name, base_model_path, finetuned_model_path)
        args = type('Args', (), {
            "model_path": finetuned_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": base_model_path,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0,
            "top_p": None,
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
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
        model_name = get_model_name_from_path(args.model_path)


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
        with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
            # Load the data from the file
            data = json.load(file)
        supercategory = name.split('_')[0]
        heirachy_of_classification = ["supercategory",'kingdom','phylum', 'class', 'order', 'family', 'genus', 'common_name']
        
        # Run llava
        misclassification_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        count_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        for common_name in all_species:
            print(common_name)
            species = [category for category in data["categories"] if category["supercategory"]==supercategory and category["common_name"] == common_name][0]
            image_ids = sampling_from_species(species, num_samples)    
            for image_id in image_ids:
                # print("Processing image: " + str(image_id))
                image = data['images'][image_id]
                file_name = image['file_name']
                inat_dir = self.root_path
                file_name = os.path.join(inat_dir, file_name)
                # args.query = "Is this " + ", ".join(provided_options)+"? Answer the question using a few words."
                for _ in range(1): # 7
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
                    # images[0].show()
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
                    # print(prompt)
                    # print(outputs)
                    outputs_lower = outputs.lower()
                    valid_output = False
                    for option in provided_options:
                        option_lower = option.lower()
                        def remove_parentheses(s):
                            """Removes parentheses from the string."""
                            return s.replace("(", "").replace(")", "")

                        def remove_bracketed_elements(s):
                            """Removes the bracketed elements from the string."""
                            return re.sub(r'\s*\(.*?\)\s*', ' ', s).strip()

                        # Existing code snippet with small adjustments
                        option_lower_clean = remove_parentheses(option_lower)
                        option_lower_no_bracket = remove_bracketed_elements(option_lower)
                        count_matrix[common_name][option] = count_matrix[common_name][option] + 1

                        # Check for matches with and without the bracketed elements
                        if (outputs_lower.startswith(option_lower_clean) or
                            outputs_lower.startswith("a " + option_lower_clean) or
                            outputs_lower.startswith(option_lower_no_bracket) or
                            outputs_lower.startswith("a " + option_lower_no_bracket)
                            or outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) )):

                            misclassification_matrix[common_name][option] = (misclassification_matrix[common_name][option] * (count_matrix[common_name][option] - 1) + 1)/count_matrix[common_name][option]
                            valid_output = True
                        else:
                            misclassification_matrix[common_name][option] = (misclassification_matrix[common_name][option] * (count_matrix[common_name][option] - 1))/count_matrix[common_name][option]
                    if not valid_output:
                        print("Error Output: " + outputs)
                        print("Prompt: " + prompt)
        size_of_llava = base_model_path.split("-")[-1]
        misclassification_matrix.to_csv(csv_path)
        del model
        import gc
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return misclassification_matrix

    def get_misclassification_matrix_pd(self, name, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None):
        subset_path = os.path.join(self.subsets_info_dir, name)
        size_of_llava = base_model_path.split("-")[-1]
        if size_of_llava == "34b":
            if not finetuned_model_path:
                misclassification_matrix_path = os.path.join(subset_path, 'misclassification_matrix.csv')
            else:
                misclassification_matrix_path = os.path.join(subset_path, f'misclassification_matrix_{finetuned_model_path}.csv')
        else:
            if not finetuned_model_path:
                misclassification_matrix_path = os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix.csv')
            else:
                misclassification_matrix_path = os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix_{finetuned_model_path}.csv')
        misclassification_matrix_loaded = pd.read_csv(misclassification_matrix_path)
        with open(misclassification_matrix_path, 'r') as file:
            reader = csv.reader(file)
            all_species = next(reader)
        misclassification_matrix = pd.DataFrame(0.0, index=all_species, columns=all_species)
        for species in all_species:
            for i in range(len(all_species)):
                misclassification_matrix[species][all_species[i]] = misclassification_matrix_loaded[species][i]
        return misclassification_matrix
    
    def get_misclassfication_in_dict(self, name, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None, val = False):
        subset_path = os.path.join(self.subsets_info_dir, name)
        if not val:
            size_of_llava = base_model_path.split("-")[-1]
            if size_of_llava == "34b":
                if not finetuned_model_path:
                    misclassification_matrix_path = os.path.join(subset_path, 'misclassification_matrix.csv')
                else:
                    misclassification_matrix_path = os.path.join(subset_path, f'misclassification_matrix_{finetuned_model_path}.csv')
            else:
                if not finetuned_model_path:
                    misclassification_matrix_path = os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix.csv')
                else:
                    misclassification_matrix_path = os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix_{finetuned_model_path}.csv')
        else:
            if not finetuned_model_path:
                misclassification_matrix_path = (os.path.join(subset_path, 'val_misclassification_matrix.csv'))
            else: 
                misclassification_matrix_path = (os.path.join(subset_path, f'val_misclassification_matrix_{"_".join(finetuned_model_path.split("/"))}.csv'))

        misclassification_matrix_loaded = pd.read_csv(misclassification_matrix_path)
        with open(misclassification_matrix_path, 'r') as file:
            reader = csv.reader(file)
            all_species = next(reader)
        misclassification_matrix = pd.DataFrame(0.0, index=all_species, columns=all_species)
        for species in all_species:
            for i in range(len(all_species)):
                misclassification_matrix[species][all_species[i]] = misclassification_matrix_loaded[species][i]
        return misclassification_matrix.to_dict(orient='dict')

    def get_confusing_classes(self, name, base_model_path = "liuhaotian/llava-v1.6-34b", threshold = 0.2, top_n = 1):
        significant_confusing_pairs = []
        subset_path = os.path.join(self.subsets_info_dir, name)
        size_of_llava = base_model_path.split("-")[-1]
        if size_of_llava == "34b":
            misclassification_matrix_path = os.path.join(subset_path, 'misclassification_matrix.csv')
        else:
            misclassification_matrix_path = os.path.join(subset_path, f'{size_of_llava}_misclassification_matrix.csv')

        misclassification_matrix_path = os.path.join(subset_path, 'misclassification_matrix.csv')
        with open(misclassification_matrix_path, 'r') as file:
            reader = csv.reader(file)
            all_species = next(reader)
        misclassification_matrix = pd.read_csv(misclassification_matrix_path)
        for species in all_species:
            misclassifications = []
            for misclassified_species, rate in misclassification_matrix[species][:].items():
                if all_species[misclassified_species] == species:
                    continue
                if rate > threshold:
                    misclassifications.append((misclassified_species, rate))
            misclassifications.sort(key=lambda x: x[1], reverse=True)
            if top_n:
                top_misclassifications = misclassifications[:top_n]
            else:
                top_misclassifications = misclassifications
            for misclassified_species, rate in top_misclassifications:
                significant_confusing_pairs.append({"ground_truth":species, "confusing_class":all_species[misclassified_species], "rate_of_error":rate})
        return significant_confusing_pairs

    def get_confusing_classes_in_all_subsets(self, threshold = 0.2):
        significant_confusing_pairs = []
        for name in self.get_list_of_subset_names():
            confusing_classes = self.get_confusing_classes(name, threshold=threshold, top_n = False)
            # Remove duplicates across subsets
            for confusing_class in confusing_classes:
                add = True
                for existing_pair in significant_confusing_pairs:
                    if existing_pair["ground_truth"] == confusing_class["ground_truth"] and existing_pair["confusing_class"] == confusing_class["confusing_class"]:
                        add = False
                        break
                if add:
                    significant_confusing_pairs.append(confusing_class)
        return significant_confusing_pairs
        
    def get_confusing_json_config_in_all_subsets(self, threshold = 0.2):
        significant_confusing_pairs = []
        for name in self.get_list_of_subset_names():
            confusing_classes = self.get_confusing_classes(name, threshold=threshold, top_n = False)
            dirs = self.get_all_species_in_subset_and_directories(name, prefix = False)
            # Remove duplicates across subsets
            for confusing_class in confusing_classes:
                add = True
                for existing_pair in significant_confusing_pairs:
                    if existing_pair["ground_truth"] == confusing_class["ground_truth"] and existing_pair["confusing_class"] == confusing_class["confusing_class"]:
                        add = False
                        break
                if add:
                    confusing_class["ground_truth_full_name"] = dirs[confusing_class["ground_truth"]]
                    confusing_class["confusing_class_full_name"] = dirs[confusing_class["confusing_class"]]
                    significant_confusing_pairs.append(confusing_class)
        return significant_confusing_pairs
    
    # def generate_images_for_subset(self, subset_name, prompting_method, significant_confusing_pairs, sd_ppl, label = "for34b", num_to_generate = 50): 
    #     # generate the images in the paths. In addition, store the prompting info
    #     subset_path = os.path.join(self.subsets_info_dir, subset_name)
    #     subset_gen_path = os.path.join(self.generated_images_path, subset_name)
    #     os.makedirs(subset_gen_path, exist_ok=True)
    #     sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')
    #     dir_for_label = os.path.join(subset_gen_path, label)
    #     os.makedirs(dir_for_label, exist_ok=True)
    #     with open(os.path.join(dir_for_label, "confusing_pairs.json"), 'w') as file:
    #         json.dump(significant_confusing_pairs, file)    
    #     if os.path.exists(sample_file_path):
    #         try:
    #             with open(sample_file_path, 'r') as sample_file:
    #                 species_list = json.load(sample_file)
    #         except Exception as e:
    #             print(f"An error occurred while reading the subset '{name}': {e}")
    #             raise e
    #     else:
    #         print(f"Subset '{name}' does not exist or 'sampled_species.json' is missing.")
    #         return
        
    #     for confusing_pair in significant_confusing_pairs:
    #         ground_truth = confusing_pair["ground_truth"]
    #         confusing_class = confusing_pair["confusing_class"]
    #         species_dir = species_list[ground_truth]
    #         image_dir_path = os.path.join(self.filtered_images_path, 'train', species_dir)
    #         try:
    #             images = os.listdir(image_dir_path)

    #             if not images:
    #                 print("No images found in the directory.")
    #             else:
    #                 sorted_images = sorted(images)
    #                 lowest_image = sorted_images[0]
    #                 print(f"The lowest image by name is: {lowest_image}")

    #         except FileNotFoundError:
    #             print("Directory not found. It seems the images have not been filtered yet.")
    #             continue
            
            
    #         dir_for_generation = os.path.join(subset_gen_path, )
    #         # convert prompting_method function to a string
    #         prompting_method_str = str(prompting_method).split(" ")[1]
    #         dir_for_generation = os.path.join(dir_for_generation, label, prompting_method_str, species_dir)

    #         os.makedirs(dir_for_generation, exist_ok=True)
    #         # if prompt_info file exists, load it. Otherwise, use prompting_method directly
    #         if os.path.exists(os.path.join(dir_for_generation, "prompt_info.json")):
    #             with open(os.path.join(dir_for_generation, "prompt_info.json"), 'r') as file:
    #                 prompt_info = json.load(file)
    #                 prompt = prompt_info['prompt']
    #                 negative_prompt = prompt_info['negative_prompt']
    #             print("Using saved prompt information from previous generations.")
    #         else:
    #             prompt, negative_prompt = prompting_method(os.path.join(image_dir_path, lowest_image), ground_truth, confusing_class)
            
    #         # Check the directory for generation, so that we skip already generated images. Obtain the largest image number
    #         largest_number = -1
    #         for filename in os.listdir(dir_for_generation):
    #             if filename.endswith(".png"):
    #                 number = int(filename.split('.')[0])
    #                 if number > largest_number:
    #                     largest_number = number
    #         for i in range(largest_number + 1, num_to_generate):
    #             image_path = os.path.join(dir_for_generation, f"{i}.png")
    #             generate_image(sd_ppl, prompt, negative_prompt, image_path, randomize_seed=True)
    #         # save prompting info
    #         prompt_info = {
    #             "ground_truth": ground_truth, 
    #             "confusing_class": confusing_class, 
    #             "image_path": lowest_image, 
    #             "prompt": prompt, 
    #             "negative_prompt": negative_prompt
    #         }
    #         with open(os.path.join(dir_for_generation, "prompt_info.json"), 'w') as file:
    #             json.dump(prompt_info, file)



    def run_classification_for_subset_on_validation(self, name, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None):
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


        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')

        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    all_species_dirs = json.load(sample_file)
                    all_species = list(all_species_dirs.keys())
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return
        args = type('Args', (), {
            "model_path": finetuned_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": base_model_path,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0,
            "top_p": None,
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
                "temperature": 0,
                "top_p": None,
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
        with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
            # Load the data from the file
            data = json.load(file)
        supercategory = name.split('_')[0]
        heirachy_of_classification = ["supercategory",'kingdom','phylum', 'class', 'order', 'family', 'genus', 'common_name']
        
        # Run llava
        misclassification_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        count_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        confusing_classes = self.get_confusing_classes(name)
        confusing_classes = [pair["ground_truth"] for pair in confusing_classes]
        for common_name in all_species:
            print(common_name)
            species_dir_name = all_species_dirs[common_name]
            images = os.listdir(os.path.join("/data2/mohant/new_val", species_dir_name))
            images = [os.path.join("/data2/mohant/new_val",species_dir_name, img) for img in images]
            # sort images by names in val/species_dir_name and /train/species_dir_name
            # train_dir = os.path.join(self.filtered_images_path, 'train', species_dir_name)
            # val_dir = os.path.join(self.filtered_images_path, 'val', species_dir_name)
            # images = [x for x in os.listdir(train_dir) if x.endswith('.jpg')]
            # val_images = [x for x in os.listdir(val_dir) if x.endswith('.jpg')]
            # images.extend(val_images)
            # images.sort()
            images = images[-20:]



            for file_name in images:
                for _ in range(1): # 7
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
                    # images[0].show()
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
                    # print(prompt)
                    # print(outputs)
                    outputs_lower = outputs.lower()
                    valid_output = False
                    for option in provided_options:
                        option_lower = option.lower()
                        count_matrix[common_name][option] = count_matrix[common_name][option] + 1
                        if outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) ) or (outputs_lower.startswith("this is a " + option_lower) ) or (outputs_lower.startswith("this is " + option_lower)) or  (outputs_lower.startswith("this is an " + option_lower) ):
                            misclassification_matrix[common_name][option] = misclassification_matrix[common_name][option]  + 1/20
                            valid_output = True
                    if not valid_output:
                        print("Error Output: " + outputs)
                        print("Prompt: " + prompt)
        if not finetuned_model_path:
            misclassification_matrix.to_csv(os.path.join(subset_path, 'val_misclassification_matrix.csv'), index=False)
        else: 
            misclassification_matrix.to_csv(os.path.join(subset_path, f'val_misclassification_matrix_{"_".join(finetuned_model_path.split("/"))}.csv'), index=False)
        del model
        import gc
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return misclassification_matrix



    def get_classification_for_pair_on_validation_as_dict(
        self, 
        name, 
        pair, 
        base_model_path="liuhaotian/llava-v1.6-34b", 
        finetuned_model_path=None
    ):
        """
        Retrieve the misclassification matrix for a specific pair on the validation set as a nested dictionary.

        :param name: The name of the subset.
        :param pair: A list or tuple containing the two species names.
        :param base_model_path: The base model path.
        :param finetuned_model_path: The finetuned model path, if any.
        :return: A nested dictionary representing the misclassification rates.
        """
        import os
        import pandas as pd
        import csv

        subset_path = os.path.join(self.subsets_info_dir, name)
        
        # Ensure that 'pair' contains exactly two species
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Parameter 'pair' must be a list or tuple containing exactly two species names.")
        
        class_name_1 = pair[0].replace(' ', '_')
        class_name_2 = pair[1].replace(' ', '_')

        # Create the filename component using the transformed class names
        pair_name_in_filename = f"{class_name_1}_vs_{class_name_2}"

        # Check if 'finetuned_model_path' is provided and construct the filename accordingly
        if not finetuned_model_path:
            # No finetuned model, use pair names in filename
            filename = f'pair_misclassification_matrix_{pair_name_in_filename}.csv'
        else:
            # Include finetuned model information in filename
            finetuned_model_name = "_".join(finetuned_model_path.split("/"))
            filename = f'pair_misclassification_matrix_{finetuned_model_name}_{pair_name_in_filename}.csv'

        misclassification_matrix_path = os.path.join(subset_path, filename)

        # Check if the misclassification matrix file exists
        if not os.path.exists(misclassification_matrix_path):
            raise FileNotFoundError(
                f"The misclassification matrix file '{misclassification_matrix_path}' does not exist."
            )

        try:
            # Read the CSV file using pandas
            misclassification_matrix_loaded = pd.read_csv(misclassification_matrix_path)
            
            # Extract the list of species from the first row
            with open(misclassification_matrix_path, 'r') as file:
                reader = csv.reader(file)
                all_species = next(reader)
            
            # Initialize an empty DataFrame with species as both index and columns
            misclassification_matrix = pd.DataFrame(0.0, index=all_species, columns=all_species)
            
            # Populate the DataFrame with values from the loaded CSV
            for species in all_species:
                for i in range(len(all_species)):
                    misclassification_matrix.at[species, all_species[i]] = misclassification_matrix_loaded.at[i, species]
            
            # Convert the DataFrame to a nested dictionary
            misclassification_dict = misclassification_matrix.to_dict(orient='index')
            
            return misclassification_dict

        except Exception as e:
            print(f"An error occurred while processing the misclassification matrix: {e}")
            return {}



    def run_classification_for_pair_on_validation(self, name, pair, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None, filtered_images_dir = "/data2/mohant/new_val"):
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

        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')
        class_name_1 = pair[0].replace(' ', '_')
        class_name_2 = pair[1].replace(' ', '_')

        # Create the filename component using the transformed class names
        pair_name_in_filename = f"{class_name_1}_vs_{class_name_2}"

        # Check if 'finetuned_model_path' is provided

        
        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    all_species_dirs = json.load(sample_file)
                    all_species = list(all_species_dirs.keys())
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return

        if not finetuned_model_path:
            # No finetuned model, use pair names in filename
            filename = f'pair_misclassification_matrix_{pair_name_in_filename}.csv'
        else:
            # Include finetuned model information in filename
            finetuned_model_name = "_".join(finetuned_model_path.split("/"))
            filename = f'pair_misclassification_matrix_{finetuned_model_name}_{pair_name_in_filename}.csv'

        if os.path.exists(os.path.join(subset_path, filename)):
            print(f"Misclassification matrix already exists at {os.path.join(subset_path, filename)}")
            return
        
        args = type('Args', (), {
            "model_path": finetuned_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": base_model_path,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0,
            "top_p": None,
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
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
        if not finetuned_model_path:
            model_name = get_model_name_from_path(args.model_path)
        else:
            if "34b" in base_model_path:
                model_name = "llava-v1.6-34b-lora"
            else:
                model_name = "llava-v1.6-vicuna-13b-lora"


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
        with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
            # Load the data from the file
            data = json.load(file)
        supercategory = name.split('_')[0]
        heirachy_of_classification = ["supercategory",'kingdom','phylum', 'class', 'order', 'family', 'genus', 'common_name']
        
        # Run llava
        all_species = pair
        # all_species = ["Indian Palm Squirrel", "Round-tailed Ground Squirrel"]
        misclassification_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        count_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        confusing_classes = self.get_confusing_classes(name)
        confusing_classes = [p["ground_truth"] for p in confusing_classes]
        image_correctness = {}
        for common_name in all_species:
            print(common_name)
            species_dir_name = all_species_dirs[common_name]
            images = os.listdir(os.path.join(filtered_images_dir, species_dir_name))
            images = [os.path.join(filtered_images_dir,species_dir_name, img) for img in images]
            # sort images by names in val/species_dir_name and /train/species_dir_name
            # train_dir = os.path.join(self.filtered_images_path, 'train', species_dir_name)
            # val_dir = os.path.join(self.filtered_images_path, 'val', species_dir_name)
            # images = [x for x in os.listdir(train_dir) if x.endswith('.jpg')]
            # val_images = [x for x in os.listdir(val_dir) if x.endswith('.jpg')]
            # images.extend(val_images)
            # images.sort()
            images = images[-20:]



            for file_name in images:
                for _ in range(1): # 7
                    provided_options = random.sample(all_species, len(all_species))
                    provided_options_capitalized = [p.capitalize() for p in provided_options]
                    args.query = "Is this " + ", ".join(provided_options_capitalized[:-1])+", or " + provided_options_capitalized[-1] + "? Directly answer the question using exactly one of the options."
                    # args.query = "Is this " + ", ".join(provided_options[:-1])+", or " + provided_options[-1] + "? Directly answer the question using exactly one of the options."
                    # args.query = "Which option best represents the image? " + ", ".join(provided_options_capitalized[:-1])+" OR " + provided_options_capitalized[-1] + "."
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
                    # images[0].show()
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
                    # print(prompt)
                    # print(outputs)
                    outputs_lower = outputs.lower()
                    valid_output = False
                    for option in provided_options:
                        option_lower = option.lower()
                        count_matrix[common_name][option] = count_matrix[common_name][option] + 1
                        if outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) ) or (outputs_lower.startswith("this is a " + option_lower) ) or (outputs_lower.startswith("this is " + option_lower)) or  (outputs_lower.startswith("this is an " + option_lower) ) or (outputs_lower.startswith("the image shows a " + option_lower)) or (outputs_lower.startswith("the image shows an " + option_lower)) or (outputs_lower.startswith("the image shows " + option_lower)):
                            misclassification_matrix[common_name][option] = misclassification_matrix[common_name][option]  + 1/20
                            valid_output = True
                            image_correctness[file_name] = option
                    if not valid_output:
                        print("Error Output: " + outputs)
                        print("Prompt: " + prompt)

        # Save the misclassification matrix
        misclassification_matrix.to_csv(os.path.join(subset_path, filename), index=False)

        with open("image_by_image.json", "w") as f:
            json.dump(image_correctness, f)

        del model
        import gc
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return misclassification_matrix








    def run_classification_for_pair_on_train(self, name, pair, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None):
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


        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')

        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    all_species_dirs = json.load(sample_file)
                    all_species = list(all_species_dirs.keys())
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return
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
            if "34b" in base_model_path:
                model_name = "llava-v1.6-34b-lora"
            else:
                model_name = "llava-v1.6-vicuna-13b-lora"


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
        with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
            # Load the data from the file
            data = json.load(file)
        supercategory = name.split('_')[0]
        heirachy_of_classification = ["supercategory",'kingdom','phylum', 'class', 'order', 'family', 'genus', 'common_name']
        
        # Run llava
        all_species = pair
        # all_species = ["Indian Palm Squirrel", "Round-tailed Ground Squirrel"]
        misclassification_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        count_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        confusing_classes = self.get_confusing_classes(name)
        confusing_classes = [p["ground_truth"] for p in confusing_classes]
        image_correctness = {}
        for common_name in all_species:
            print(common_name)
            species_dir_name = all_species_dirs[common_name]
            if common_name in confusing_classes:
                images = os.listdir(os.path.join("/local1/VDA/filtered_images_inat/new_train", species_dir_name))
                images = [os.path.join("/local1/VDA/filtered_images_inat/new_train",species_dir_name, img) for img in images]
                # sort images by names in val/species_dir_name and /train/species_dir_name
                # train_dir = os.path.join(self.filtered_images_path, 'train', species_dir_name)
                # val_dir = os.path.join(self.filtered_images_path, 'val', species_dir_name)
                # images = [x for x in os.listdir(train_dir) if x.endswith('.jpg')]
                # val_images = [x for x in os.listdir(val_dir) if x.endswith('.jpg')]
                # images.extend(val_images)
                # images.sort()
                images = images[-100:]
            else:
                images = os.listdir(os.path.join(self.root_path, 'train_mini', species_dir_name))
                images = [os.path.join(self.root_path, 'train_mini', species_dir_name, img) for img in images]
                images = images[-100:]
            # images = 



            for file_name in images:
                for _ in range(1): # 7
                    provided_options = random.sample(all_species, len(all_species))
                    provided_options_capitalized = [p.capitalize() for p in provided_options]
                    args.query = "Is this " + ", ".join(provided_options_capitalized[:-1])+", or " + provided_options_capitalized[-1] + "? Directly answer the question using exactly one of the options."
                    # args.query = "Is this " + ", ".join(provided_options[:-1])+", or " + provided_options[-1] + "? Directly answer the question using exactly one of the options."
                    # args.query = "Which option best represents the image? " + ", ".join(provided_options_capitalized[:-1])+" OR " + provided_options_capitalized[-1] + "."
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
                    # images[0].show()
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
                    # print(prompt)
                    # print(outputs)
                    outputs_lower = outputs.lower()
                    valid_output = False
                    for option in provided_options:
                        option_lower = option.lower()
                        count_matrix[common_name][option] = count_matrix[common_name][option] + 1
                        if outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) ) or (outputs_lower.startswith("this is a " + option_lower) ) or (outputs_lower.startswith("this is " + option_lower)) or  (outputs_lower.startswith("this is an " + option_lower) ) or (outputs_lower.startswith("the image shows a " + option_lower)) or (outputs_lower.startswith("the image shows an " + option_lower)) or (outputs_lower.startswith("the image shows " + option_lower)):
                            misclassification_matrix[common_name][option] = misclassification_matrix[common_name][option]  + 1/100
                            valid_output = True
                            image_correctness[file_name] = option
                    if not valid_output:
                        print("Error Output: " + outputs)
                        print("Prompt: " + prompt)

        class_name_1 = pair[0].replace(' ', '_')
        class_name_2 = pair[1].replace(' ', '_')

        # Create the filename component using the transformed class names
        pair_name_in_filename = f"{class_name_1}_vs_{class_name_2}"

        # Check if 'finetuned_model_path' is provided
        if not finetuned_model_path:
            # No finetuned model, use pair names in filename
            filename = f'train_pair_misclassification_matrix_{pair_name_in_filename}.csv'
        else:
            # Include finetuned model information in filename
            finetuned_model_name = "_".join(finetuned_model_path.split("/"))
            filename = f'train_pair_misclassification_matrix_{finetuned_model_name}_{pair_name_in_filename}.csv'

        # Save the misclassification matrix
        misclassification_matrix.to_csv(os.path.join(subset_path, filename), index=False)

        with open("image_by_image.json", "w") as f:
            json.dump(image_correctness, f)

        del model
        import gc
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return misclassification_matrix


    def run_classification_for_general_pair_on_validation(self, pair, pair_dirs, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None, num_val = 20):
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
        # all_species = ["Indian Palm Squirrel", "Round-tailed Ground Squirrel"]
        misclassification_matrix = {class_1: {class_1: 0, class_2: 0}, class_2: {class_1: 0, class_2: 0}}
        for i in range(2):
            common_name = pair[i]
            species_dir_name = pair_dirs[i]
            images = os.listdir(species_dir_name)
            images = [os.path.join(species_dir_name, img) for img in images]
            images = images[-num_val:]

            for file_name in images:
                provided_options = random.sample(all_species, len(all_species))
                provided_options_capitalized = [p.capitalize() for p in provided_options]
                args.query = "Is this " + ", ".join(provided_options_capitalized[:-1])+", or " + provided_options_capitalized[-1] + "? Directly answer the question using exactly one of the options."
                # args.query = "Is this " + ", ".join(provided_options[:-1])+", or " + provided_options[-1] + "? Directly answer the question using exactly one of the options."
                # args.query = "Which option best represents the image? " + ", ".join(provided_options_capitalized[:-1])+" OR " + provided_options_capitalized[-1] + "."
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

        # Save the misclassification matrix
        del model
        import gc
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return misclassification_matrix

    def left_pad_sequences(sequences, padding_value=0):
        # Reverse sequences
        sequences = [seq.flip(0) for seq in sequences]
        # Pad sequences
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        # Reverse sequences back
        if batch_first:
            padded_sequences = padded_sequences.flip(1)
        else:
            padded_sequences = padded_sequences.flip(0)
        return padded_sequences


    def run_classification_for_pair_on_validation_batched(self, name, pair, base_model_path = "liuhaotian/llava-v1.6-34b", finetuned_model_path = None, filtered_images_dir = "/data2/mohant/new_val"):
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

        subset_path = os.path.join(self.subsets_info_dir, name)
        sample_file_path = os.path.join(subset_path, 'sampled_species_dirs.json')
        class_name_1 = pair[0].replace(' ', '_')
        class_name_2 = pair[1].replace(' ', '_')

        # Create the filename component using the transformed class names
        pair_name_in_filename = f"{class_name_1}_vs_{class_name_2}"

        # Check if 'finetuned_model_path' is provided

        
        if os.path.exists(sample_file_path):
            try:
                with open(sample_file_path, 'r') as sample_file:
                    all_species_dirs = json.load(sample_file)
                    all_species = list(all_species_dirs.keys())
            except Exception as e:
                print(f"An error occurred while reading the subset '{name}': {e}")
                return

        if not finetuned_model_path:
            # No finetuned model, use pair names in filename
            filename = f'pair_misclassification_matrix_{pair_name_in_filename}.csv'
        else:
            # Include finetuned model information in filename
            finetuned_model_name = "_".join(finetuned_model_path.split("/"))
            filename = f'pair_misclassification_matrix_{finetuned_model_name}_{pair_name_in_filename}.csv'

        if os.path.exists(os.path.join(subset_path, filename)):
            print(f"Misclassification matrix already exists at {os.path.join(subset_path, filename)}")
            return
        
        args = type('Args', (), {
            "model_path": finetuned_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": base_model_path,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0.2,
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
                "temperature": 0.2,
                "top_p": 1.0,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
        if not finetuned_model_path:
            model_name = get_model_name_from_path(args.model_path)
        else:
            if "34b" in base_model_path:
                model_name = "llava-v1.6-34b-lora"
            else:
                model_name = "llava-v1.6-vicuna-13b-lora"


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
        with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
            # Load the data from the file
            data = json.load(file)
        supercategory = name.split('_')[0]
        heirachy_of_classification = ["supercategory",'kingdom','phylum', 'class', 'order', 'family', 'genus', 'common_name']
        
        # Run llava
        all_species = pair
        # all_species = ["Indian Palm Squirrel", "Round-tailed Ground Squirrel"]
        misclassification_matrix = pd.DataFrame(0, index=all_species, columns=all_species)
        confusing_classes = self.get_confusing_classes(name)
        confusing_classes = [p["ground_truth"] for p in confusing_classes]
        image_correctness = {}
        for common_name in all_species:
            print(common_name)
            species_dir_name = all_species_dirs[common_name]
            images = os.listdir(os.path.join(filtered_images_dir, species_dir_name))
            images = [os.path.join(filtered_images_dir,species_dir_name, img) for img in images]
            # sort images by names in val/species_dir_name and /train/species_dir_name
            # train_dir = os.path.join(self.filtered_images_path, 'train', species_dir_name)
            # val_dir = os.path.join(self.filtered_images_path, 'val', species_dir_name)
            # images = [x for x in os.listdir(train_dir) if x.endswith('.jpg')]
            # val_images = [x for x in os.listdir(val_dir) if x.endswith('.jpg')]
            # images.extend(val_images)
            # images.sort()
            images = images[-20:]


            batch_size = 20
            for i in range(0, len(images), batch_size):
                batch_files = images[i:i+batch_size]
                batch_prompts = []
                batch_images = []
                batch_image_sizes = []
                batch_file_names = []
                batch_input_ids = []
                batch_attention_masks = []


                for file_name in batch_files:
                    provided_options = random.sample(all_species, len(all_species))
                    provided_options_capitalized = [p.capitalize() for p in provided_options]
                    args.query = "Is this " + ", ".join(provided_options_capitalized[:-1])+", or " + provided_options_capitalized[-1] + "? Directly answer the question using exactly one of the options."
                    # args.query = "Is this " + ", ".join(provided_options[:-1])+", or " + provided_options[-1] + "? Directly answer the question using exactly one of the options."
                    # args.query = "Which option best represents the image? " + ", ".join(provided_options_capitalized[:-1])+" OR " + provided_options_capitalized[-1] + "."
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
                    batch_prompts.append(prompt)
                    batch_file_names.append(file_name)

                    image_files = image_parser(args)
                    images = load_images(image_files)
                    # images[0].show()
                    image_sizes = [x.size for x in images]
                    batch_images.extend(images)
                    batch_image_sizes.extend(image_sizes)

                images_tensor = process_images(
                    batch_images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
                
                for prompt in batch_prompts:
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    )
                    attention_mask = torch.ones_like(input_ids)
                    batch_input_ids.append(input_ids)
                    batch_attention_masks.append(attention_mask)
                
                batch_input_ids = left_pad_sequences(batch_input_ids, padding_value=tokenizer.pad_token_id)
                batch_attention_masks = left_pad_sequences(batch_attention_masks, padding_value=0)
                batch_input_ids = batch_input_ids.cuda()
                batch_attention_masks = batch_attention_masks.cuda()

                
                with torch.inference_mode():
                    output_ids = model.generate(
                        batch_input_ids,
                        attention_mask=batch_attention_masks,
                        images=images_tensor,
                        image_sizes=batch_image_sizes,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # print(prompt)
                # print(outputs)
                for output in outputs:
                    outputs_lower = output.strip().lower()
                    valid_output = False
                    for option in provided_options:
                        option_lower = option.lower()
                        if outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) ) or (outputs_lower.startswith("this is a " + option_lower) ) or (outputs_lower.startswith("this is " + option_lower)) or  (outputs_lower.startswith("this is an " + option_lower) ) or (outputs_lower.startswith("the image shows a " + option_lower)) or (outputs_lower.startswith("the image shows an " + option_lower)) or (outputs_lower.startswith("the image shows " + option_lower)):
                            misclassification_matrix[common_name][option] = misclassification_matrix[common_name][option]  + 1/20
                            valid_output = True
                            image_correctness[file_name] = option
                    if not valid_output:
                        print("Error Output: " + outputs)
                        print("Prompt: " + prompt)

        # Save the misclassification matrix
        misclassification_matrix.to_csv(os.path.join(subset_path, filename), index=False)

        with open("image_by_image.json", "w") as f:
            json.dump(image_correctness, f)

        del model
        import gc
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return misclassification_matrix

    def run_classification_on_novel_class(self, supercategory, class_name, num_samples = 35, subset_size = 14, base_model_path = "liuhaotian/llava-v1.6-34b"):
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

        def is_sub_species(species, higher_rank_classification):
            for rank in higher_rank_classification:
                if not higher_rank_classification[rank] == None:
                    if not higher_rank_classification[rank] == species[rank]:
                        return False
            return True

        def sampling_from_species(species, n_samples):
            target_level = 'common_name'
            for i in range(len(heirachy_of_classification)):
                if species[heirachy_of_classification[i]] == None:
                    target_level = heirachy_of_classification[i-1]
                    break
            count = 0
            index = 0
            samples = []
            indices = random.sample(range(len( data['annotations'])), len( data['annotations']))
            while count < n_samples:
                index_original = indices[index]
                annotation = data['annotations'][index_original]
                category_id = annotation["category_id"]
                category = data["categories"][category_id]
                if is_sub_species(category, species):
                    samples.append(index_original)
                    count = count + 1
                index = index + 1
            return samples

        novel_class_path = "/data2/mohant/novel_dataset/dataset"

        size_of_llava = base_model_path.split("-")[-1]
        args = type('Args', (), {
            "model_path": base_model_path,#"model_path": "liuhaotian/llava-v1.6-34b",
            "model_base": None,
            "model_name": None,
            "query": "What are the things I should be cautious about when I visit here?",
            "conv_mode": None,
            "image_file": "https://llava-vl.github.io/static/images/view.jpg",
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        model_name = get_model_name_from_path(args.model_path)
        with open(os.path.join(self.root_path , 'train_mini.json'), 'r') as file:
        # Load the data from the file
            data = json.load(file)
        supercategory_cleaned = supercategory.replace("_", " ")
        all_species = list({category["common_name"] for category in data["categories"] if category["supercategory"] == supercategory_cleaned})
        all_species = random.sample(all_species, subset_size)
        all_species_dirs = {}
        for common_name in all_species:
            for category in data["categories"]:
                if category["supercategory"] == supercategory_cleaned:
                    if category["common_name"] == common_name:
                        all_species_dirs[common_name] = (category["image_dir_name"])
                        break


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
        heirachy_of_classification = ["supercategory",'kingdom','phylum', 'class', 'order', 'family', 'genus', 'common_name']
        common_name = ' '.join(class_name.split('_'))
        all_species.append(common_name)
        # Run llava
        misclassification_matrix = {}
        for n in all_species:
            misclassification_matrix[n] = 0
        for folder_name in [class_name]:
            file_names = os.listdir(os.path.join(novel_class_path, supercategory, folder_name))
            if (len(file_names) < num_samples):
                print(f"Not enough number of files for {class_name}")
            for file_name in file_names[:num_samples]:
                # print("Processing image: " + str(image_id))
                file_name = os.path.join(novel_class_path, supercategory, folder_name, file_name)
                # args.query = "Is this " + ", ".join(provided_options)+"? Answer the question using a few words."
                for _ in range(1): # 7
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
                    # images[0].show()
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
                    # print(prompt)
                    # print(outputs)
                    outputs_lower = outputs.lower()
                    valid_output = False
                    for option in provided_options:
                        option_lower = option.lower()
                        def remove_parentheses(s):
                            """Removes parentheses from the string."""
                            return s.replace("(", "").replace(")", "")

                        def remove_bracketed_elements(s):
                            """Removes the bracketed elements from the string."""
                            return re.sub(r'\s*\(.*?\)\s*', ' ', s).strip()

                        # Existing code snippet with small adjustments
                        option_lower_clean = remove_parentheses(option_lower)
                        option_lower_no_bracket = remove_bracketed_elements(option_lower)

                        # Check for matches with and without the bracketed elements
                        if (outputs_lower.startswith(option_lower_clean) or
                            outputs_lower.startswith("a " + option_lower_clean) or
                            outputs_lower.startswith(option_lower_no_bracket) or
                            outputs_lower.startswith("a " + option_lower_no_bracket)
                            or outputs_lower.startswith("" + option_lower) or (outputs_lower.startswith("a " + option_lower) )):

                            misclassification_matrix[option] = misclassification_matrix[option] + 1.0/num_samples
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
        return misclassification_matrix,all_species_dirs
