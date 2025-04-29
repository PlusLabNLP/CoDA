import os
import json
import copy
import random

class ConfusingDataset:
    def __init__(self, data_dir, synthetic_imgs_num=5, real_imgs_num=5, seed=0):
        config_path = f"{data_dir}/config.json"
        self.config = json.load(open(config_path, "r"))
        self.num_images_per_class = self.config["num_imgs_per_class"]
        self.models = self.config["models"]
        self.prompts_types = self.config["prompts_types"]
        self.synthetic_images_path = self.config["synthetic_images_path"]
        self.real_images_path = self.config["real_images_path"]
        self.val_images_path = self.config["val_images_path"]
        self.pairs = self.config["pairs"]
        self.synthetic_imgs_num = synthetic_imgs_num
        self.real_imgs_num = real_imgs_num
        self.seed = seed
        random.seed(self.seed)
        
    def __len__(self):
        return len(self.pairs) * self.num_images_per_class
    
    def __get__pairs__(self):
        return self.pairs
    
    def __get_prompts_types__(self):
        return self.prompts_types
    
    def train_prompt_assemble(self, id, main_class, confusing_class, image_path, correct_class):
        example = {
            "id": 45,
            "image": "/home/bingxuan/bingxuan/data/object_data/SUN397/c/campus/sun_adiqdyqsqarvtact.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "You are an image classification specialist with expertise in categorizing images into specific groups. Given an image, identify its category from the following options: A or B. Provide your answer as only one category name for precise classification. Please response with the category name only."
                },
                {
                    "from": "gpt",
                    "value": "campus"
                }
            ]
        }
        main_class = main_class.replace("_", " ").capitalize()
        confusing_class = confusing_class.replace("_", " ").capitalize()
        answer = main_class if correct_class == 0 else confusing_class
        classes = [main_class, confusing_class]
        random.shuffle(classes)
        provided_options_capitalized = classes
        template = copy.deepcopy(example)
        template["id"] = id
        template["image"] = image_path
        template["conversations"][0]["value"] = f"You are an image classification specialist with expertise in categorizing images into specific groups. Given an image, identify its category from the following options: " + ", ".join(provided_options_capitalized[:-1]) + ", or " + provided_options_capitalized[-1] + ". Provide your answer as only one category name for precise classification. Please response with the category name only."
        template["conversations"][1]["value"] = answer
        return template
      
        
    def prepare_finetune_data(self, prompt_type, model):
        
        num_real_images = self.real_imgs_num
        num_synthetic_images = self.synthetic_imgs_num
    
        train_data = []
        id_count = 0
        for pair in self.pairs:
            main_class = pair["ground_truth"]
            confusing_class = pair["confusing_class"]
            
            real_main_class_path = self.real_images_path + pair["ground_truth_full_name"]
            real_confusing_class_path = self.real_images_path + pair["confusing_class_full_name"]
            real_main_class_images = [img for img in os.listdir(real_main_class_path) if img.endswith(".jpg") or img.endswith(".png")]
            real_confusing_class_images = [img for img in os.listdir(real_confusing_class_path) if img.endswith(".jpg") or img.endswith(".png")]
            
            main_class_real_images = random.sample(real_main_class_images, num_real_images)
            confusing_class_real_images = random.sample(real_confusing_class_images, num_real_images)
            
            for i, image in enumerate(main_class_real_images):
                image_path = os.path.join(real_main_class_path, image)
                train_data.append(self.train_prompt_assemble(id_count, main_class, confusing_class, image_path, 0))
                id_count += 1
            
            for i, image in enumerate(confusing_class_real_images):
                image_path = os.path.join(real_confusing_class_path, image)
                train_data.append(self.train_prompt_assemble(id_count, main_class, confusing_class, image_path, 1))
                id_count += 1
                
            if num_synthetic_images > 0:    
                
                synthetic_main_class_path = self.synthetic_images_path +"/"+ model + "/" + prompt_type +"/" + pair["ground_truth_full_name"]
                synthetic_confusing_class_path = self.synthetic_images_path +"/"+ model + "/" + prompt_type +"/" + pair["confusing_class_full_name"]
                        
                synthetic_main_class_images = os.listdir(synthetic_main_class_path)
                synthetic_main_class_images = [img for img in synthetic_main_class_images if img.endswith(".jpg") or img.endswith(".png")]
                main_class_synthetic_images = random.sample(synthetic_main_class_images, num_synthetic_images)
                main_class_synthetic_images = [img for img in main_class_synthetic_images if img.endswith(".jpg") or img.endswith(".png")]
                
                for i, image in enumerate(main_class_synthetic_images):
                    image_path = os.path.join(synthetic_main_class_path, image)
                    train_data.append(self.train_prompt_assemble(id_count, main_class, confusing_class, image_path, 0))
                    id_count += 1
                
                synthetic_confusing_class_images = os.listdir(synthetic_confusing_class_path)
                synthetic_confusing_class_images = [img for img in synthetic_confusing_class_images if img.endswith(".jpg") or img.endswith(".png")]
                confusing_class_synthetic_images = random.sample(synthetic_confusing_class_images, num_synthetic_images)
                confusing_class_synthetic_images = [img for img in confusing_class_synthetic_images if img.endswith(".jpg") or img.endswith(".png")]
                for i, image in enumerate(confusing_class_synthetic_images):
                    image_path = os.path.join(synthetic_confusing_class_path, image)
                    train_data.append(self.train_prompt_assemble(id_count, main_class, confusing_class, image_path, 1))
                    id_count += 1
                
        random.shuffle(train_data)
        return train_data
    