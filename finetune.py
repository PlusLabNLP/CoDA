import os
import json
import time
import random
random.seed(0)

from utils.ConfusingDataset import ConfusingDataset

def finetune_run(working_dir, data_dir, number_of_images, number_of_epochs, prompt_type, port, base_model, seed, synthetic_imgs_num, real_imgs_num):
    
    confusing_dataset = ConfusingDataset(data_dir, synthetic_imgs_num, real_imgs_num, seed)
    num_pairs = len(confusing_dataset.pairs)
    print("##############################################")
    print("Dataset information:")
    print(f"Number of images per class: {confusing_dataset.num_images_per_class}")
    print(f"Prompts types: {confusing_dataset.prompts_types}")
    print(f"Synthetic images path: {confusing_dataset.synthetic_images_path}")
    print(f"Real images path: {confusing_dataset.real_images_path}")
    print(f"Validation images path: {confusing_dataset.val_images_path}")
    print(f"Pairs: {confusing_dataset.pairs}")
    print(f"Number of pairs: {num_pairs}")
    print(f"Seed: {seed}")
    print("##############################################\n")
    
    print("##############################################")
    print("Running finetune for the following parameters:")
    print(f"number_of_images: {number_of_images}")
    print(f"number_of_epochs: {number_of_epochs}")
    print(f"synthetic_imgs_num: {synthetic_imgs_num}")
    print(f"real_imgs_num: {real_imgs_num}")
    print(f"prompt_type: {prompt_type}")
    print(f"port: {port}")
    print(f"base_model: {base_model}")
    print("##############################################\n")
    
    model = base_model.split("-")[-1]
    exp_name = f"{model}_{prompt_type}_{synthetic_imgs_num}_{real_imgs_num}_{seed}"
    os.makedirs(f"{working_dir}/finetune_images/{exp_name}", exist_ok=True)
    os.makedirs(f"{working_dir}/ckpts/{exp_name}", exist_ok=True)
    os.makedirs(f"{working_dir}/logs/{exp_name}", exist_ok=True)
    
    
    for num_images in number_of_images:
        train_data = confusing_dataset.prepare_finetune_data(prompt_type, model)
        with open(f"{working_dir}/finetune_images/{exp_name}/train_data.json", "w") as f:
            json.dump(train_data, f)
        for num_epochs in number_of_epochs:
            print("----------------------------------------------")
            print(f"Finetuning with {num_images} images for {num_epochs} epochs")
            print("----------------------------------------------")
            MODEL_NAME = f"{working_dir}/ckpts/{exp_name}"
            PORT = port
            MASTER_PORT = "126" + str(PORT.split(",")[0]) + str(random.randint(0, 9))
            os.system(f"screen -L -Logfile {working_dir}/logs/{exp_name}/{num_images}_{num_epochs}.log -dmS {exp_name}_{num_images}_{num_epochs} sh ./llava_ft/train.sh {working_dir}/finetune_images/{exp_name}/train_data.json {MODEL_NAME} {num_epochs} {num_images} {PORT} {MASTER_PORT} {working_dir}/finetune_images/train_data.json {base_model}")
            start_time = time.time()
            while True:
                if os.path.exists(f"{MODEL_NAME}/README.md"):
                    break
                time.sleep(10)
                if time.time() - start_time > 5 * 60 * 60:
                    print("Finetuning takes more than 5 hours, quitting!")
                    os.system(f"screen -S {exp_name}_{num_images}_{num_epochs} -X quit")
                    break
            print("Finished finetuning!")
            print(f"Total time taken: {time.time() - start_time} seconds\n\n")
            
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="YOUR_DATA_PATH")
    parser.add_argument("--working_dir", type=str, default="YOUR_WORKING_DIR")
    parser.add_argument("--port", type=str, default="4,5,6")
    parser.add_argument("--base_model", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--number_of_images", type=int, default=5)
    parser.add_argument("--number_of_epochs", type=int, default=30)
    parser.add_argument("--synthetic_imgs_num", type=int, default=5)
    parser.add_argument("--real_imgs_num", type=int, default=5)
    parser.add_argument("--prompt_types", type=str, default="contrastive_visual,visual,text")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    working_dir = args.working_dir
    data_path = args.data_path
    port = args.port
    base_model = args.base_model
    number_of_images = [args.number_of_images]
    number_of_epochs = [args.number_of_epochs]
    synthetic_imgs_num = args.synthetic_imgs_num
    real_imgs_num = args.real_imgs_num
    prompt_types = args.prompt_types.split(",")
    seed = args.seed
    
    # approachs = ["contrastive_visual",
    #              "visual", 
    #              "contrastive_text", 
    #              "text",
    #              "contrastive_visual_text", 
    #              "visual_text",
    #              "crop",
    #              "flip",
    #              "armanda"]
    
    for prompt_type in prompt_types:
        for synthetic_ratio in synthetic_ratio:
            finetune_run(working_dir, data_path, number_of_images, number_of_epochs, prompt_type, port, base_model, seed, synthetic_imgs_num, real_imgs_num)
            time.sleep(10)
                
    print("Finished running all experiments!")