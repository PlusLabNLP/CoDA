import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm 
from SubsetManager.SubsetManager import SubsetManager

def filter_subset(subset_name, top_n):
    s = SubsetManager()
    species_dirs = s.get_all_species_in_subset_and_directories(subset_name)
    prefix = "/data2/mohant/inat/train_mini/"
    updated_animal_dict = {key: value.replace(prefix, '', 1) for key, value in species_dirs.items()}
    confusing_classes = s.get_confusing_classes(subset_name, top_n = None)
    class_considered = set()
    for dict_item in confusing_classes:
        class_considered.add(dict_item['ground_truth'])
        class_considered.add(dict_item['confusing_class'])
    for key, value in updated_animal_dict.items():
        if key in class_considered:
            print(subset_name)
            filter_one(input_folder = "/local1/VDA/train/" + value, output_folder = "/local1/VDA/auto_filter_images/" + value, top_n = top_n, common_name = key)

def verify_subset(subset_name, top_n):
    s = SubsetManager()
    # Get all species in the subset along with their directories
    species_dirs = s.get_all_species_in_subset_and_directories(subset_name)
    prefix = "/data2/mohant/inat/train_mini/"
    # Update the paths by removing the prefix
    updated_animal_dict = {key: value.replace(prefix, '', 1) for key, value in species_dirs.items()}
    confusing_classes = s.get_confusing_classes(subset_name, top_n = None)
    class_considered = set()
    for dict_item in confusing_classes:
        class_considered.add(dict_item['ground_truth'])
        class_considered.add(dict_item['confusing_class'])
    
    # Set the base directory where the files have been moved
    base_dir = f"/local1/VDA/filtered/val/"
    
    # Initialize a list to keep track of missing classes
    missing_classes = []
    
    # Iterate over the expected classes and check if their directories exist
    for key, value in updated_animal_dict.items():
        if key in class_considered:
            class_dir = os.path.join(base_dir, value)
            if not os.path.isdir(class_dir):
                # If the directory does not exist, add the class to the missing list
                missing_classes.append((key, class_dir))
    
    # Report the results
    if not missing_classes:
        print(f"All classes exist in the '{subset_name}' subset.")
    else:
        print(f"The following classes are missing in the '{subset_name}' subset:")
        for class_name, class_dir in missing_classes:
            print(f"- {class_name}: {class_dir}")


def filter_one(input_folder, output_folder, top_n, common_name):
    # ---------------------------
    # Configuration
    # ---------------------------

    # List of text prompts to compare with
    text_prompts = [
        "a photo of an animal",
        f"a photo of a {common_name}"
    ]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # Load Model and Processor
    # ---------------------------
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Prepare text inputs once
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        text_embeddings = model.get_text_features(**text_inputs)
        # Normalize text embeddings
        text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)

    # Lists to store image paths and their logits
    logits_list = []
    processed_image_files = []

    # Get list of image files
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

    print(f"Found {len(image_files)} images in {input_folder}.")

    # Iterate through images with a progress bar
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        try:
            # Open image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue

        # Process image
        try:
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                image_embeddings = model.get_image_features(**inputs)
                # Normalize image embeddings
                image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
                # Compute cosine similarity
                cosine_sim = (image_embeddings @ text_embeddings.T).squeeze(0)  # Shape: [num_texts]
                logits_per_image = cosine_sim * 100  # Scale by 100 as in CLIP's default

                logits_list.append(logits_per_image.cpu())
                processed_image_files.append(image_file)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue
        finally:
            # Free up memory
            del image, inputs, image_embeddings, cosine_sim, logits_per_image
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if not logits_list:
        print("No logits were computed. Exiting.")
        return

    # Stack all logits: Shape [num_images, num_texts]
    all_logits = torch.stack(logits_list)  # Shape: [N, T]

    # ---------------------------
    # Post-processing
    # ---------------------------
    # For each text, perform softmax across images (dim=0)
    softmax = torch.nn.Softmax(dim=0)
    all_softmax = softmax(all_logits)  # Shape: [N, T]

    # Sum softmax scores across texts for each image
    image_scores = all_softmax.prod(dim=1)  # Shape: [N]

    # Sort images based on scores in descending order
    sorted_indices = torch.argsort(image_scores, descending=True)
    sorted_scores, sorted_indices = image_scores[sorted_indices], sorted_indices

    # ---------------------------
    # Copy Top N Images
    # ---------------------------
    top_n = min(top_n, len(sorted_scores))
    print(f"\nTop {top_n} images:")

    for i in range(top_n):
        idx = sorted_indices[i].item()
        image_file = processed_image_files[idx]
        score = sorted_scores[idx].item()
        print(f"{i+1}: {image_file} with score {score:.4f}")
        # Copy to output folder
        src_path = os.path.join(input_folder, image_file)
        dst_path = os.path.join(output_folder, image_file)
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying image {image_file}: {e}")

    print(f"\nCopied top {top_n} images to {output_folder}.")

def filter_one_return_single_path(input_folder, common_name, top_n = 1):
    # ---------------------------
    # Configuration
    # ---------------------------

    # List of text prompts to compare with
    text_prompts = [
        "a photo of an animal",
        f"a photo of a {common_name}"
    ]
    # Ensure output folder exists

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Load Model and Processor
    # ---------------------------
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Prepare text inputs once
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        text_embeddings = model.get_text_features(**text_inputs)
        # Normalize text embeddings
        text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)

    # Lists to store image paths and their logits
    logits_list = []
    processed_image_files = []

    # Get list of image files
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]


    # Iterate through images with a progress bar
    # for image_file in tqdm(image_files, desc="Processing images"):
    for image_file in image_files: 
        image_path = os.path.join(input_folder, image_file)
        try:
            # Open image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue

        # Process image
        try:
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                image_embeddings = model.get_image_features(**inputs)
                # Normalize image embeddings
                image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
                # Compute cosine similarity
                cosine_sim = (image_embeddings @ text_embeddings.T).squeeze(0)  # Shape: [num_texts]
                logits_per_image = cosine_sim * 100  # Scale by 100 as in CLIP's default

                logits_list.append(logits_per_image.cpu())
                processed_image_files.append(image_file)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue
        finally:
            # Free up memory
            del image, inputs, image_embeddings, cosine_sim, logits_per_image
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if not logits_list:
        print("No logits were computed. Exiting.")
        return

    # Stack all logits: Shape [num_images, num_texts]
    all_logits = torch.stack(logits_list)  # Shape: [N, T]

    # ---------------------------
    # Post-processing
    # ---------------------------
    # For each text, perform softmax across images (dim=0)
    softmax = torch.nn.Softmax(dim=0)
    all_softmax = softmax(all_logits)  # Shape: [N, T]

    # Sum softmax scores across texts for each image
    image_scores = all_softmax.prod(dim=1)  # Shape: [N]

    # Sort images based on scores in descending order
    sorted_indices = torch.argsort(image_scores, descending=True)
    sorted_scores, sorted_indices = image_scores[sorted_indices], sorted_indices

    # ---------------------------
    # Copy Top N Images
    # ---------------------------
    ret = []
    top_n = min(top_n, len(sorted_scores))

    for i in range(top_n):
        idx = sorted_indices[i].item()
        image_file = processed_image_files[idx]
        if top_n == 1:
            return os.path.join(input_folder, image_file)
        else:
            ret.append(os.path.join(input_folder, image_file))
    return ret

def print_contrastiveness(list_main_images_paths, list_confusing_images_paths, attributes):
    # ---------------------------
    # Configuration
    # ---------------------------
    len_main = len(list_main_images_paths)
    len_confusing = len(list_confusing_images_paths)
    # List of text prompts to compare with
    
    # print(attributes)
    # Ensure output folder exists
    attributes = [attr.strip('\'').strip('\"') for attr in attributes]
    text_prompts = attributes
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Load Model and Processor
    # ---------------------------
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Prepare text inputs once
    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        text_embeddings = model.get_text_features(**text_inputs)
        # Normal
        text_embeddings /= text_embeddings.norm(p=2, dim=-1, keepdim=True)
    logits_list = []
    processed_image_files = []

    # Get list of image files
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
    report_acc = {}
    for attr in attributes:
        report_acc[attr] = 0.0
    # Iterate through images with a progress bar
    for image_file in tqdm(list_main_images_paths, desc="Processing images"):
        for confusing_image_file in list_confusing_images_paths:
            try:
                # Open image
                image = Image.open(image_file).convert("RGB")
                confusing_image = Image.open(confusing_image_file).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                continue

        # Process image
            try:
                # inputs = processor(images=image, return_tensors="pt").to(device)
                inputs = processor(images=[image, confusing_image], return_tensors="pt").to(device)

                with torch.no_grad():
                    image_embeddings = model.get_image_features(**inputs)
                    # Normalize image embeddings
                    image_embeddings /= image_embeddings.norm(p=2, dim=-1, keepdim=True)
                    # Compute cosine similarity
                    cosine_sim = (image_embeddings @ text_embeddings.T)  # Shape: [2, num_texts]
                    logits_per_image = cosine_sim * 100  # Scale by 100 as in CLIP's default
                    # softmax alone image dimension
                    softmax = torch.nn.Softmax(dim=0)
                    logits_per_image = softmax(logits_per_image)
                    # print(logits_per_image)
                    # Add accuracy score
                    for i in range(len(text_prompts)):
                        report_acc[text_prompts[i]] += float(logits_per_image[0][i].item()/(len_confusing * len_main))

            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                continue
            finally:
                # Free up memory
                del image, inputs, image_embeddings, cosine_sim, logits_per_image
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    return (report_acc)



if __name__ == "__main__":
    for i in range(1, 8):
        print(i)
        filter_subset(f"Reptiles_Random_{i}", 100)
    for i in range(1, 14):
        print(i)
        filter_subset(f"Mammals_Random_{i}", 100)
    for i in range(1, 16):
        print(i)
        filter_subset(f"Birds_Random_{i}", 100)