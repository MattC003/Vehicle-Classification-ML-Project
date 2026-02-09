import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import hashlib

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

directory = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive'
os.chdir(directory)

prompts = ["full body exterior of vehicle", "car interior"]

def classify_image(image_path):
    image = Image.open(image_path)
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    exterior_prob = probs[0][0].item()
    interior_prob = probs[0][1].item()

    if interior_prob > exterior_prob:
        print(f"Deleting {image_path}")
        os.remove(image_path)
    else:
        print(f"Keeping {image_path}")

def calculate_image_hash(image_path):
    with Image.open(image_path) as img:
        img = img.convert("L").resize((8, 8), Image.LANCZOS)
        return hashlib.md5(img.tobytes()).hexdigest()

def remove_duplicates():
    hash_dict = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            img_hash = calculate_image_hash(file_path)
            if img_hash in hash_dict:
                print(f"Deleting duplicate image {file_path}")
                os.remove(file_path)
            else:
                hash_dict[img_hash] = file_path

def classify_and_filter_image(image_path):
    image = Image.open(image_path)
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    full_body_prob = probs[0][0].item()
    closeup_prob = probs[0][1].item()

    if closeup_prob > full_body_prob:
        print(f"Deleting {image_path}")
        os.remove(image_path)
    else:
        print(f"Keeping {image_path}")

def parse_filename(filename):
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    make = parts[0]
    model = parts[1]
    try:
        year = int(parts[2])
    except ValueError:
        return None
    return (make, model, year)

def remove_duplicates_by_filename():
    unique_images = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            car_info = parse_filename(filename)
            if car_info:
                make, model, year = car_info
                key = (make, model, year)
                if key not in unique_images:
                    unique_images[key] = file_path
                else:
                    print(f"Deleting duplicate image {file_path}")

if __name__ == '__main__':
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            classify_and_filter_image(file_path)
    remove_duplicates()
    remove_duplicates_by_filename()
