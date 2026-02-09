import os
import shutil

def organize_files():
    base_dir = os.getcwd()
    items = os.listdir(base_dir)
    
    for item in items:
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            continue
        
        try:
            category = item.split('_')[-2]
        except IndexError:
            print(f"Skipping: {item} (insufficient segments)")
            continue
        
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            print(f"Created folder: {category_dir}")
        
        destination = os.path.join(category_dir, item)
        shutil.move(item_path, destination)
        print(f"Moved: {item} -> {category_dir}")

if __name__ == "__main__":
    organize_files()
