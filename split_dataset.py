import os
import shutil
from sklearn.model_selection import train_test_split

# Splits a dataset laid out as:
#
#   zoo5/
#     classA/
#       img1.jpg
#       img2.jpg
#     classB/
#       img3.jpg
#
# into 80% train and 20% val while preserving class folders:
#
#   zoo5/train/classA/*.jpg
#   zoo5/val/classA/*.jpg

dataset_dir = 'zoo5'
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

if not os.path.isdir(dataset_dir):
    raise SystemExit(f"Dataset directory not found: {dataset_dir}")

train_root = os.path.join(dataset_dir, 'train')
val_root = os.path.join(dataset_dir, 'val')
os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)

def move_file(src_dir, dst_dir, filename):
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    try:
        shutil.move(src, dst)
    except Exception as e:
        print(f"Failed to move {src} -> {dst}: {e}")

total_train = 0
total_val = 0

# Process class subdirectories inside dataset_dir
for entry in sorted(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, entry)
    # skip the train/val folders if present
    if entry in ('train', 'val'):
        continue
    if not os.path.isdir(class_dir):
        continue

    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(class_dir, f))]
    if not images:
        print(f"No images found in class folder: {entry}, skipping")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    dst_train_class = os.path.join(train_root, entry)
    dst_val_class = os.path.join(val_root, entry)

    for img in train_imgs:
        move_file(class_dir, dst_train_class, img)
    for img in val_imgs:
        move_file(class_dir, dst_val_class, img)

    print(f"Class '{entry}': moved {len(train_imgs)} train, {len(val_imgs)} val")
    total_train += len(train_imgs)
    total_val += len(val_imgs)

# Optionally process images that might be directly in the dataset root
root_images = [f for f in os.listdir(dataset_dir)
               if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(dataset_dir, f))]
if root_images:
    train_imgs, val_imgs = train_test_split(root_images, test_size=0.2, random_state=42)
    for img in train_imgs:
        move_file(dataset_dir, train_root, img)
    for img in val_imgs:
        move_file(dataset_dir, val_root, img)
    print(f"Root images: moved {len(train_imgs)} train, {len(val_imgs)} val")
    total_train += len(train_imgs)
    total_val += len(val_imgs)

print("Dataset split complete!")
print(f"Total train images: {total_train}")
print(f"Total validation images: {total_val}")