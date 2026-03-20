import os
import random
import shutil

base_dir = r"D:\passable_area_system\data"

image_dir = os.path.join(base_dir, "raw")
mask_dir = os.path.join(base_dir, "masks")

train_img = os.path.join(base_dir, "train/images")
train_mask = os.path.join(base_dir, "train/masks")
val_img = os.path.join(base_dir, "val/images")
val_mask = os.path.join(base_dir, "val/masks")

for p in [train_img, train_mask, val_img, val_mask]:
    os.makedirs(p, exist_ok=True)

images = os.listdir(image_dir)
random.shuffle(images)

split = int(len(images) * 0.8)

train_list = images[:split]
val_list = images[split:]

for img in train_list:
    shutil.copy(os.path.join(image_dir, img), os.path.join(train_img, img))
    shutil.copy(os.path.join(mask_dir, img), os.path.join(train_mask, img))

for img in val_list:
    shutil.copy(os.path.join(image_dir, img), os.path.join(val_img, img))
    shutil.copy(os.path.join(mask_dir, img), os.path.join(val_mask, img))

print("dataset split completed")
print("total:", len(images))