# convert_anylabeling_to_yolo_obb_augmentation_fixed_ui.py
import os
import json
import shutil
import numpy as np
import cv2
import random
from pathlib import Path
import albumentations as A
import yaml
from shapely.geometry import MultiPoint

def convert_polygon_to_obb(points, image_path):
    """
    Mengkonversi polygon menjadi 4 titik koordinat yang dinormalisasi
    """
    points = np.array(points)
    
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    normalized_points = []
    for point in points:
        normalized_points.extend([
            point[0] / w,  # x
            point[1] / h  # y
        ])
    
    return normalized_points

def get_best_obb_box_shapely(points, image_path):
    """
    Ambil 4 point terbaik yang membentuk minimum rotated rectangle (OBB) dari sekumpulan titik menggunakan Shapely.
    Hasil point sudah dinormalisasi terhadap ukuran gambar.
    """
    points = np.array(points, dtype=np.float32)
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Buat MultiPoint dan dapatkan minimum rotated rectangle
    multipoint = MultiPoint(points)
    min_rect = multipoint.minimum_rotated_rectangle

    # Ambil koordinat corner rectangle (biasanya 5 point, titik terakhir == titik pertama)
    coords = np.array(min_rect.exterior.coords)[:4]  # Ambil 4 point pertama

    # Normalisasi point
    normalized_points = []
    for x, y in coords:
        # normalized_points.append((x / w, y / h))
        normalized_points.extend([x / w, y / h])

    return normalized_points



def convert_anylabeling_to_yolo_obb(image_path, json_file, output_dir, dataset_dir, class_mapping=None):
    """
    Mengkonversi file anotasi X-AnyLabeling ke format YOLO OBB
    
    Args:
        json_file (str): Path ke file JSON X-AnyLabeling
        output_dir (str): Directory output untuk file YOLO OBB
        class_mapping (dict): Mapping dari label ke class ID
    """
    # Buat output directory jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Baca file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Jika class_mapping tidak diberikan, buat dari label yang ada
    if class_mapping is None:
        unique_labels = set(shape['label'] for shape in data['shapes'])
        class_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Buat file classes.txt
    if not os.path.exists(os.path.join(dataset_dir, 'classes.txt')):
        with open(os.path.join(dataset_dir, 'classes.txt'), 'w') as f:
            for label in sorted(class_mapping.keys()):
                f.write(f"{label}\n")
    else:
        with open(os.path.join(dataset_dir, 'classes.txt'), 'r') as f:
            existing_classes = set(line.strip() for line in f)
        
        with open(os.path.join(dataset_dir, 'classes.txt'), 'a') as f:
            for label in sorted(class_mapping.keys()):
                if label not in existing_classes:
                    f.write(f"{label}\n")
        
    # Proses setiap shape
    for shape in data['shapes']:
        if shape['shape_type'] != 'polygon':
            continue
            
        label = shape['label']
        points = shape['points']
        
        # Konversi polygon ke format yang dinormalisasi
        # normalized_points = convert_polygon_to_obb(points, image_path)
        normalized_points = get_best_obb_box_shapely(points, image_path)
        
        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        yolo_line = f"{class_mapping[label]} {' '.join(map(str, normalized_points))}\n"
        
        # Tulis ke file .txt dengan nama yang sama dengan gambar
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_file, 'a') as f:
            f.write(yolo_line)

def split_dataset(output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Membagi dataset menjadi train, validation, dan test set
    
    Args:
        output_dir (str): Directory yang berisi dataset
        train_ratio (float): Persentase data training (default: 0.7)
        val_ratio (float): Persentase data validation (default: 0.2)
        test_ratio (float): Persentase data test (default: 0.1)
        seed (int): Seed untuk random generator
    """
    # Set random seed
    random.seed(seed)
    
    # Buat directory untuk setiap set
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
    
    # Dapatkan semua file gambar
    images_dir = os.path.join(output_dir, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    labels_dir = os.path.join(output_dir, 'labels')
    # label_files = [f for f in os.listdir(labels_dir) if f.endswith(('.txt'))]
    
    # Acak urutan file
    random.shuffle(image_files)
    
    # Hitung jumlah file untuk setiap set
    n_files = len(image_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Bagi file ke dalam set
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Fungsi untuk menyalin file ke directory tujuan
    def copy_files(files, target_dir):
        for img_file in files:
            # Salin file gambar
            shutil.copy2(
                os.path.join(images_dir, img_file),
                os.path.join(target_dir, 'images', img_file)
            )
            
            # Salin file anotasi
            base_name = os.path.splitext(img_file)[0]
            txt_file = f"{base_name}.txt"
            if os.path.exists(os.path.join(labels_dir, txt_file)):
                shutil.copy2(
                    os.path.join(labels_dir, txt_file),
                    os.path.join(target_dir, 'labels', txt_file)
                )
    
    # Salin file ke masing-masing set
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    # Print statistik
    print(f"\nDataset split complete:")
    print(f"Total files: {n_files}")
    print(f"Training set: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"Validation set: {len(val_files)} files ({val_ratio*100:.1f}%)")
    print(f"Test set: {len(test_files)} files ({test_ratio*100:.1f}%)")

def create_yaml_file(output_dir, dataset_name):
    """
    Membuat file YAML untuk konfigurasi dataset YOLO
    
    Args:
        output_dir (str): Directory output dataset
        dataset_name (str): Nama dataset (untuk nama file YAML)
    """
    # Buat path relatif dari output_dir
    base_path = os.path.abspath(output_dir)
    
    # Buat dictionary untuk YAML
    yaml_data = {
        'path': base_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {}
    }
    
    # Baca classes.txt untuk mendapatkan nama kelas
    classes_file = os.path.join(output_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            yaml_data['names'] = {i: name for i, name in enumerate(classes)}
    
    # Tulis ke file YAML
    yaml_file = os.path.join(output_dir, f'{dataset_name}.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"\nCreated YAML file: {yaml_file}")
    print("YAML content:")
    print(yaml.dump(yaml_data, default_flow_style=False))

def process_directory(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, dataset_name='dataset', augmented=False, num_augmented=0):
    """
    Proses semua file JSON dalam directory, salin gambar yang sesuai, dan bagi dataset
    
    Args:
        input_dir (str): Directory input yang berisi file JSON
        output_dir (str): Directory output untuk file YOLO OBB
        train_ratio (float): Persentase data training
        val_ratio (float): Persentase data validation
        test_ratio (float): Persentase data test
        dataset_name (str): Nama dataset untuk file YAML
    """
    # Hapus output directory jika sudah ada
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Buat output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Buat directory untuk gambar
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Buat directory untuk labels
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # Proses semua file JSON
    for json_file in Path(input_dir).glob('*.json'):
        # Salin file gambar yang sesuai
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        # Coba beberapa ekstensi gambar yang umum
        for ext in ['.jpg', '.jpeg', '.png']:
            img_file = os.path.join(input_dir, f"{base_name}{ext}")
            if os.path.exists(img_file):
                # Proses file JSON
                convert_anylabeling_to_yolo_obb(img_file, str(json_file), labels_dir, output_dir)
                
                # Salin gambar ke directory images
                shutil.copy2(img_file, os.path.join(images_dir, f"{base_name}{ext}"))
                print(f"Copied image: {img_file}")
                break
        
        print(f"Processed {json_file}")
    
    if augmented == True:
        augment_dataset(images_dir, labels_dir, output_dir, num_augmented)
    
    # Bagi dataset
    split_dataset(output_dir, train_ratio, val_ratio, test_ratio)
    
    # Buat file YAML
    create_yaml_file(output_dir, dataset_name)

def augment_dataset(image_dir, label_dir, output_dir, augment_count):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # # Define augmentation pipeline    
    # Augmentasi Aman untuk OBB
    # transform = A.Compose([
    #     A.RandomRotate90(p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.3),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.HueSaturationValue(p=0.5),
    #     A.Blur(p=0.3),
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    # ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Blur(p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.CLAHE(p=0.3),  # Contrast Limited Adaptive Histogram Equalization
        A.RandomGamma(p=0.3),  # Random gamma correction
        A.GaussNoise(std_range=(0.2, 0.44), p=0.3),  # Gaussian noise
        A.MotionBlur(p=0.2),  # Motion blur
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),  # RGB channel shift
        A.ToGray(p=0.1),  # Convert to grayscale
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
    # Process each image
    for img_file in image_dir.glob('*'):
        # Read image
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Error reading image: {img_file}")
                continue
            
        # Read corresponding label file
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        # Create augmented versions
        for i in range(augment_count):
            # Convert all points to keypoints format
            all_keypoints = []
            for line in lines:
                parts = line.strip().split()
                coords = list(map(float, parts[1:]))
                points = [(coords[j] * image.shape[1], coords[j+1] * image.shape[0]) for j in range(0, len(coords), 2)]
                all_keypoints.extend(points)
            
            # Apply augmentation
            transformed = transform(image=image, keypoints=all_keypoints)
            aug_img = transformed['image']
            aug_keypoints = transformed['keypoints']
            
            # Process augmented keypoints back to YOLO format
            aug_labels = []
            keypoint_idx = 0
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                num_points = (len(parts) - 1) // 2
                
                # Get augmented coordinates for this object
                obj_points = aug_keypoints[keypoint_idx:keypoint_idx + num_points]
                keypoint_idx += num_points
                
                # Convert to normalized coordinates
                normalized_points = []
                for x, y in obj_points:
                    normalized_x = x / aug_img.shape[1]
                    normalized_y = y / aug_img.shape[0]
                    normalized_points.extend([normalized_x, normalized_y])
                
                # Add to augmented labels
                aug_labels.append(f"{class_id} {' '.join(map(str, normalized_points))}")
            
            # Save augmented image
            aug_img_path = output_img_dir / f"{img_file.stem}_aug_{i}.jpg"
            cv2.imwrite(str(aug_img_path), aug_img)
            
            # Save all augmented labels in one file
            aug_label_path = output_label_dir / f"{img_file.stem}_aug_{i}.txt"
            with open(aug_label_path, 'w') as f:
                f.write('\n'.join(aug_labels))

# Contoh penggunaan
if __name__ == "__main__":
    input_dir = r"Jakarta_Building_Anylabeling/Jakarta_Building"
    output_dir = r"Datasets/augmented_dataset"
    is_augmented = True
    num_augmented = 3

    process_directory(
        input_dir, 
        output_dir, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        dataset_name='Datasets',
        is_augmented=is_augmented,
        num_augmented=num_augmented
    ) 