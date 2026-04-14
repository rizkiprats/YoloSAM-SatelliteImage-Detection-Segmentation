import json
import os
import numpy as np
from pathlib import Path
import shutil
import random
import yaml
import cv2

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
        normalized_points = convert_polygon_to_obb(points, image_path)
        
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

def process_directory(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, dataset_name='dataset'):
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
    
    # Bagi dataset
    split_dataset(output_dir, train_ratio, val_ratio, test_ratio)
    
    # Buat file YAML
    create_yaml_file(output_dir, dataset_name)

if __name__ == "__main__":
    # Contoh penggunaan
    input_dir = r"Jakarta_Building_Anylabeling/Jakarta_Building"
    output_dir = r"Datasets/augmented_dataset"
    
    # Bagi dataset dengan rasio 70% training, 20% validation, 10% test
    process_directory(
        input_dir, 
        output_dir, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        dataset_name='Datasets'  # Nama file YAML akan menjadi 'Datasets.yaml'
    ) 