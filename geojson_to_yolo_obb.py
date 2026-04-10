import os
from convert_anylabeling_to_yolo_obb_augmentation import augment_dataset
from shapely.geometry import MultiPoint
import cv2
import numpy as np
import yaml
import shutil
import random

def reproject_raster(input_tif, output_tif, target_crs="EPSG:4326"):
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)

def convert_geojson_to_yolo_obb_geopandas(geojson_path, geotiff_path, output_txt, index_class):
    import geopandas as gpd
    import rasterio
    import shapely

    with rasterio.open(geotiff_path) as src:
        raster_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height

        print(f"Raster CRS: {raster_crs}")

    gdf = gpd.read_file(geojson_path)

    print(f"GeoJSON CRS: {gdf.crs}")

    if gdf.crs is None:
        print("⚠️ CRS GeoJSON tidak ada. Menetapkan EPSG:4326 sebagai default.")
        gdf.set_crs("EPSG:4326", inplace=True)

    if raster_crs is None:
        raise ValueError("GeoTIFF tidak memiliki CRS!")
    
    # # Tambahkan ini untuk mengatasi CRS lokal
    # if not raster_crs.is_projected or not raster_crs.to_epsg():
    #     print("⚠️ CRS raster tidak standar. Menggunakan EPSG:3857 sebagai fallback.")
    #     raster_crs = "EPSG:3857"
    
    # Tambahkan ini untuk mengatasi CRS lokal
    if not raster_crs.to_epsg():
        print("⚠️ CRS raster tidak standar. Menggunakan EPSG:3857 sebagai fallback.")
        raster_crs = "EPSG:3857"
        
    # Ubah ke CRS raster agar koordinat cocok dengan ukuran gambar
    gdf = gdf.to_crs(raster_crs)

    # # Mengubah polygon menjadi bounding box 4 titik yang oriented dengan minimum rotasi 
    # with open(output_txt, 'w') as f:
    #     for geom in gdf.geometry:
    #         if not geom.is_valid or geom.is_empty:
    #             continue
    #         polygon = geom.simplify(0.5).minimum_rotated_rectangle
    #         if not isinstance(polygon, shapely.Polygon):
    #             continue
    #         coords = list(polygon.exterior.coords)[:4]  # ambil 4 titik saja
    #         coords = [(x, y) for x, y in coords]
    #         # Konversi world coords ke pixel coords
    #         pixel_coords = [~transform * (x, y) for x, y in coords]
    #         # Normalisasi ke 0-1
    #         norm_coords = []
    #         for x, y in pixel_coords:
    #             norm_coords.append(x / width)
    #             norm_coords.append(y / height)
    #         # Pastikan ada 8 nilai (4 titik)
    #         if len(norm_coords) == 8 and all(0 <= c <= 1 for c in norm_coords):
    #             f.write(f"0 {' '.join(map(str, norm_coords))}\n")

    # # Mempertahankan polygon asli
    with open(output_txt, 'w') as f:
        for geom in gdf.geometry:
            if not geom.is_valid or geom.is_empty:
                continue

            polygon = geom

            if not isinstance(polygon, shapely.Polygon):
                continue

            # Ambil semua titik polygon (tanpa minimum_rotated_rectangle)
            coords = list(polygon.exterior.coords)

            # Konversi world coords ke pixel coords
            pixel_coords = [~transform * (x, y) for x, y in coords]
            
            # pixel_coords = MultiPoint(pixel_coords)
            # min_rect = pixel_coords.minimum_rotated_rectangle
            # pixel_coords = np.array(min_rect.exterior.coords)[:4]

            # Normalisasi ke 0-1
            norm_coords = []
            for x, y in pixel_coords:
                norm_coords.append(x / width)
                norm_coords.append(y / height)

            # Pastikan semua nilai dalam rentang 0-1
            if all(0 <= c <= 1 for c in norm_coords):
                # f.write(f"0 {' '.join(map(str, norm_coords))}\n")
                f.write(f"{index_class} {' '.join(map(str, norm_coords))}\n")


            
def save_images_and_labels_from_geojson(image_path, geojson_path, output_dir, index_class):
    import shutil
    extension = os.path.splitext(image_path)[1].lower()
    
    # Buat output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_dir_labels = os.path.join(output_dir, "labels")
    output_dir_images = os.path.join(output_dir, "images")
    
    os.makedirs(output_dir_labels, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(geojson_path))[0]
    output_file_labels = os.path.join(output_dir_labels, f"{base_name}.txt")
    
    convert_geojson_to_yolo_obb_geopandas(geojson_path, image_path, output_file_labels, index_class)
    
    # Salin gambar ke directory images
    shutil.copy2(image_path, os.path.join(output_dir_images, f"{base_name}{extension}"))
    print(f"Copied image: {image_path}")

    os.remove(image_path)
    print(f"Removed Copied image: {image_path}")
    
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
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
    
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
    
    # # Salin file classes.txt ke setiap set
    # for dir_path in [train_dir, val_dir, test_dir]:
    #     shutil.copy2(
    #         os.path.join(output_dir, 'classes.txt'),
    #         os.path.join(dir_path, 'classes.txt')
    #     )
    
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

def batch_process_geojson_tif(input_folder, output_dir, target_crs="EPSG:4326", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, dataset_name='dataset', augmented=False, num_augmented=0):
    import shutil
    
    # Hapus output directory jika sudah ada
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Baca classes.txt untuk mendapatkan nama kelas
    classes_list = []
    classes_file = os.path.join(input_folder, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            classes_list = [name for name in classes if name]

    print("classes_list", classes_list)

    for i, class_name in enumerate(classes_list):
        if os.path.exists(os.path.join(input_folder, class_name)):
            class_folder = os.path.join(input_folder, class_name)
            print(f"Found folder class: {class_name}")

            tif_files = [f for f in os.listdir(class_folder) if f.lower().endswith('.tif')]
            for tif_file in tif_files:
                base_name = os.path.splitext(tif_file)[0]
                geojson_file = base_name + ".geojson"
                tif_path = os.path.join(class_folder, tif_file)
                geojson_path = os.path.join(class_folder, geojson_file)
                if os.path.exists(geojson_path):
                    print(f"Processing: {tif_file} & {geojson_file}")
                    reprojected_image_path = tif_path.replace(".tif", "_reprojected.tif")
                    reproject_raster(tif_path, reprojected_image_path, target_crs=target_crs)
                    save_images_and_labels_from_geojson(reprojected_image_path, geojson_path, output_dir, index_class=i)
                else:
                    print(f"GeoJSON not found for {tif_file}, skipping.")
            
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    if augmented == True:
        augment_dataset(image_dir=images_dir, label_dir=labels_dir, output_dir=output_dir, augment_count=num_augmented)
        
    split_dataset(output_dir, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    
    if os.path.exists(classes_file):
        shutil.copy2(classes_file, os.path.join(output_dir, 'classes.txt'))
        create_yaml_file(output_dir, dataset_name)

# Contoh penggunaan batch
if __name__ == "__main__":
    input_folder = "Jakarta_Building_GeoJSON_Data"
    output_dir = "Datasets/geojson_converted_dataset"
    is_augmented = True
    num_augmented = 3
    
    
    batch_process_geojson_tif(
        input_folder, 
        output_dir, 
        target_crs="EPSG:4326", 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1, 
        dataset_name='dataset', 
        augmented=is_augmented, 
        num_augmented=num_augmented
    )