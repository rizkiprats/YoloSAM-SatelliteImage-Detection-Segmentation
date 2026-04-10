import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)

from PyQt5.QtGui import QPixmap, QImage


def show_annotations(image_path, annotation_path, class_names=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Tidak dapat membaca gambar dari {image_path}")
        return None

    h, w, _ = image.shape
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3)).tolist()

    try:
        with open(annotation_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File anotasi tidak ditemukan di {annotation_path}")
        return image

    for line in lines:
        data = line.strip().split()
        if len(data) < 9:
            continue

        class_id = int(data[0])
        points = np.array([(float(data[i]) * w, float(data[i+1]) * h) for i in range(1, len(data), 2)], np.int32)
        color = colors[class_id % len(colors)]

        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        text = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        text_pos = (points[0][0], points[0][1] - 5)
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


class AnnotationViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Viewer")
        self.image_dir = ""
        self.label_dir = ""
        self.class_names = []
        self.image_files = []
        self.index = 0

        self.scene = QGraphicsScene()
        self.view = GraphicsViewWithZoom(self.scene)  # Ganti ke custom view
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.pixmap_item = None
        self.current_pixmap = None
        self.scale_factor = 1.0

        self.load_folder_button = QPushButton("Pilih Folder Dataset")
        self.load_classes_button = QPushButton("Pilih File classes.txt")
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.reset_zoom_button = QPushButton("Reset Zoom")

        self.load_folder_button.clicked.connect(self.load_folder)
        self.load_classes_button.clicked.connect(self.load_classes_file)
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_folder_button)
        button_layout.addWidget(self.load_classes_button)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.zoom_in_button)
        button_layout.addWidget(self.zoom_out_button)
        button_layout.addWidget(self.reset_zoom_button)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder Dataset")
        if not folder:
            return

        self.image_dir = os.path.join(folder, "images")
        self.label_dir = os.path.join(folder, "labels")

        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            QMessageBox.critical(self, "Error", "Folder 'images' atau 'labels' tidak ditemukan.")
            return

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff', '.tif'))
        ])
        self.index = 0
        if self.image_files:
            self.show_image()
        else:
            QMessageBox.warning(self, "Tidak Ada Gambar", "Tidak ditemukan gambar di folder images.")

    def load_classes_file(self):
        class_file, _ = QFileDialog.getOpenFileName(self, "Pilih File classes.txt", "", "Text Files (*.txt)")
        if class_file:
            try:
                with open(class_file, "r") as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                QMessageBox.information(self, "Sukses", f"{len(self.class_names)} kelas dimuat dari file.")
            except Exception as e:
                QMessageBox.critical(self, "Gagal", f"Gagal membaca file classes.txt:\n{str(e)}")

    def show_image(self):
            if not self.image_files:
                return

            image_filename = self.image_files[self.index]
            image_path = os.path.join(self.image_dir, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_filename)

            annotated_image = show_annotations(image_path, label_path, self.class_names)

            if annotated_image is not None:
                rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.current_pixmap = pixmap
                self.scene.clear()
                self.pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)
                # Set scene rect persis sesuai gambar
                self.scene.setSceneRect(0, 0, w, h)
                self.reset_zoom(center=True)
            else:
                self.scene.clear()
                self.scene.addText("Gagal memuat gambar.")
                self.scene.setSceneRect(0, 0, 800, 600)  # fallback

    def zoom_in(self):
        self.view.zoom(1.25)

    def zoom_out(self):
        self.view.zoom(0.8)

    def reset_zoom(self, center=False):
        self.view.reset_zoom(center=center)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Pilih Folder Dataset")
        if not folder:
            return

        self.image_dir = os.path.join(folder, "images")
        self.label_dir = os.path.join(folder, "labels")

        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            QMessageBox.critical(self, "Error", "Folder 'images' atau 'labels' tidak ditemukan.")
            return

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff', '.tif'))
        ])
        self.index = 0
        if self.image_files:
            self.show_image()
        else:
            QMessageBox.warning(self, "Tidak Ada Gambar", "Tidak ditemukan gambar di folder images.")

    def show_next_image(self):
        if self.index < len(self.image_files) - 1:
            self.index += 1
            self.show_image()

    def show_prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()


from PyQt5.QtWidgets import QGraphicsView

class GraphicsViewWithZoom(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zoom = 0

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom(1.25)
        else:
            self.zoom(0.8)

    def zoom(self, factor):
        self._zoom += 1 if factor > 1 else -1
        self.scale(factor, factor)

    def reset_zoom(self, center=False):
        self.resetTransform()
        self._zoom = 0
        if center:
            # Pusatkan ke tengah gambar (bukan sceneRect, tapi boundingRect item)
            items = self.scene().items()
            if items:
                pixmap_item = items[0]
                rect = pixmap_item.boundingRect()
                self.centerOn(rect.center())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AnnotationViewer()
    viewer.resize(1366, 768)
    viewer.show()
    sys.exit(app.exec_())
