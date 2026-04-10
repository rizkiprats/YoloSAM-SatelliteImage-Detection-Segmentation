import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QFileDialog, QHBoxLayout, QMessageBox, QDoubleSpinBox, QSpinBox, QLabel, QCheckBox
)
from geojson_to_yolo_obb import batch_process_geojson_tif

class ConverterAugmentUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoJson to YOLO OBB Converter")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.input_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.dataset_name_edit = QLineEdit("dataset")

        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.7)

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 1.0)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)

        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.1)

        input_dir_btn = QPushButton("Browse")
        input_dir_btn.clicked.connect(self.browse_input_dir)
        output_dir_btn = QPushButton("Browse")
        output_dir_btn.clicked.connect(self.browse_output_dir)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(input_dir_btn)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_dir_btn)

        form_layout.addRow("Input Directory:", input_layout)
        form_layout.addRow("Output Directory:", output_layout)
        form_layout.addRow("Dataset Name (YAML):", self.dataset_name_edit)
        form_layout.addRow("Train Ratio:", self.train_spin)
        form_layout.addRow("Val Ratio:", self.val_spin)
        form_layout.addRow("Test Ratio:", self.test_spin)

        self.aug_spin = QSpinBox()
        self.aug_spin.setRange(0, 50)
        self.aug_spin.setValue(10)
        layout.addWidget(QLabel("Jumlah Augmentasi per Gambar:"))
        layout.addWidget(self.aug_spin)

        self.augment_cb = QCheckBox("Aktifkan Augmentasi")
        self.augment_cb.setChecked(True)
        layout.addWidget(self.augment_cb)

        self.run_button = QPushButton("Convert and Split")
        self.run_button.clicked.connect(self.run_process)

        layout.addLayout(form_layout)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir_path:
            self.input_dir_edit.setText(dir_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def run_process(self):
        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        dataset_name = self.dataset_name_edit.text().strip()
        train_ratio = self.train_spin.value()
        val_ratio = self.val_spin.value()
        test_ratio = self.test_spin.value()
        is_augmented = self.augment_cb.isChecked()
        num_augmented = self.aug_spin.value()

        if not input_dir or not output_dir or not dataset_name:
            QMessageBox.warning(self, "Warning", "Semua field harus diisi.")
            return

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            QMessageBox.warning(self, "Warning", "Jumlah rasio harus = 1.0")
            return
        
        if is_augmented == True and num_augmented == 0:
            QMessageBox.warning(self, "Warning", "Jumlah augmented harus lebih dari 1")
            return

        try:
            if is_augmented == True:
                batch_process_geojson_tif(
                    input_dir,
                    output_dir,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    dataset_name=dataset_name,
                    augmented=is_augmented,
                    num_augmented=num_augmented
                )
                QMessageBox.information(self, "Sukses", "Konversi, Augmented dan split dataset berhasil.")
            else:
                batch_process_geojson_tif(
                    input_dir,
                    output_dir,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    dataset_name=dataset_name
                )
                QMessageBox.information(self, "Sukses", "Konversi dan split dataset berhasil.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ConverterAugmentUI()
    ui.show()
    sys.exit(app.exec_())