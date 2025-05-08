# ... [IMPORTS] ...
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt
import mediapipe as mp
from datetime import datetime
import subprocess
import platform


class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("EchoVerse")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e2f")

        self.model_dir = "model"
        json_path = os.path.join(self.model_dir, "model-bw.json")
        weights_path = os.path.join(self.model_dir, "model-bw.weights.h5")
        class_indices_path = os.path.join(self.model_dir, "class_indices.json")

        self._validate_files(json_path, weights_path, class_indices_path)
        self.loaded_model = self._load_model(json_path, weights_path)
        self.class_labels = self._load_class_labels(class_indices_path)

        self._setup_ui()

        self.cm_open = False
        self.cm_fig = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.test_images_dir = "test_images"
        os.makedirs(self.test_images_dir, exist_ok=True)

        self.current_sentence = ""

    def _validate_files(self, *file_paths):
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Error: {file_path} not found!")

    def _load_model(self, json_path, weights_path):
        with open(json_path, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        print("✅ Model loaded successfully!")
        return model

    def _load_class_labels(self, class_indices_path):
        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}

    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton",
                        font=("Segoe UI", 10),
                        padding=5,
                        borderwidth=0,
                        relief="flat",
                        foreground="white",
                        background="#3a3a5c")
        style.map("TButton",
                  background=[("active", "#5c5cb8")],
                  foreground=[("active", "white")])

        top_frame = tk.Frame(self.root, bg="#1e1e2f")
        top_frame.pack(fill=tk.X, pady=10)

        title_label = tk.Label(top_frame, text="✨ EchoVerse ✨",
                               font=("Segoe UI", 24, "bold"),
                               bg="#1e1e2f", fg="#b3b3ff")
        title_label.pack(pady=10)

        matrix_frame = tk.Frame(top_frame, bg="#1e1e2f")
        matrix_frame.pack(fill=tk.X, padx=15)

        self.confusion_btn = ttk.Button(matrix_frame, text="📊 Show Confusion Matrix",
                                        command=self.toggle_confusion_matrix)
        self.confusion_btn.pack(side=tk.RIGHT)

        self.label = tk.Label(self.root, text="Upload an Image or Use Webcam",
                              font=("Segoe UI", 13),
                              bg="#1e1e2f", fg="#ccccff")
        self.label.pack(pady=10)

        btn_frame = tk.Frame(self.root, bg="#1e1e2f")
        btn_frame.pack(pady=10)

        self.upload_btn = ttk.Button(btn_frame, text="📁 Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.webcam_btn = ttk.Button(btn_frame, text="📷 Use Webcam", command=self.open_webcam)
        self.webcam_btn.grid(row=0, column=1, padx=10)

        self.clear_btn = ttk.Button(btn_frame, text="🗑️ Clear", command=self.clear_display)
        self.clear_btn.grid(row=0, column=2, padx=10)

        self.delete_btn = ttk.Button(btn_frame, text="🧹 Delete Last Character", command=self.delete_last_character)
        self.delete_btn.grid(row=0, column=3, padx=10)

        self.open_folder_btn = ttk.Button(btn_frame, text="📂 Open Test Images", command=self.open_test_images_folder)
        self.open_folder_btn.grid(row=0, column=4, padx=10)
        
        self.img_label = tk.Label(self.root, bg="#1e1e2f")
        self.img_label.pack(pady=20)

        self.result_label = tk.Label(self.root, text="Prediction: ",
                                     font=("Segoe UI", 14, "bold"),
                                     bg="#1e1e2f", fg="#99ccff")
        self.result_label.pack(pady=10)

        self.sentence_label = tk.Label(self.root, text="Sentence: ",
                                       font=("Segoe UI", 14),
                                       bg="#1e1e2f", fg="#99ffcc")
        self.sentence_label.pack(pady=5)

    def delete_last_character(self):
        if self.current_sentence:
            self.current_sentence = self.current_sentence[:-1]
            self.sentence_label.config(text="Sentence: " + self.current_sentence)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.display_image(file_path)
            self.predict_image(file_path)

    def display_image(self, img_path):
        img = Image.open(img_path).convert("L").resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def clear_display(self):
        self.img_label.config(image="")
        self.img_label.image = None
        self.result_label.config(text="Prediction: ")
        self.sentence_label.config(text="Sentence: ")
        self.current_sentence = ""

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(128, 128), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

    def predict_image(self, img_path):
        img_array = self.preprocess_image(img_path)
        prediction = self.loaded_model.predict(img_array)

        max_index = np.argmax(prediction[0])
        label = self.class_labels.get(max_index, "Unknown")
        confidence = prediction[0][max_index]

        self.result_label.config(text=f"Prediction: {label} ({confidence:.2f})")
        self.display_image(img_path)  # ✅ Show predicted image in UI
        self.current_sentence += label
        self.sentence_label.config(text="Sentence: " + self.current_sentence)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{confidence:.2f}_{timestamp}.png"

        try:
            image_pil = Image.open(img_path).convert("L").resize((128, 128))
            image_pil.save(os.path.join(self.test_images_dir, filename))
        except Exception as e:
            print(f"⚠️ Error saving predicted image: {e}")

    def open_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            hand_img = None
            label = "Unknown"
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    h, w, _ = frame.shape
                    xmin = int(min(x_coords) * w) - 20
                    ymin = int(min(y_coords) * h) - 20
                    xmax = int(max(x_coords) * w) + 20
                    ymax = int(max(y_coords) * h) + 20

                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(w, xmax), min(h, ymax)

                    hand_img = frame[ymin:ymax, xmin:xmax]
                    if hand_img.size > 0:
                        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (128, 128))
                        img_array = np.expand_dims(resized, axis=(0, -1)) / 255.0

                        prediction = self.loaded_model.predict(img_array)
                        max_index = np.argmax(prediction[0])
                        label = self.class_labels.get(max_index, "Unknown")
                        confidence = prediction[0][max_index]

                        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and hand_img is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{label}_{confidence:.2f}_{timestamp}.png"
                filepath = os.path.join(self.test_images_dir, filename)
                cv2.imwrite(filepath, hand_img)
                print(f"✅ Image captured and saved as {filepath}")
                self.predict_image(filepath)

        cap.release()
        cv2.destroyAllWindows()

    def open_test_images_folder(self):
        path = os.path.abspath(self.test_images_dir)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    def toggle_confusion_matrix(self):
        if self.cm_open:
            plt.close(self.cm_fig)
            self.cm_open = False
            self.cm_fig = None
            self.confusion_btn.config(text="📊 Show Confusion Matrix")
            return

        test_dir = "data/test"
        if not os.path.exists(test_dir):
            print("❌ Test data directory not found!")
            return

        datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(128, 128),
            color_mode="grayscale",
            class_mode="categorical",
            shuffle=False,
            batch_size=1
        )

        predictions = self.loaded_model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        cm = confusion_matrix(true_classes, predicted_classes)
        self.cm_fig = plt.figure(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix")
        plt.show()
        self.cm_open = True
        self.confusion_btn.config(text="❌ Close Confusion Matrix")


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
