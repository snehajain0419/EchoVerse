# echoverse_app.py


import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

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
        self.theme = "light"
        self._set_theme_colors()
        self.root.configure(bg=self.bg_color)

        self.model_dir = "model"
        json_path = os.path.join(self.model_dir, "model-bw.json")
        weights_path = os.path.join(self.model_dir, "model-bw.weights.h5")
        class_indices_path = os.path.join(self.model_dir, "class_indices.json")

        self._validate_files(json_path, weights_path, class_indices_path)
        self.loaded_model = self._load_model(json_path, weights_path)
        self.class_labels = self._load_class_labels(class_indices_path)

        self.cm_open = False
        self.cm_fig = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.test_images_dir = "test_images"
        os.makedirs(self.test_images_dir, exist_ok=True)

        self.current_sentence = ""

        self.menu_frame = None
        self._setup_ui()

    def _set_theme_colors(self):
        if self.theme == "light":
            self.bg_color = "#f5f5dc"
            self.frame_color = "#fffaf0"
            self.fg_color = "#3e2723"
            self.button_bg = "#8d6e63"
            self.button_hover = "#a1887f"
        else:
            self.bg_color = "#d2b48c"
            self.frame_color = "#cbb294"
            self.fg_color = "#3e2723"
            self.button_bg = "#5d4037"
            self.button_hover = "#8d6e63"

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self._set_theme_colors()
        self.root.configure(bg=self.bg_color)
        self._setup_ui()

    def _validate_files(self, *file_paths):
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

    def _load_model(self, json_path, weights_path):
        with open(json_path, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        return model

    def _load_class_labels(self, class_indices_path):
        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}

    def _setup_ui(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.configure(bg=self.bg_color)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 12), padding=10, relief="flat",
                        background=self.button_bg, foreground="white")
        style.map("TButton", background=[("active", self.button_hover)], foreground=[("active", "white")])

        title = tk.Label(self.root, text="✨ EchoVerse ✨", font=("Segoe UI", 24, "bold"),
                         bg=self.bg_color, fg=self.fg_color)
        title.pack(pady=10)

        self.hamburger_btn = tk.Button(self.root, text="☰", font=("Segoe UI", 18), command=self.toggle_menu,
                                       bg=self.button_bg, fg="white", bd=0, relief="flat")
        self.hamburger_btn.place(x=10, y=10, width=50, height=50)

        self.menu_buttons = [
            ("🌗 Toggle Theme", self.toggle_theme),
            ("📂 Open Test Images", self.open_test_images_folder),
            ("📊 Confusion Matrix", self.toggle_confusion_matrix),
            ("❌ Exit", self.root.destroy)
        ]

        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(pady=20)

        def create_button(text, command, col):
            btn = tk.Button(btn_frame, text=text, command=command, font=("Segoe UI", 12),
                            bg=self.button_bg, fg="white", relief="flat", bd=0,
                            activebackground=self.button_hover)
            btn.grid(row=0, column=col, padx=15, ipadx=20, ipady=10)

        create_button("📁 Upload Image", self.upload_image, 0)
        create_button("📷 Use Webcam", self.open_webcam, 1)
        create_button("🗑️ Clear", self.clear_display, 2)
        create_button("🧹 Delete Last", self.delete_last_character, 3)

        self.result_frame = tk.Frame(self.root, bg=self.frame_color, highlightbackground="#333", highlightthickness=1, padx=10, pady=10)
        self.result_frame.pack(pady=5)
        self.result_label = tk.Label(self.result_frame, text="Prediction: ", font=("Segoe UI", 14, "bold"), bg=self.frame_color, fg=self.fg_color)
        self.result_label.pack()
        self.processing_label = tk.Label(self.result_frame, text="", font=("Segoe UI", 10), bg=self.frame_color, fg="gray")
        self.processing_label.pack()

        self.sentence_frame = tk.Frame(self.root, bg=self.frame_color, highlightbackground="#333", highlightthickness=1, padx=10, pady=10)
        self.sentence_frame.pack(pady=5)
        self.sentence_label = tk.Label(self.sentence_frame, text="Sentence: ", font=("Segoe UI", 14), bg=self.frame_color, fg=self.fg_color)
        self.sentence_label.pack()

        self.img_label = tk.Label(self.root, bg=self.bg_color)
        self.img_label.pack(pady=10)

        self.webcam_status = tk.Label(self.root, text="", font=("Segoe UI", 10, "italic"), bg=self.bg_color, fg="#666")
        self.webcam_status.pack(pady=5)

    def toggle_menu(self):
        if hasattr(self, "menu_container") and self.menu_container.winfo_ismapped():
            self.menu_container.place_forget()
        else:
            if hasattr(self, "menu_container"):
                self.menu_container.place(x=10, y=70)
                return

            self.menu_container = tk.Frame(self.root, bg=self.bg_color, bd=0)
            self.menu_container.place(x=10, y=70)
            canvas = tk.Canvas(self.menu_container, width=200, height=180,
                               bg=self.bg_color, bd=0, highlightthickness=0)
            canvas.create_oval(-100, 0, 300, 180, fill=self.button_bg, outline="")
            canvas.pack()

            self.menu_frame = tk.Frame(self.menu_container, bg=self.button_bg)
            self.menu_frame.place(x=10, y=20)

            for i, (text, command) in enumerate(self.menu_buttons):
                btn = tk.Button(self.menu_frame, text=text, command=command,
                                font=("Segoe UI", 10), bg=self.button_bg, fg="white", relief="flat", anchor="w")
                btn.pack(fill="x", padx=10, pady=5)

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
        

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(128, 128), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

    def predict_image(self, img_input):
        if isinstance(img_input, str):
            img_array = self.preprocess_image(img_input)
            original_img = Image.open(img_input).convert("L").resize((128, 128))
        else:
            gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            img_array = np.expand_dims(resized, axis=(0, -1)) / 255.0
            original_img = Image.fromarray(resized)

        prediction = self.loaded_model.predict(img_array)
        max_index = np.argmax(prediction[0])
        label = self.class_labels.get(max_index, "Unknown")
        confidence = prediction[0][max_index]

        self.result_label.config(text=f"Prediction: {label} ({confidence:.2f})")
        self.current_sentence += label
        self.sentence_label.config(text="Sentence: " + self.current_sentence)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{confidence:.2f}_{timestamp}.png"
        try:
            original_img.save(os.path.join(self.test_images_dir, filename))
            self.display_image(original_img)
        except Exception as e:
            print(f"Error saving image: {e}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.predict_image(file_path)

    def display_image(self, img):
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def clear_display(self):
        self.current_sentence = ""
        self.result_label.config(text="Prediction: ")
        self.sentence_label.config(text="Sentence: ")
        self.img_label.config(image="")

    def delete_last_character(self):
        self.current_sentence = self.current_sentence[:-1]
        self.sentence_label.config(text="Sentence: " + self.current_sentence)

    def open_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        self.webcam_status.config(text="Webcam active. Press 'c' to capture, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
            key = cv2.waitKey(1)

            if key == ord('c'):
                self.predict_image(frame)
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.webcam_status.config(text="")


# Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
