import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageFont, ImageDraw
import threading

# Загрузка предобученной модели
from keras_preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

model = load_model('fer_model.h5')

emotion_labels = ['Злость', 'Отвращение', 'Испуг', 'Счастье', 'Грусть', 'Нейтральное', 'Удивление']

font_path = 'DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 20)


# Функция для распознавания эмоций
def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    predictions = model.predict(face)[0]
    return predictions


class EmotionRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognizer")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Создание вкладок
        self.create_image_tab()
        self.create_video_tab()
        self.create_webcam_tab()

    def create_image_tab(self):
        image_tab = ttk.Frame(self.notebook)
        self.notebook.add(image_tab, text='Загрузка изображения')

        self.image_label = tk.Label(image_tab)
        self.image_label.pack(side="left", fill="both", expand=True)

        self.image_stats_frame = ttk.Frame(image_tab, padding="10")
        self.image_stats_frame.pack(side="right", fill="both", expand=True)

        self.figure_image, self.ax_image = plt.subplots()
        self.canvas_image = FigureCanvasTkAgg(self.figure_image, master=self.image_stats_frame)
        self.canvas_image.get_tk_widget().pack(side="top", fill="both", expand=True)

        load_image_button = ttk.Button(image_tab, text='Загрузить изображение', command=self.load_image)
        load_image_button.pack(side="bottom")

    def create_video_tab(self):
        video_tab = ttk.Frame(self.notebook)
        self.notebook.add(video_tab, text='Загрузка видео')

        self.video_label = tk.Label(video_tab)
        self.video_label.pack(side="left", fill="both", expand=True)

        self.video_stats_frame = ttk.Frame(video_tab, padding="10")
        self.video_stats_frame.pack(side="right", fill="both", expand=True)

        self.figure_video, self.ax_video = plt.subplots()
        self.canvas_video = FigureCanvasTkAgg(self.figure_video, master=self.video_stats_frame)
        self.canvas_video.get_tk_widget().pack(side="top", fill="both", expand=True)

        load_video_button = ttk.Button(video_tab, text='Загрузить видео', command=self.load_video)
        load_video_button.pack(side="bottom")

    def create_webcam_tab(self):
        webcam_tab = ttk.Frame(self.notebook)
        self.notebook.add(webcam_tab, text='Поток с веб-камеры')

        self.video_frame = tk.Label(webcam_tab)
        self.video_frame.pack(side="left", fill="both", expand=True)

        self.webcam_stats_frame = ttk.Frame(webcam_tab, padding="10")
        self.webcam_stats_frame.pack(side="right", fill="both", expand=True)

        self.figure_webcam, self.ax_webcam = plt.subplots()
        self.canvas_webcam = FigureCanvasTkAgg(self.figure_webcam, master=self.webcam_stats_frame)
        self.canvas_webcam.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.video_capture = cv2.VideoCapture(0)
        self.update_webcam()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            emotions, annotated_image = self.process_frame(image)
            self.display_image(annotated_image, self.image_label, emotions, self.ax_image, self.canvas_image)

    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            cap = cv2.VideoCapture(file_path)
            all_emotions = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                emotions, annotated_frame = self.process_frame(frame)
                all_emotions.append(emotions)

                self.display_video(annotated_frame, self.video_label, emotions, self.ax_video, self.canvas_video)
                self.root.update_idletasks()
                self.root.update()
            cap.release()

            avg_emotions = self.calculate_average_emotions(all_emotions)
            self.update_plot(avg_emotions, self.ax_video, self.canvas_video)

    def update_webcam(self):
        ret, frame = self.video_capture.read()
        if ret:
            emotions, annotated_frame = self.process_frame(frame)
            self.display_video(annotated_frame, self.video_frame, emotions, self.ax_webcam, self.canvas_webcam)
        self.root.after(10, self.update_webcam)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        emotions = {i: {label: 0 for label in emotion_labels} for i in range(len(faces))}
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)

        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y + h, x:x + w]
            predictions = predict_emotion(face)
            emotion_label_idx = np.argmax(predictions)
            emotion_label = emotion_labels[emotion_label_idx]
            emotion_probability = np.max(predictions)

            draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
            draw.text((x, y - 30), f"Лицо {i + 1}. {emotion_label}", font=font, fill=(0, 255, 0, 255))

            for idx, label in enumerate(emotion_labels):
                emotions[i][label] = predictions[idx]

        annotated_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        return emotions, annotated_frame

    def display_image(self, frame, label, emotions, ax, canvas):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.update_plot(emotions, ax, canvas)

    def display_video(self, frame, label, emotions, ax, canvas):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.update_plot(emotions, ax, canvas)

    def update_plot(self, emotions, ax, canvas):
        ax.clear()
        for i, (face_id, emotion_probs) in enumerate(emotions.items()):
            ax.bar(np.arange(len(emotion_labels)) + i * 0.1, [emotion_probs[label] for label in emotion_labels], width=0.1, label=f'Лицо {face_id + 1}')
        ax.set_xticks(np.arange(len(emotion_labels)) + 0.1)
        ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
        ax.set_ylabel('Вероятность')
        ax.set_title('Распределение эмоций')
        ax.legend()
        canvas.draw()

    def calculate_average_emotions(self, all_emotions):
        avg_emotions = {i: {label: 0 for label in emotion_labels} for i in range(len(all_emotions[0]))}
        for emotions in all_emotions:
            for i, emotion_probs in emotions.items():
                for label, prob in emotion_probs.items():
                    avg_emotions[i][label] += prob
        for i, emotion_probs in avg_emotions.items():
            for label in emotion_labels:
                avg_emotions[i][label] /= len(all_emotions)
        return avg_emotions

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerApp(root)
    root.mainloop()