import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils import emotion_labels

model = load_model('../models/modelo_treinamento.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_available_cameras():
    index = 0
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras

# interface grafica
root = tk.Tk()
root.title("Seleção de Câmera para Reconhecimento de Emoções")

# instrução
label = tk.Label(root, text="Selecione a câmera:")
label.pack(pady=5)

# opcoes de cameras disponíveis
camera_var = tk.IntVar()
available_cameras = get_available_cameras()
camera_combobox = ttk.Combobox(root, values=available_cameras, textvariable=camera_var)
camera_combobox.pack(pady=5)

def start_camera():
    camera_index = camera_var.get()
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Erro: Não foi possível acessar a câmera com índice: {camera_index}.")
        sys.exit(1)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            predictions = model.predict(roi_gray)
            max_index = np.argmax(predictions[0])
            emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de Emoç~eos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

        root.after(10, update_frame)
    update_frame()

start_button = tk.Button(root, text="Iniciar", command=start_camera)
start_button.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", root.quit)

root.mainloop()
