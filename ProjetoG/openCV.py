import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# Configura MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inicializa a captura de vídeo
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro ao abrir a câmera!")
    exit()

while True:
    # Captura o frame da câmera
    ret, frame = camera.read()
    if not ret:
        print("Falha ao capturar o frame!")
        break

    # Converte a imagem para RGB (necessário para MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Desenha os pontos da mão
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Obtém as coordenadas dos pontos dos dedos (por exemplo, dedo indicador e mínimo)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Converte as coordenadas normalizadas para a escala da imagem
            h, w, _ = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            pinky_x, pinky_y = int(pinky_tip.x * w), int(pinky_tip.y * h)

            # Exemplo de controle baseado na posição dos pontos dos dedos
            # Ajuste as condições de controle conforme necessário
            if index_y < pinky_y:
                pyautogui.press('volumeup')
            elif index_y > pinky_y:
                pyautogui.press('volumedown')

    # Exibe o frame
    cv2.imshow('Camera Feed', frame)

    # Sai do loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a câmera e feche todas as janelas
camera.release()
cv2.destroyAllWindows()
