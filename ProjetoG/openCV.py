import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# Configura MediaPipe Hands
mpMaos = mp.solutions.hands
mpDesenho = mp.solutions.drawing_utils
maos = mpMaos.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Inicializa a captura de vídeo
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro ao abrir a câmera!")
    exit()

# Variáveis para armazenar o estado anterior e as coordenadas anteriores
estadoAnterior = None
coordenadasAnteriores = None

while True:
    # Captura o quadro da câmera
    ret, quadro = camera.read()
    if not ret:
        print("Falha ao capturar o quadro!")
        break

    # Converte a imagem para RGB (necessário para MediaPipe)
    quadroRgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
    resultados = maos.process(quadroRgb)

    # Desenha os pontos da mão
    if resultados.multi_hand_landmarks:
        for pontosDaMao in resultados.multi_hand_landmarks:
            mpDesenho.draw_landmarks(quadro, pontosDaMao, mpMaos.HAND_CONNECTIONS)

            # Obtém as coordenadas dos pontos dos dedos (ex: dedo indicador e dedo mínimo)
            pontaIndicador = pontosDaMao.landmark[mpMaos.HandLandmark.INDEX_FINGER_TIP]
            pontaMinimo = pontosDaMao.landmark[mpMaos.HandLandmark.PINKY_TIP]
            palma = pontosDaMao.landmark[mpMaos.HandLandmark.WRIST]

            # Converte as coordenadas normalizadas para a escala da imagem
            altura, largura, _ = quadro.shape
            indicadorY = int(pontaIndicador.y * altura)
            minimoY = int(pontaMinimo.y * altura)
            palmaX = int(palma.x * largura)

            # Determina o estado atual do gesto de volume
            if indicadorY < minimoY:
                estadoAtual = 'volumeUp'
            elif indicadorY > minimoY:
                estadoAtual = 'volumeDown'
            else:
                estadoAtual = 'neutro'

            # Executa o comando de volume apenas se o estado mudou
            if estadoAtual != estadoAnterior:
                if estadoAtual == 'volumeUp':
                    pyautogui.press('volumeup')
                elif estadoAtual == 'volumeDown':
                    pyautogui.press('volumedown')
                estadoAnterior = estadoAtual

            # Detecta movimentos de swipe (deslize)
            if coordenadasAnteriores:
                deltaX = palmaX - coordenadasAnteriores[0]

                # Verifica se o movimento é predominantemente horizontal
                if abs(deltaX) > 30:  # Limite para considerar um swipe
                    if deltaX > 0:  # Swipe para a direita
                        pyautogui.press('nexttrack')  # Passa para a próxima música/foto
                        print("Swipe para a direita detectado.")
                    else:  # Swipe para a esquerda
                        pyautogui.press('prevtrack')  # Volta para a música/foto anterior
                        print("Swipe para a esquerda detectado.")

            # Atualiza as coordenadas anteriores
            coordenadasAnteriores = (palmaX, indicadorY)

    else:
        coordenadasAnteriores = None  # Reseta as coordenadas quando a mão não é detectada

    # Exibe o quadro
    cv2.imshow('Camera Feed', quadro)

    # Sai do loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a câmera e feche todas as janelas
camera.release()
cv2.destroyAllWindows()
