import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math
import time

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
comandoAtivo = True
ultimoGestoPlayPause = 0
cooldown = 3  # Cooldown de 3 segundos

def dedoLevantado(ponta, base):
    return ponta.y < base.y

def dedoAbaixado(ponta, base):
    return ponta.y > base.y

def gestoHangLoose(pontosDaMao):
    polegarLevantado = dedoLevantado(pontosDaMao.landmark[mpMaos.HandLandmark.THUMB_TIP],
                                      pontosDaMao.landmark[mpMaos.HandLandmark.THUMB_MCP])
    minimoLevantado = dedoLevantado(pontosDaMao.landmark[mpMaos.HandLandmark.PINKY_TIP],
                                    pontosDaMao.landmark[mpMaos.HandLandmark.PINKY_MCP])
    indicadorAbaixado = dedoAbaixado(pontosDaMao.landmark[mpMaos.HandLandmark.INDEX_FINGER_TIP],
                                    pontosDaMao.landmark[mpMaos.HandLandmark.INDEX_FINGER_MCP])
    medioAbaixado = dedoAbaixado(pontosDaMao.landmark[mpMaos.HandLandmark.MIDDLE_FINGER_TIP],
                                 pontosDaMao.landmark[mpMaos.HandLandmark.MIDDLE_FINGER_MCP])
    anelarAbaixado = dedoAbaixado(pontosDaMao.landmark[mpMaos.HandLandmark.RING_FINGER_TIP],
                                  pontosDaMao.landmark[mpMaos.HandLandmark.RING_FINGER_MCP])
    
    return polegarLevantado and minimoLevantado and indicadorAbaixado and medioAbaixado and anelarAbaixado

def gestoReset(pontosDaMao):
    return all(dedoAbaixado(pontosDaMao.landmark[i], pontosDaMao.landmark[i-2])
               for i in [mpMaos.HandLandmark.THUMB_TIP,
                        mpMaos.HandLandmark.INDEX_FINGER_TIP,
                        mpMaos.HandLandmark.MIDDLE_FINGER_TIP,
                        mpMaos.HandLandmark.RING_FINGER_TIP,
                        mpMaos.HandLandmark.PINKY_TIP])

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

            # Obtém as coordenadas dos pontos dos dedos
            pontaPolegar = pontosDaMao.landmark[mpMaos.HandLandmark.THUMB_TIP]
            pontaIndicador = pontosDaMao.landmark[mpMaos.HandLandmark.INDEX_FINGER_TIP]
            pontaMeio = pontosDaMao.landmark[mpMaos.HandLandmark.MIDDLE_FINGER_TIP]
            pontaAnelar = pontosDaMao.landmark[mpMaos.HandLandmark.RING_FINGER_TIP]
            pontaMinimo = pontosDaMao.landmark[mpMaos.HandLandmark.PINKY_TIP]

            # Verifica os gestos
            if gestoReset(pontosDaMao):
                comandoAtivo = False
                estadoAnterior = None
                coordenadasAnteriores = None
                print("Gesto de Reset detectado. Todos os comandos foram parados.")
            elif gestoHangLoose(pontosDaMao):
                if comandoAtivo:
                    current_time = time.time()
                    if current_time - ultimoGestoPlayPause > cooldown:
                        pyautogui.press('space')  # Espaço para play/pause
                        print("Gesto de Hang Loose detectado.")
                        ultimoGestoPlayPause = current_time
                estadoAnterior = None
            else:
                if not comandoAtivo:
                    comandoAtivo = True  # Retoma o comando se a mão não estiver fechada

                if comandoAtivo:
                    # Verifica o gesto de controle de volume
                    polegarLevantado = dedoLevantado(pontaPolegar, pontosDaMao.landmark[mpMaos.HandLandmark.THUMB_MCP])
                    indicadorLevantado = dedoLevantado(pontaIndicador, pontosDaMao.landmark[mpMaos.HandLandmark.INDEX_FINGER_MCP])
                    medioLevantado = dedoLevantado(pontaMeio, pontosDaMao.landmark[mpMaos.HandLandmark.MIDDLE_FINGER_MCP])
                    anelarLevantado = dedoLevantado(pontaAnelar, pontosDaMao.landmark[mpMaos.HandLandmark.RING_FINGER_MCP])
                    minimoLevantado = dedoLevantado(pontaMinimo, pontosDaMao.landmark[mpMaos.HandLandmark.PINKY_MCP])

                    # Verifica se o indicador e polegar estão levantados e os outros dedos estão abaixados
                    if polegarLevantado and indicadorLevantado and not (medioLevantado or anelarLevantado or minimoLevantado):
                        # Converte as coordenadas normalizadas para a escala da imagem
                        altura, largura, _ = quadro.shape
                        polegarX, polegarY = int(pontaPolegar.x * largura), int(pontaPolegar.y * altura)
                        indicadorX, indicadorY = int(pontaIndicador.x * largura), int(pontaIndicador.y * altura)
                        palmaX = int(pontosDaMao.landmark[mpMaos.HandLandmark.WRIST].x * largura)

                        # Calcula a distância entre o polegar e o indicador
                        distancia = math.hypot(indicadorX - polegarX, indicadorY - polegarY)

                        # Define os limites para a distância e o volume
                        minDistancia = 20
                        maxDistancia = 200
                        minVolume = 0
                        maxVolume = 100

                        # Converte a distância para o valor de volume
                        volume = np.interp(distancia, [minDistancia, maxDistancia], [minVolume, maxVolume])

                        # Executa o comando de volume apenas se o estado mudou
                        if estadoAnterior is not None and volume != estadoAnterior:
                            if volume > estadoAnterior:
                                pyautogui.press('volumeup')
                            elif volume < estadoAnterior:
                                pyautogui.press('volumedown')

                        estadoAnterior = volume  # Atualiza o estadoAnterior com o volume atual

                        # Detecta movimentos de swipe (deslize)
                        if coordenadasAnteriores:
                            deltaX = palmaX - coordenadasAnteriores[0]

                            # Verifica se o movimento é predominantemente horizontal
                            if abs(deltaX) > 10:  # Limite para considerar um swipe
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
 