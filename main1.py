import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict

def analisar_alinhamento_fardo(caminho_imagem, caminho_modelo):
    """
    Analisa a imagem de um fardo para verificar se está alinhado ou não.
    """
    if not os.path.exists(caminho_imagem):
        print(f"Erro: Imagem não encontrada em '{caminho_imagem}'")
        return None, "Erro: Imagem não encontrada"
    if not os.path.exists(caminho_modelo):
        print(f"Erro: Modelo não encontrado em '{caminho_modelo}'")
        return None, "Erro: Modelo não encontrado"

    try:
        modelo = YOLO(caminho_modelo)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None, "Erro ao carregar modelo"

    imagem = cv2.imread(caminho_imagem)
    imagem_resultado = imagem.copy()

    resultados = modelo(imagem, verbose=False)

    centros = []
    alturas_caixas = []
    for resultado in resultados:
        boxes = resultado.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianca = box.conf[0]

            if confianca > 0.5:
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                centros.append((centro_x, centro_y))
                alturas_caixas.append(y2 - y1)

    status_alinhamento = "Nao Alinhado"
    cor_status = (0, 0, 255)

    if len(centros) >= 4:
        pontos = np.array(centros, dtype=np.float32)
        rect = cv2.minAreaRect(pontos)
        angulo = rect[2]
        matriz_rotacao = cv2.getRotationMatrix2D(tuple(rect[0]), angulo, 1.0)
        pontos_rotacionados = cv2.transform(np.array([pontos]), matriz_rotacao)[0]
        pontos_rotacionados = sorted(pontos_rotacionados, key=lambda p: (p[1], p[0]))

        if alturas_caixas:
            altura_media = np.mean(alturas_caixas)
            TOLERANCIA_Y = altura_media * 0.5
        else:
            TOLERANCIA_Y = 20

        linhas = defaultdict(list)
        linha_atual = []
        ponto_referencia_y = pontos_rotacionados[0][1]

        for ponto in pontos_rotacionados:
            if abs(ponto[1] - ponto_referencia_y) < TOLERANCIA_Y:
                linha_atual.append(ponto)
            else:
                linhas[ponto_referencia_y] = sorted(linha_atual, key=lambda p: p[0])
                linha_atual = [ponto]
                ponto_referencia_y = ponto[1]
        linhas[ponto_referencia_y] = sorted(linha_atual, key=lambda p: p[0])

        contagens_por_linha = [len(l) for l in linhas.values()]

        if len(set(contagens_por_linha)) == 1:
            status_alinhamento = "Alinhado"
            cor_status = (0, 255, 0)
        else:
            status_alinhamento = "Nao Alinhado"
            cor_status = (0, 0, 255)

    for ponto in centros:
        cv2.circle(imagem_resultado, (int(ponto[0]), int(ponto[1])), 5, (0, 0, 255), -1)

    cv2.putText(imagem_resultado, status_alinhamento, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_status, 2, cv2.LINE_AA)

    return imagem_resultado, status_alinhamento

# --- Execução do Script ---
if __name__ == "__main__":
    MODELO_PATH = "pesos/best.pt"
    NOME_IMAGEM = "fardo_teste5.jpg"

    IMAGEM_TESTE_PATH = os.path.join("imagens_teste", NOME_IMAGEM)
    print(f"Analisando a imagem: {IMAGEM_TESTE_PATH}")

    imagem_final, status = analisar_alinhamento_fardo(IMAGEM_TESTE_PATH, MODELO_PATH)

    if imagem_final is not None:
        print(f"Resultado da análise: {status}")
        cv2.imshow("Resultado da Analise", imagem_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
