import cv2
import numpy as np
from ultralytics import YOLO
import os
import math

def distancia(p1, p2):
    """Calcula a distância entre dois pontos."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def eh_triangulo_retangulo(p1, p2, p3, tolerancia=0.05):
    """
    Verifica se os três pontos formam um triângulo retângulo.
    Usa o Teorema de Pitágoras com tolerância percentual.
    """
    d1 = distancia(p1, p2)
    d2 = distancia(p2, p3)
    d3 = distancia(p1, p3)

    lados = sorted([d1, d2, d3])  # menor -> maior
    cat1, cat2, hip = lados

    # Verifica se cat1² + cat2² ≈ hip²
    return abs((cat1**2 + cat2**2) - (hip**2)) <= tolerancia * (hip**2)

def analisar_alinhamento_fardo(caminho_imagem, caminho_modelo):
    """
    Analisa a imagem do fardo verificando alinhamento com base
    na formação de um triângulo retângulo usando as extremidades.
    """
    if not os.path.exists(caminho_imagem):
        return None, "Erro: Imagem não encontrada"
    if not os.path.exists(caminho_modelo):
        return None, "Erro: Modelo não encontrado"

    try:
        modelo = YOLO(caminho_modelo)
    except Exception as e:
        return None, f"Erro ao carregar modelo: {e}"

    imagem = cv2.imread(caminho_imagem)
    imagem_resultado = imagem.copy()

    resultados = modelo(imagem, verbose=False)

    centros = []
    for resultado in resultados:
        boxes = resultado.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confianca = box.conf[0]
            if confianca > 0.5:
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                centros.append((centro_x, centro_y))

    status_alinhamento = "Nao Alinhado"
    cor_status = (0, 0, 255)
    motivo = ""

    if len(centros) >= 3:
        # Ordena pelos X (encontrar extremidades esquerda/direita)
        centros_ordenados = sorted(centros, key=lambda p: p[0])
        p_esq = centros_ordenados[0]
        p_dir = centros_ordenados[-1]

        # Pega ponto mais próximo do meio para ser referência
        meio_x = (p_esq[0] + p_dir[0]) / 2
        p_meio = min(centros_ordenados[1:-1], key=lambda p: abs(p[0] - meio_x))

        if eh_triangulo_retangulo(p_esq, p_meio, p_dir):
            status_alinhamento = "Alinhado"
            cor_status = (0, 255, 0)
            motivo = "Triângulo formado pelas extremidades e o ponto central é retângulo (90°)."
        else:
            motivo = "Os pontos das extremidades e o ponto central não formam um triângulo retângulo."
        
        # Desenhar triângulo
        cv2.line(imagem_resultado, p_esq, p_meio, (255, 0, 0), 2)
        cv2.line(imagem_resultado, p_meio, p_dir, (255, 0, 0), 2)
        cv2.line(imagem_resultado, p_dir, p_esq, (255, 0, 0), 2)

        for p in [p_esq, p_meio, p_dir]:
            cv2.circle(imagem_resultado, p, 8, (0, 255, 0), -1)

    else:
        motivo = "Não há pontos suficientes para verificar alinhamento."

    # Desenha todos os centros detectados
    for ponto in centros:
        cv2.circle(imagem_resultado, (int(ponto[0]), int(ponto[1])), 5, (0, 0, 255), -1)

    # Texto principal na imagem
    cv2.putText(imagem_resultado, status_alinhamento, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_status, 2, cv2.LINE_AA)

    return imagem_resultado, f"{status_alinhamento}: {motivo}"


# --- Execução do Script ---
if __name__ == "__main__":
    MODELO_PATH = "pesos/best.pt"
    NOME_IMAGEM = "fardo_teste1.jpg"

    IMAGEM_TESTE_PATH = os.path.join("imagens_teste", NOME_IMAGEM)
    print(f"Analisando a imagem: {IMAGEM_TESTE_PATH}")

    imagem_final, status = analisar_alinhamento_fardo(IMAGEM_TESTE_PATH, MODELO_PATH)

    if imagem_final is not None:
        print(f"Resultado da análise: {status}")
        cv2.imshow("Resultado da Analise", imagem_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
