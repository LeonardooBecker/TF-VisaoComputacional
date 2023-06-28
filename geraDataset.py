import torch
import cv2
import os
import pandas as pd

from easy_paddle_ocr import TextRecognizer
text_recognizer = TextRecognizer()

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./last.pt')

video_path="./Front/NO20230518-101357-000018F.mp4"

cap = cv2.VideoCapture(video_path)

_,frame=cap.read()
height, width, channels = frame.shape

arquivo = open('./coordenadasPlacas.txt', 'a')

if not cap.isOpened():
    raise IOError("Não foi possível abrir a webcam")

i = 0

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inferência
    results = model(frame)

    # Confiança mínima
    conf_threshold = 0.8
    for detecao in results.pred:
        if (len(detecao) > 0):
            elemEscolhido=detecao[0]

            maiorConfianca=0
            for det in detecao:
                confiancaAtual=det[4].item()
                if(confiancaAtual>maiorConfianca):
                    maiorConfianca=confiancaAtual
                    elemEscolhido=det
		
            valorClasse = int(elemEscolhido[-1].item())
            if ((elemEscolhido[4].item() > conf_threshold)):
                cv2.imwrite(f'./imgs/frame{i}({valorClasse}).jpg', frame)
                x1=int(elemEscolhido[0].item())
                y1=int(elemEscolhido[1].item())
                x2=int(elemEscolhido[2].item())
                y2=int(elemEscolhido[3].item())
                
                image=frame
                color = (0, 255, 0)
                thickness = 2
                # Desenhar o retângulo na imagem
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                cv2.imwrite(f'./imgs/frame{i}des({valorClasse}).jpg', image)

                with open(f'./imgs/frame{i}.txt', 'w') as f:
                    class_id = valorClasse  # ID da classe, pode variar dependendo do seu conjunto de dados
                    x_min = (x1+(abs(x2-x1)/2)) / width
                    y_min = (y1+(abs(y2-y1)/2)) / height
                    widthImg = abs(x2 - x1) / width
                    heightImg = abs(y2 - y1) / height
                    label = f"{class_id} {x_min} {y_min} {widthImg} {heightImg}"
                    f.write(label + '\n')

                roi=[]
                roi.append( frame[2000:2120,2850:3280])
                roi.append(frame[2000:2120,3300:3700])

                for im in roi:
                    prediction = text_recognizer.read(im)
                    arquivo.write(f'{prediction["text"]};')
                    print(f'[+] text: {prediction["text"]}')
                    print(f'[+] confidence: {int(prediction["confidence"]*100)}%')
                arquivo.write(f"{class_id}\n")
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

arquivo.close()
cap.release()
cv2.destroyAllWindows()
