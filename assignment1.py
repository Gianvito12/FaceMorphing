import cv2
import numpy as np
import sys
from affine_tools import resizer,aligner,get_landmarks,check_landmarks,delaunay_triangulation,warping
def main():
    if len(sys.argv) != 3:
        print("Uso: python script.py <percorso_immagine_sorgente> <percorso_immagine_destinazione>")
        sys.exit(1)

    img_src_path = sys.argv[1]
    img_tgt_path = sys.argv[2]

    #Legge l'immagine e la trasforma in BGR
    img_src = cv2.imread(img_src_path,cv2.IMREAD_COLOR)
    img_tgt = cv2.imread(img_tgt_path,cv2.IMREAD_COLOR)

    
    if img_src is None or img_tgt is None:
        print("Errore: Impossibile leggere una o entrambe le immagini.")
        sys.exit(1)


    img_src,img_tgt=resizer(img_src,img_tgt)

    # Allinea le immagini
    img_src, img_tgt = aligner(img_src, img_tgt)

    if img_tgt is None or img_src is None:
        print("Errore: Impossibile allineare le immagini.")
        sys.exit(1)


    # Estrai i landmark della prima immagine
    landmarks1 = get_landmarks(img_src)

    # Estrai i landmark della seconda immagine
    landmarks2 = get_landmarks(img_tgt)

    if landmarks1 is None or landmarks2 is None:
        sys.exit(1)

    #Controlla eventuali landmark ripetuti
    landmarks1,landmarks2 = check_landmarks(landmarks1,landmarks2)

    # Calcola la triangolazione di Delaunay sui landmark
    triangles = delaunay_triangulation(img_src,landmarks1)

    # Definisce il numero di fotogrammi intermedi
    numero_immagini = 11
    t = np.linspace(0, 1, numero_immagini, endpoint=True)

    # Configura il writer video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_none.avi', fourcc=fourcc, fps=3, frameSize=(img_src.shape[1], img_src.shape[0]))
    #Genera i fotogrammi intermedi
    for j in range(len(t)):
        # Calcola i landmark intermedi
        intermediate_landmarks = []
        print(t[j])

        # Interpola le posizioni dei landmark
        for i in range(len(landmarks1)):
            # Calcola coordinate x interpolate
            x = np.float32(landmarks1[i][0] * (1 - t[j]) + landmarks2[i][0] * t[j])
            # Calcola coordinate y interpolate
            y = np.float32(landmarks1[i][1] * (1 - t[j]) + landmarks2[i][1] * t[j])

            intermediate_landmarks.append((x, y))

        # Applica warping alle immagini
        warped_img1 = warping(landmarks1, intermediate_landmarks, triangles, img_src)
        warped_img2 = warping(landmarks2, intermediate_landmarks, triangles, img_tgt)
        warped_tot = cv2.addWeighted(warped_img1,1-t[j],warped_img2,t[j],0.0)


        out.write(warped_tot)
    # Rilascia il writer video
    out.release()

if __name__ == "__main__":
    main()