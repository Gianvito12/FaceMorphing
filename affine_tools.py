import cv2
import numpy as np
import dlib

def resizer(im1,im2):
    area1=im1.shape[0]*im1.shape[1]
    area2=im2.shape[0]*im2.shape[1]
    if area1<area2:
        im2=cv2.resize(im2,(im1.shape[1],im1.shape[0]),interpolation=cv2.INTER_LINEAR)
    elif area1>area2:
        im1=cv2.resize(im1,(im2.shape[1],im2.shape[0]),interpolation=cv2.INTER_LINEAR)
    return im1,im2

def aligner(im1, im2):
    # Carica il classificatore Haar per il rilevamento dei volti
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    
    # Rileva i volti in entrambe le immagini
    faces1 = face_cascade.detectMultiScale(im1, scaleFactor=1.1, minNeighbors=9)
    faces2 = face_cascade.detectMultiScale(im2, scaleFactor=1.1, minNeighbors=9)
    
    if len(faces1) == 0 or len(faces2) == 0:
        print("Errore: non sono stati rilevati volti in una delle immagini.")
        return None, None
    
    # Estrai le coordinate del primo e del secondo
    x1, y1, w1, h1 = faces1[0]
    x2, y2, w2, h2 = faces2[0]
    
    # Ottieni le dimensioni delle immagini
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    
    # Calcola i centri dei volti
    face_center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
    face_center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
    
    # Calcola la traslazione necessaria per centrare i volti
    translation = face_center1 - face_center2
    
    # Matrice di traslazione
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]]
    ], dtype=np.float32)
    
    # Applica la traslazione
    im2_aligned = cv2.warpAffine(im2, translation_matrix, (width2, height2))
    
    # Calcola i fattori di zoom per entrambe le immagini
    # Rapporto tra larghezza target e larghezza del volto
    width_target = 0.67 * im1.shape[1] 
    scale1 = width_target / w1  
    scale2 = width_target / w2  
    
    # Matrici di trasformazione per lo zoom centrato
    zoom_matrix1 = cv2.getRotationMatrix2D(face_center1, 0, scale1)
    zoom_matrix2 = cv2.getRotationMatrix2D(face_center2, 0, scale2)
    
    # Applica lo zoom a entrambe le immagini
    im1_zoomed = cv2.warpAffine(im1, zoom_matrix1, (width1, height1))
    im2_zoomed = cv2.warpAffine(im2_aligned, zoom_matrix2, (width2, height2))
    
    return im1_zoomed, im2_zoomed



def get_landmarks(img):
    # Carica il predittore di landmark facciali pre-addestrato
    predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat"
    )
    
    # Inizializza il rilevatore di volti frontali pre-addestrato
    detector = dlib.get_frontal_face_detector()
    
    # Rileva i volti nell'immagine
    faces = detector(img, 0)
    
    # Ottieni la shape dell'immagine
    height, width = img.shape[:2]

    # Definisci i punti dei landmarks
    landmarks_points = [
        (0, 0),  
        (width - 1, 0), 
        (0, height - 1),  
        (width - 1, height - 1),  
        (0, height // 2),  
        (width - 1, height // 2), 
        (width // 2, 0),  
        (width // 2, height - 1), 
    ]

    # Verifica se sono stati rilevati volti
    if len(faces) > 0:
        face = faces[0]
        # Estrai i 68 landmark facciali
        landmarks = predictor(img, face)
        
        # Aggiungi i landmark facciali alla lista
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))
    else:
        print("Faccia non rilevata")
        return None
    
    return landmarks_points

def check_landmarks(landmarks1, landmarks2):
    list_index=[]
    #Controllo landmarks duplicati  
    for i in range(len(landmarks1)):
        if landmarks1[i] in landmarks1[i+1:]:
            list_index.append(i)
    for i in range(len(landmarks2)):
        if landmarks2[i] in landmarks2[i+1:]:
            if i not in list_index:
                list_index.append(i)
    if list_index:
        # Rimuovi i landmark duplicati da entrambi i set in ordine decrescente
        # per evitare di modificare gli indici degli altri landmark
        for i in sorted(list_index, reverse=True):
            landmarks1.pop(i)
            landmarks2.pop(i)

    return landmarks1, landmarks2

def delaunay_triangulation(img, landmarks):
    # Definisce il rettangolo che contiene l'immagine
    rect = (0, 0, img.shape[1], img.shape[0])
    
    # Crea un'istanza di Subdiv2D per la triangolazione di Delaunay
    subdiv = cv2.Subdiv2D(rect)
    
    # Inserisce i punti dei landmark
    for point in landmarks:
        subdiv.insert(point)
    
    # Ottiene la lista dei triangoli
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    
    # Converte i triangoli nei loro indici corrispondenti nei landmark
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        indices = []
        for pt in [pt1, pt2, pt3]:
            index = landmarks.index(pt) 
            indices.append(index)
        if len(indices) == 3:
            delaunay_triangles.append(indices)
    
    return delaunay_triangles



def warping(landmark_points, intermediate_landmarks, index_triangles, image):
    height, width = image.shape[:2]
    
    # Crea griglie di coordinate x e y usando meshgrid
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_y = map_y.astype(np.float32)
    map_x = map_x.astype(np.float32)
    
    for triangle in index_triangles:
        source_triangle = np.array(
            [
                landmark_points[triangle[0]],
                landmark_points[triangle[1]],
                landmark_points[triangle[2]],
            ],
            np.float32,
        )
        
        destination_triangle = np.array(
            [
                intermediate_landmarks[triangle[0]],
                intermediate_landmarks[triangle[1]],
                intermediate_landmarks[triangle[2]],
            ],
            np.float32,
        )
        
        mask = np.zeros(image.shape[:2])
        warp_matrix = cv2.getAffineTransform(source_triangle, destination_triangle)
        full_warp_matrix = np.vstack((warp_matrix, [0, 0, 1]))
        inverse_matrix = np.linalg.inv(full_warp_matrix)

        # Riempie il triangolo di destinazione nella maschera
        cv2.fillConvexPoly(mask, np.int32(destination_triangle), 255)
        
        # Trova gli elementi interni al triangolo
        triangle_inner_elements = np.nonzero(mask)
        y_to_transform, x_to_transform = triangle_inner_elements

        # Converte le coordinate in coordinate omogenee
        homogeneous_coordinates = np.vstack((x_to_transform,y_to_transform, np.ones_like(x_to_transform)))
        
        # Trasforma le coordinate usando la matrice inversa
        transformed_coordinates = inverse_matrix @ homogeneous_coordinates
        
        # Estrae le coordinate x e y trasformate
        x_transformed = transformed_coordinates[0]
        y_transformed = transformed_coordinates[1]
        
        # Aggiorna le mappe di x e y per i pixel nel triangolo
        map_x[y_to_transform, x_to_transform] = x_transformed
        map_y[y_to_transform, x_to_transform] = y_transformed

        # Crea una maschera booleana per selezionare solo gli indici validi
        valid_mask = (0 <= x_transformed) & (x_transformed < width) & (0 <= y_transformed) & (y_transformed < height)

        # Applica la maschera per aggiornare solo gli elementi validi delle mappe
        map_x[y_to_transform[valid_mask], x_to_transform[valid_mask]] = x_transformed[valid_mask]
        map_y[y_to_transform[valid_mask], x_to_transform[valid_mask]] = y_transformed[valid_mask]
    
    # Applica la trasformazione di rimappatura all'immagine
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return warped_image
