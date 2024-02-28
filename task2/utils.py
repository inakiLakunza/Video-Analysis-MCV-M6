import numpy as np
import cv2
import datetime
from lxml import etree


import cv2
import numpy as np
import os

from PIL import Image


def create_gif(image_paths, output_gif_path, duration=500):
    images = [Image.open(image_path) for image_path in image_paths[:50]]
    # Save as GIF
    images[0].save(
    output_gif_path,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=1 # 0 means infinite loop
 )


def read_video(vid_path: str, color_space: str, max_iterations: int = 1000):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    rgb_frames = []
    color_frames = []
    iterations = 0
    while iterations < max_iterations:
        ret, frame = vid.read()
        if ret:
            if color_space == 'RGB':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                color_frames.append(frame_rgb)
            elif color_space == 'HSV':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_frames.append(frame_rgb)
            elif color_space == 'Lab':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                color_frames.append(frame_rgb)
            elif color_space == 'YUV':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                color_frames.append(frame_rgb)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame_rgb)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
            iterations += 1
        else:
            break
    vid.release()
    return np.array(frames), np.array(color_frames), np.array(rgb_frames)


def compute_ap(gt_boxes, pred_boxes):
    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes))
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.

    # Iterate over the predicted boxes
    for i, pred_box in enumerate(pred_boxes):
        ious = [binaryMaskIOU(pred_box, gt_box) for gt_box in gt_boxes]
        if len(ious) == 0:
            fp[i] = 1
            continue
        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)

        if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
            tp[i] = 1
            gt_matched[max_iou_idx] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(gt_boxes)
    # if len(gt_boxes) > 0:
    #     recall = tp / len(gt_boxes)
    # else:
    #     recall = 0
    precision = tp / (tp + fp)

    # Generate graph with the 11-point interpolated precision-recall curve
    recall_interp = np.linspace(0, 1, 11)
    precision_interp = np.zeros(11)
    for i, r in enumerate(recall_interp):
        array_precision = precision[recall >= r]
        if len(array_precision) == 0:
            precision_interp[i] = 0
        else:
            precision_interp[i] = max(precision[recall >= r])

    ap = np.mean(precision_interp)
    return ap
    
def compute_weighted_avg(matrix:np.ndarray, weights:np.ndarray):
    # Calcula el producto de cada matriz de medias por su matriz de pesos correspondiente
    if len(matrix) > 10:
        matrix = matrix[:10]
        weights = weights[:10]
    productos = [media * peso for media, peso in zip(matrix, weights)]
    
    # Suma los productos para obtener la suma ponderada
    suma_ponderada = sum(productos)
    
    # Suma de los pesos para normalizar
    suma_pesos = sum(weights)
    
    # Calcula la media ponderada
    media_ponderada = suma_ponderada / suma_pesos
    
    return media_ponderada

    


def compute_gaussian_weighted_avg(matrix: np.ndarray, sigma: float = 1.2):
    n = len(matrix)
    if n > 10:
        matrix = matrix[:10]

    # Create weights based on a Gaussian distribution centered at the midpoint
    half_index = n // 2
    weights = np.exp(-0.5 * ((np.arange(n) - half_index) / sigma) ** 2)

    # Normalize weights to sum up to 1
    weights /= np.sum(weights)

    # Calculate the weighted average
    weighted_avg = np.average(matrix, weights=weights, axis=0)

    return weighted_avg


def read_annotations(annotations_path: str):
    """
    Function to read the GT annotations from ai_challenge_s03_c010-full_annotation.xml

    At the moment we will only check that the track is for "car" and has "parked" as false
    and we will save the bounding box attributes from the 'box' element.
    """
    tree = etree.parse(annotations_path)
    root = tree.getroot()
    car_boxes = {}

    for track in root.xpath(".//track[@label='car']"):
        track_id = track.get("id")
        for box in track.xpath(".//box"):
            parked_attribute = box.find(".//attribute[@name='parked']")
            if parked_attribute is not None and parked_attribute.text == 'false':
                frame = box.get("frame")
                box_attributes = {
                    "xtl": box.get("xtl"),
                    "ytl": box.get("ytl"),
                    "xbr": box.get("xbr"),
                    "ybr": box.get("ybr"),
                    # in the future we will need more attributes
                }
                if frame in car_boxes:
                    car_boxes[frame].append(box_attributes)
                else:
                    car_boxes[frame] = [box_attributes]

    return car_boxes


def split_frames(frames):
    """
    Returns 25% and 75% split partition of frames.
    """
    return frames[:int(frames.shape[0] * 0.25)], frames[int(frames.shape[0] * 0.25):]


def make_video(estimation):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=uint8)
    """
    size = estimation.shape[1], estimation.shape[2]
    duration = estimation.shape[0]
    fps = 10
    out = cv2.VideoWriter(f'./last_frames_estimation_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for i in range(duration):
        data = (estimation[i]).astype(np.uint8)
        # I am converting the data to gray but we should look into this...
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        out.write(data)
    out.release()

def compute_metric(mask1_list, mask2_list, threshold=0.5):
    val = 0
    for mask1 in mask1_list:
        score = 0
        for mask2 in mask2_list:
            IoU = binaryMaskIOU(mask1, mask2)
            if IoU > score:
                score = IoU
        if score > threshold:
            val += 1
    if len(mask1_list) > 0:
        val = val / len(mask1_list)
    else:
        val = 0
    return val


def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1 == 255, mask2 == 255))
    union = mask1_area+mask2_area-intersection
    if union == 0: # Evitar dividir entre 0
        return 0
    iou = intersection/(union)
    return iou



def save_img(img, rho, idx, directorio='./images/pruebas_means'):
    """
    Guarda la imagen correspondiente a la media en un archivo PNG.

    Argumentos:
    mean: Matriz NumPy, la imagen de la media a guardar.
    rho: Valor utilizado en el nombre del archivo para identificar la imagen.
    idx: Índice utilizado en el nombre del archivo para identificar la imagen.
    directorio: Directorio donde se guardará la imagen. Por defecto es './images/pruebas_means'.

    Retorna:
    Nada. La función guarda la imagen en el directorio especificado.
    """
    # Nombre del archivo de imagen
    os.makedirs(directorio, exist_ok = True)

    filename_mean = f'result_{str(rho)}_{idx}.png'

    # Guardar la imagen usando OpenCV
    cv2.imwrite(os.path.join(directorio, filename_mean), img.astype("uint8"))

def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1 == 255, mask2 == 255))
    union = mask1_area+mask2_area-intersection
    if union == 0: # Evitar dividir entre 0
        return 0
    iou = intersection/(union)
    return iou


def MeanIOU(gt_dict, pred_dict):
    result = 0
    for name, mask in gt_dict.items():
        result += binaryMaskIOU(gt_dict[name], pred_dict[name])
    
    result /= len(gt_dict)
    return result


def calcular_pesos_exponenciales(num_iterations, decay:float=0.7):
    pesos = np.exp(np.linspace(-decay, 0, num_iterations))
    pesos = pesos / np.sum(pesos)  # Normalizar los pesos para que sumen 1
    return pesos



if __name__ == "__main__":
    import os
    import pickle
    import matplotlib.pyplot as plt
    import glob
    

    image_list = []
    filenames = []

    for filename in glob.glob('images/pruebas_foreground_0.2_3.5/*.png'): 
        path = (os.path.join(filename))
        filenames.append(path)
    
    filenames = (sorted(filenames, key=lambda x: int(x.split("/")[-1].split(".")[-2].split("_")[-1]))[:60])
    for path in filenames:
        image_list.append(cv2.imread(path))

    # Output GIF path
    # Create GIF
    make_video(np.array(image_list))

