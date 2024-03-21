import os
import sys

import numpy as np
import cv2
import json
import csv
from tqdm import tqdm
import pickle

import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from visualization import display_tsne_plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from utils import map_labels_to_integers2






# MODEL ====================================================================

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(4096, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        #print(f'Lo que le llega en nuestro embedding {x["pool"].shape} {x["0"].shape} {x["1"].shape} {x["2"].shape}')
        x = x["pool"].flatten(start_dim=1)
        x = self.activation(x)
        x = self.linear(x)
        return x




if __name__ == "__main__":
    
    CAM_1_DONE = True


    TIME_OFFSET: dict[str, float] = {
        "c010": 8.7,
        "c011": 8.5,
        "c012": 5.9,
        "c013": 0. ,
        "c014": 5. ,
        "c015": 8.5
    }

    CSVS_PARKED_REMOVED_ROOT = "./csvs_removed_parked"
    TRIPLET_SAVED_MODELS_ROOT = "./triplet_train/saved_models"

    CAM_ORDER = ["c013", "c014", "c012", "c011", "c015", "c010"]

    FRAMES_PATH_ROOT = "/ghome/group07/test/W4/frame_dataset_PNG"

    CROP_SAVE_PATH = "/ghome/group07/test/W4/part2/crops_main"

    THRESHOLD = 4.

    OUT_CSV_ROOT = "/ghome/group07/test/W4/part2/out_csvs"
    TRY_NAME = "th_4"


    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1').backbone
    embed = EmbeddingLayer(embed_size=2048)
    model = torch.nn.Sequential(*list(model.children())[:], embed)
    model.to(device)

    '''
    transform = transforms.Compose(
            [
                #transforms.Resize((224, 224)), 
                #transforms.PILToTensor(),
                # add more if needed

                transforms.resize((224, 244, 3)),
                transforms.torch.from_numpy()

            ]
        )
    '''
    model.load_state_dict(torch.load("/ghome/group07/test/W4/part2/triplet_train/saved_models/model_emb2048_margin05_p2_epochs50_batch16.pth"))
    model.eval()
    

    first_cam = True

    label_counter: int = 1
    for cam in CAM_ORDER:
        if first_cam:
            pred_embds = []
            true_labels = []
        else:
            new_pred_embds = {}
            new_true_labels = {}

        frames_path = os.path.join(FRAMES_PATH_ROOT, "S03", cam, "color")

        csv_path = os.path.join(CSVS_PARKED_REMOVED_ROOT, "S03_"+cam+"_parked_removed.csv")
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

        for i in tqdm(range(len(rows))):
            frame_id = int(rows[i][0])-1
            track_id = int(rows[i][1][1:])
            x_min, y_min = float(rows[i][2]), float(rows[i][3])
            width, height = float(rows[i][4]), float(rows[i][5])

            x_max = x_min+width
            y_max = y_min+height

            img_path = os.path.join(frames_path, str(frame_id)+".png")
            full_img = cv2.imread(img_path)
            img_h, img_w, c = full_img.shape

            width_fifth = float(width/5)
            height_fifth = float(height/5)
            #width_fifth = float(width/15)
            #height_fifth = float(height/15)


            x_min_crop, y_min_crop, x_max_crop, y_max_crop = None, None, None, None
            
            if x_max+width_fifth >= img_w: x_max_crop = img_w-1
            else: x_max_crop = x_max+width_fifth-1

            if y_max+height_fifth >= img_h: y_max_crop = img_h-1
            else: y_max_crop = y_max+height_fifth-1

            if x_min-width_fifth < 0: x_min_crop = 1
            else: x_min_crop = x_min-width_fifth+1

            if y_min-height_fifth < 0: y_min_crop = 1
            else: y_min_crop = y_min-height_fifth+1


            x_min_crop, y_min_crop, x_max_crop, y_max_crop = int(x_min_crop),int(y_min_crop),int(x_max_crop),int(y_max_crop)

            crop = full_img[y_min_crop:y_max_crop, x_min_crop:x_max_crop, :]
            #img_name = "S03"+"_"+cam+"_"+str(frame_id)+"_"+str(track_id)+"_3.png"
            #img_save_path = os.path.join(CROP_SAVE_PATH, img_name)
            #cv2.imwrite(img_save_path, crop)
            #print(f"Image saved in {img_save_path}")

            # Get embedding
            anchor_img = crop[:,:,::-1]
            #print(anchor_img.shape)
            anchor_img = np.resize(anchor_img, (224, 224, 3))
            #print(anchor_img.shape)
            anchor_img = torch.from_numpy(anchor_img).to(device)
            #print(anchor_img.shape)
            anchor_img = anchor_img.permute(2, 0, 1) # from NHWC to NCHW

            with torch.no_grad():
                anchor_out = model(anchor_img.float()).cpu().numpy() # (1, 2048)
                if not first_cam:
                    if track_id not in new_true_labels:
                        new_true_labels[track_id] = [anchor_out]
                    else:
                        inner_list = new_true_labels[track_id].copy()
                        inner_list.append(anchor_out)
                        new_true_labels[track_id] = inner_list

                else:    
                    pred_embds.append(anchor_out.squeeze(0))
                    true_labels.append(track_id)



        if first_cam:

            #true_labels, unique_labels = map_labels_to_integers2(true_labels, start=label_counter)
            #label_counter += len(unique_labels)

            features_x = np.array(pred_embds)
            features_y = np.array(true_labels)

            #with open("./save_pickles/cam1_features_x.pkl", "wb") as f:
            #    pickle.dump(features_x, f)
            #with open("./save_pickles/cam1_features_y.pkl", "wb") as f:
            #    pickle.dump(features_y, f)

            # Train retrieval KNN ========================================================
            knn = KNeighborsClassifier(n_neighbors=7)
            knn.fit(features_x, features_y)

            #pred_labels = knn.predict(features_x)
            display_tsne_plot(features_x, features_y ,true_labels, try_name=TRY_NAME, title="KNN_first_cam_C013")
            #accuracy_test = accuracy_score(features_y, pred_labels)  # Test data true labels
            #utils.plot_prec_rec_curve_multiclass(features_y, pred_labels, output="./precision_recall_plot.png", n_classes=80)
            #print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
            first_cam=False
            


        else:

            # ANALYZE EACH TRACK OF THE NEW CAMERA
            # SEE IF IT BELONGS TO A TRACK WE ALREADY HAVE
            # FROM ANOTHER CAMERA OR NOT

            mapping_dict = {}
            for track_id, pred_embds_new in new_true_labels.items():
                pred_embds_np = np.array(pred_embds_new)

                dists, neighbors = [], []
                for pred_emb  in pred_embds_np:
                    dist, neighbor = knn.kneighbors(pred_emb)
                    dists.append(dist)
                    inner_n=[]
                    for n in neighbor:
                        inner_n.append(features_y[n])
                    neighbors.append(inner_n)

                neighbors_dict = {}
                for dist_list, neighbor_list in zip(dists, neighbors):
                    neighbor_list_inner = neighbor_list[0].tolist()
                    dist_list_inner = dist_list.tolist()[0]
                    for neighbor, dist in zip(neighbor_list_inner, dist_list_inner):
                        if neighbor not in neighbors_dict:
                            neighbors_dict[neighbor] = [dist]
                        else:
                            inner_list = neighbors_dict[neighbor].copy()
                            inner_list.append(dist)
                            neighbors_dict[neighbor] = inner_list

                higher_num = 0
                higher_label = None
                for neighbor, dists in neighbors_dict.items():
                    if len(dists) > higher_num:
                        higher_num = len(dists)
                        higher_label = neighbor

                median_dist = np.median(neighbors_dict[higher_label])
                
                #int_track_id = int(track_id[1:])
                if median_dist < THRESHOLD:
                    mapping_dict[track_id] = int(higher_label)
                else: 
                    mapping_dict[track_id] = track_id

            
            # load base csv
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)

            # save results in new csv
            out_csv_path = os.path.join(OUT_CSV_ROOT, TRY_NAME)
            os.makedirs(out_csv_path, exist_ok=True)
            with open(os.path.join(out_csv_path, cam+".csv"), "w") as write_file:
                writer = csv.writer(write_file)
                for i in range(len(rows)):
                    row_to_mod = rows[i]
                    mapped_label = mapping_dict[int(row_to_mod[1][1:])]
                    row_to_mod[1] = mapped_label

                    writer.writerow(row_to_mod)


            # NEW FIT OF THE KNN

            #print(type(features_x), features_x.shape, features_x)
            #print("\n\n\n")
            #print(type(new_true_labels), len(new_true_labels), new_true_labels)        
            
            for track_id, pred_emb_new in new_true_labels.items():
                for arr in pred_emb_new:
                    emb = arr.tolist()[0]
                    pred_embds.append(np.array(emb))
                    true_labels.append(mapping_dict[track_id])

            features_x = np.array(pred_embds)
            features_y = np.array(true_labels)
            knn.fit(features_x, features_y)
            display_tsne_plot(features_x, features_y ,true_labels, try_name=TRY_NAME, title="KNN_after_cam_"+cam)

    with open("./save_pickles/"+TRY_NAME+"final_features_x.pkl", "wb") as f:
        pickle.dump(features_x, f)
    with open("./save_pickles/"+TRY_NAME+"final_features_y.pkl", "wb") as f:
        pickle.dump(features_y, f)
            


            







            
