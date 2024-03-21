import torch
import sys
import numpy as np
import json
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score
from model_class import Faster_embedding
import pickle as pkl
#from losses import ContrastiveLoss
import utils
import random
import matplotlib.pyplot as plt
from pycocotools import mask
import cv2
import tqdm
import pickle
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from visualization import display_tsne_plot
from sklearn.metrics import accuracy_score

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
    
class ImageDataset(Dataset):
    def __init__(self, train_dict, transform=None):
        arr = np.empty((len(train_dict), 4), dtype=object)
        arr[:] = train_dict

        self.anchors = arr[:, 0]
        self.positives = arr[:, 1]
        self.negatives = arr[:, 2]

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)), 
                transforms.PILToTensor(),
                # add more if needed
            ]
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return self.anchors.shape[0]

    def __getitem__(self, idx):
        anchor_img_name = self.anchors[idx]
        positive_img_name = self.positives[idx]
        negative_img_name = self.negatives[idx]

        anchor_seq_name = anchor_img_name.split("_")[0]
        positive_seq_name = positive_img_name.split("_")[0]
        negative_seq_name = negative_img_name.split("_")[0]
        anchor_path = os.path.join(PATH_PARENT_DIRECTORY, anchor_seq_name, self.anchors[idx]+".png")
        pos_path = os.path.join(PATH_PARENT_DIRECTORY, positive_seq_name, self.positives[idx]+".png")
        neg_path = os.path.join(PATH_PARENT_DIRECTORY, negative_seq_name, self.negatives[idx]+".png")

        anchor_img = self.transform(Image.open(anchor_path).convert('RGB')).to(device)
        pos_img = self.transform(Image.open(pos_path).convert('RGB')).to(device)
        neg_img = self.transform(Image.open(neg_path).convert('RGB')).to(device)
        return anchor_img, pos_img, neg_img


if __name__ == '__main__':
    
    #f = open('./configs/task_e_train_config.json')
    #config = json.load(f)
    generate_data_dicts = False
    generate_dataloader = False
    train_model = False

    PATH_PARENT_DIRECTORY = "//ghome/group07/test/W4/part2/triplet_train/saved_crops"
    #PATH_TRAINING_SET = os.path.join(PATH_PARENT_DIRECTORY, "train2014")
    PATH_VAL_SET = "/ghome/group07/test/W4/part2/triplet_train/saved_crops/S03"
    #PATH_INSTANCES_TRAIN = os.path.join(PATH_PARENT_DIRECTORY, "instances_train2014.json")
    #PATH_INSTANCES_VAL = os.path.join(PATH_PARENT_DIRECTORY, "instances_val2014.json")
    #RETRIEVAL_ANNOTATIONS = "/ghome/group07/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json"
    FT_DATASET_NAME = "coco"

    PATH_ROOT_PICKLES = "/ghome/group07/test/W4/part2/triplet_train/pickles"
    path_ids_train = os.path.join(PATH_ROOT_PICKLES, "list_train_emb.pkl")
    path_labels_train = os.path.join(PATH_ROOT_PICKLES, "dict_train_emb.pkl")
    path_ids_test = os.path.join(PATH_ROOT_PICKLES, "list_test_emb.pkl")
    path_labels_test = os.path.join(PATH_ROOT_PICKLES, "dict_test_emb.pkl")


    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1').backbone
    embed = EmbeddingLayer(embed_size=2048)
    model = torch.nn.Sequential(*list(model.children())[:], embed)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Antes
    #model = models.resnet18(weights=True)
    # model = torch.nn.Sequential(*list(model.children())[:-1], embed)

    '''
    f = open(RETRIEVAL_ANNOTATIONS)
    object_annotations = json.load(f)

    # Generate data dicts if true...
    if generate_data_dicts:
        os.makedirs('./data/custom_ids', exist_ok=True)
        os.makedirs('./data/custom_labels', exist_ok=True)
        utils.save_custom_data(object_annotations["train"], path_ids_train, path_labels_train)
        utils.save_custom_data(object_annotations["test"], path_ids_test, path_labels_test)
    '''   

    with open(path_ids_train, "rb") as f:
        ids_train = pkl.load(f)
    with open(path_ids_test, "rb") as f:
        ids_test = pkl.load(f)

    with open(path_labels_train, "rb") as f:
        labels_train = pkl.load(f)
    with open(path_labels_test, "rb") as f:
        labels_test = pkl.load(f)
    
    
    # Generate train dict (anchor,  positive,  negative,  anchor labels)
    if generate_dataloader:
        os.makedirs('./data/dataloader', exist_ok=True)
        dataloader_train = utils.create_image_loader(labels_train)
        with open('./data/dataloader/train_1_1.pkl', "wb") as f:
            pkl.dump(dataloader_train, f)
    
    with open('./data/dataloader/train_1_1.pkl', "rb") as f:
        dataloader_train = pkl.load(f)
        dataloader_train = ImageDataset(dataloader_train)
        dataloader_train = DataLoader(dataloader_train, batch_size=16, drop_last=True, shuffle=True)

    
    # Train
    if train_model:
        model.train()
        num_epochs = 50
        for epoch in range(num_epochs):
            running_loss = []
            for batch_idx, (anchor_img, pos_img, neg_img)in enumerate(tqdm.tqdm(dataloader_train)):
                # print(f'anchor: {anchor}')
                # print(f'positive: {positive}')
                # print(f'negative: {negative}')
                # print(f'anchor_labels: {anchor_labels}')
                optimizer.zero_grad()

                # Embeddings
                anchor_img = anchor_img.float()
                pos_img = pos_img.float()
                neg_img = neg_img.float()
                anchor_out = model(anchor_img)
                pos_out = model(pos_img)
                neg_out = model(neg_img)

                # Compute Triplet Loss
                loss = triplet_loss(anchor_out, pos_out, neg_out)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
            
            # Hay que hacer esto con tensor (torch().mean())
            print(f'EPOCH {epoch} Avg Triplet Loss: {torch.Tensor(running_loss).mean()}')
                
        
        SAVE_PATH = "/ghome/group07/test/W4/part2/triplet_train/saved_models/model_emb2048_margin05_p2_epochs50_batch16.pth"
        torch.save(model.state_dict(), SAVE_PATH)
    



    
    # Validate (and t-SNE plot) ===========================================================
    transform = transforms.Compose(
            [
                transforms.Resize((224, 224)), 
                transforms.PILToTensor(),
                # add more if needed
            ]
        )
    model.load_state_dict(torch.load("/ghome/group07/test/W4/part2/triplet_train/saved_models/model_emb2048_margin05_p2_epochs50_batch16.pth"))
    model.eval()
    pred_embds = []
    true_labels = []
    for img_name, labels in tqdm.tqdm(labels_test.items()):
        # Read image
        anchor_path = os.path.join(PATH_VAL_SET, img_name+'.png')
        anchor_img = transform(Image.open(anchor_path).convert('RGB')).to(device)

        # Get embedding
        with torch.no_grad():
            anchor_out = model(anchor_img.float()).cpu().numpy() # (1, 2048)
        for label in labels:
            pred_embds.append(anchor_out.squeeze(0))
            true_labels.append(label)
    
    '''
    # get the labels name
    instances_path = '/ghome/group07/mcv/datasets/C5/COCO/instances_val2014.json'
    captions = []
    with open(instances_path, 'r') as f:
        data = json.load(f)
    category_dict = {cat['id']: cat['name'] for cat in data["categories"]}
    for label in true_labels:
        #print(f'Label: {label} category_dict[{label}] = {category_dict[int(label)]}')
        captions.append(category_dict[int(label)])
    '''    
    
    features_x = np.array(pred_embds)
    # Aqui pillo la primera label pero deveriamos pillar todas...
    features_y = np.array(true_labels)
    # display_tsne_plot(features_x, features_y, captions, title="TSNE_3epochs_margin05")


    # Train retrieval KNN ========================================================
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(features_x, features_y)
    pred_labels = knn.predict(features_x)
    pred_labels = np.array(pred_labels)
    display_tsne_plot(features_x, pred_labels, true_labels, title="KNN_Predicted_Embeddings_TSNE_50epochs_margin05_batch16_sin_label")
    accuracy_test = accuracy_score(features_y, pred_labels)  # Test data true labels
    #utils.plot_prec_rec_curve_multiclass(features_y, pred_labels, output="./precision_recall_plot.png", n_classes=80)
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")

    


    
