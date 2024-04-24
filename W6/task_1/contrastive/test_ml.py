from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os

import torch.nn as nn
import torch
from torchvision.transforms import v2


from models.feature_extractor_layers import ImageFeatureExtractor3D
import pipes


from datasets import TSNHMDB51Dataset

from sklearn.metrics import accuracy_score

import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"


def create_3d_resnet_linear():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    model.blocks[-1].proj = nn.Identity()
    
    for params in list(model.parameters()):
        params.requires_grad = False
            
    new_model = nn.Sequential(
    model,
    nn.Linear(2048, 512, bias=True),
    )        
            
    return new_model

def create_3d_resnet():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    model.blocks[-1].proj = nn.Identity()
    
    for params in list(model.parameters())[:-1]:
        params.requires_grad = False
                
    return model

def create_x3d_xs():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True)
    model.blocks[5].proj = nn.Identity()

            
    return model



if __name__ == "__main__":
    

    SAVE_FOLDER = "./weights"
    transforms = {}
    
    transforms["resnet"] = v2.Compose([
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transforms["x3d_xs"] = v2.Compose([
            v2.CenterCrop(182),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    pretrained_weights = os.listdir(SAVE_FOLDER)
        # Create datasets
        
    datasets_resnet = pipes.create_datasets(dataset=TSNHMDB51Dataset,
        frames_dir="../../frames",
        annotations_dir="../../data/hmdb51/testTrainMulti_601030_splits",
        clip_length=8,
        crop_size=224,
        temporal_stride=12,
        n_segments = 2)
    
    
    
    datasets_x3d = pipes.create_datasets(dataset=TSNHMDB51Dataset,
        frames_dir="../../frames",
        annotations_dir="../../data/hmdb51/testTrainMulti_601030_splits",
        clip_length=6,
        crop_size=182,
        temporal_stride=12,
        n_segments = 4)
    
    data = {"resnet": datasets_resnet, "x3d_xs":datasets_x3d}
    

    for fn, step in [(create_3d_resnet, "resnet"), (create_x3d_xs, "x3d_xs")]:
        print("Evaluating on:, ", step)
        trans = transforms[step]
        train_set = data[step]["training"]
        test_set = data[step]["testing"]
        ### training directly on inference model
        model = fn()
        
        model = model.to(device)
        train_embeddings, train_labels = pipes.get_all_embeddings(train_set, model, trans)
        test_embeddings, test_labels = pipes.get_all_embeddings(test_set, model, trans)
        
        train_embeddings = train_embeddings.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        
        test_embeddings = test_embeddings.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        
        clf = SVC(random_state=0).fit(train_embeddings, train_labels)
        
        print("EVALUATING THE TRAINING PART ")
        y_pred = clf.predict(train_embeddings)
        
        # Calcular el accuracy global
        accuracy = accuracy_score(train_labels, y_pred)
        print("TRAINING ACCURACY:", accuracy)
        
        print("EVALUATING THE TESTING PART ")
        y_pred = clf.predict(test_embeddings)
        
        # Calcular el accuracy global
        accuracy = accuracy_score(test_labels, y_pred)
        print("TESTING ACCURACY:", accuracy)
        
        
        print("EVALUATING ACC PER CLASS ")
        print(pipes.accuracies_por_clases(clf, test_embeddings, test_labels))
        
    exit()

    for i in tqdm.tqdm(range(len(pretrained_weights)), desc="Evaluating process"):
        w = pretrained_weights[i]
        print("Evaluating on model: ", w)
        weights_path = os.path.join(SAVE_FOLDER, w)
        if "resnet" in w:
            trans = transforms["resnet"]
            train_set = data["resnet"]["training"]
            test_set = data["resnet"]["testing"]
            try:
                model = ImageFeatureExtractor3D(name_model="resnet")

                
                
                checkpoint = torch.load(weights_path, map_location=device) 
                checkpoint_keys = checkpoint.keys()
                state_dict_keys = model.state_dict().keys()
                

                model.load_state_dict(checkpoint)

                        
            except:
                model = ImageFeatureExtractor3D(name_model="resnet")

                checkpoint = torch.load(weights_path, map_location=device) 
                checkpoint_keys = checkpoint.keys()
                state_dict_keys = model.state_dict().keys()
                model.load_state_dict(checkpoint)
                                
                                
        elif "x3d_xs" in w:
            trans = transforms["x3d_xs"]
            train_set = data["x3d_xs"]["training"]
            test_set = data["x3d_xs"]["testing"]
            try:
                model = ImageFeatureExtractor3D(name_model="x3d_xs")
                checkpoint = torch.load(weights_path, map_location=device) 

                model.load_state_dict(checkpoint)                
            except:
                model = ImageFeatureExtractor3D(name_model="x3d_xs")
                checkpoint = torch.load(weights_path, map_location=device) 

                model.load_state_dict(checkpoint)
                
        ### training directly on inference model
        model = create_3d_resnet()
        
        model = model.to(device)   
        train_embeddings, train_labels = pipes.get_all_embeddings(train_set, model, trans)
        test_embeddings, test_labels = pipes.get_all_embeddings(test_set, model, trans)
        
        train_embeddings = train_embeddings.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        
        test_embeddings = test_embeddings.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        
        clf = LogisticRegression(random_state=0).fit(train_embeddings, train_labels)
        
        print("EVALUATING THE TRAINING PART ")
        y_pred = clf.predict(train_embeddings)
        
        # Calcular el accuracy global
        accuracy = accuracy_score(train_labels, y_pred)
        print("TRAINING ACCURACY:", accuracy)
        
        print("EVALUATING THE TESTING PART ")
        y_pred = clf.predict(test_embeddings)
        
        # Calcular el accuracy global
        accuracy = accuracy_score(test_labels, y_pred)
        print("TESTING ACCURACY:", accuracy)
        
        
        print("EVALUATING ACC PER CLASS ")
        print(pipes.accuracies_por_clases(clf, test_embeddings, test_labels))

        
        



            