import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score
#from losses import ContrastiveLoss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
import pandas as pd
import umap
import matplotlib.patheffects as PathEffects


distinct_colors = [
        [r / 255, g / 255, b / 255]
        for r, g, b in [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 128],  # Purple
            [255, 165, 0]   # Orange
        ]
    ]


def display_UMAP_plot(features_x, features_y, captions, title="UMAP"):

    features_data = np.hstack([features_x, features_y[:, None]])

    N_COMPONENTS = 2
    out_tsne = umap.UMAP(n_components=N_COMPONENTS, min_dist=0.1, metric='euclidean').fit_transform(features_data)

    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=captions))
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", palette=distinct_colors, data=df, legend=True)
    plt.title(title)
    plt.savefig(f'./{title.replace(" ", "_")}.png', dpi=300)


def display_tsne_plot(features_x, features_y, captions, title="TSNE"):

    features_data = np.hstack([features_x, features_y[:, None]])
    #feature_colors = [distinct_colors[l] for l in features_y]
    palette = sns.color_palette("hsv", n_colors=len(set(captions)))

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1, metric='euclidean').fit_transform(features_data)

    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=captions))
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", palette=palette, data=df, legend='brief')
    label_positions = df.groupby('label').mean().reset_index()
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., ncol=3)
    for _, row in label_positions.iterrows():
        plt.text(row['x'], row['y'], row['label'], horizontalalignment='center', verticalalignment='center', 
                 fontsize=9, weight='bold', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'./{title.replace(" ", "_")}.png', dpi=300)


if __name__ == '__main__':
    batch_size = 16
    num_epochs = 50
    weigths_path = None

    EMB_SHAPE = 2096

    print("Loading data.")
    MIT_split_train = Dataset('/ghome/group07/mcv/datasets/C3/MIT_split/train')
    MIT_split_test = Dataset('/ghome/group07/mcv/datasets/C3/MIT_split/test')
     
    dataset = ImageFolder('/ghome/group07/mcv/datasets/C3/MIT_split/train', transform =transforms.ToTensor())
    test_dataset = ImageFolder('/ghome/group07/mcv/datasets/C3/MIT_split/test', transform =transforms.ToTensor())

    # Split training into Train - Val
    total_size = len(MIT_split_train)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size
    MIT_split_train, MIT_split_val = random_split(MIT_split_train, [train_size, val_size])

    train_dataloader = DataLoader(MIT_split_train, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(MIT_split_val, batch_size=16, shuffle=True)

    print("Looping through test dataloader...")
    test_dataloader = DataLoader(MIT_split_test, batch_size=16, shuffle=True)
    for batch_idx, (images, labels, captions) in enumerate(test_dataloader):
        print(batch_idx, images.shape, labels, captions)

    loss = losses.TripletMarginLoss()

    print("\nInitializing the Model...")
    model = ResNet_embedding(loss=loss)
    model.train_model_with_dataloader(train_dataloader, num_epochs)
    clf, train_labels = model.train_knn(train_dataloader)
    avg_precision, mapk1, mapk5 = model.test(train_labels, test_dataloader, clf)
    print(f"\n\nObtained average precision: {avg_precision}")
    #print(f"\nObtained apk1: {apk1} and apk5: {apk5}")
    print(f"\nObtained mapk1: {mapk1} and mapk5: {mapk5}")

    features_x, features_y, captions = model.extract_features_with_captions(test_dataloader)
    display_tsne_plot(features_x, features_y, captions, title="TSNE Triplet")
    display_UMAP_plot(features_x, features_y, captions, title="UMAP Triplet")

    


