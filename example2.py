from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

from torch import linalg as LA

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder('C:\\Users\\pecko\\Pictures\\Camera Roll\\test_images2')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

'''
torch.set_printoptions(threshold=10_000)
print(embeddings)
'''
x = 0
for i in embeddings:
    torch.save(i, ".\\tensors\\tensor_"+str(names[x])+str(x)+".pt")
    x+=1

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
print(pd.DataFrame(dists, columns=names, index=names))

dists = [[LA.vector_norm(e1 - e2, ord=2).item() for e2 in embeddings] for e1 in embeddings]
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
print(pd.DataFrame(dists, columns=names, index=names))

