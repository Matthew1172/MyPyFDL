import math
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

from DatabaseConnection import DatabaseConnection
from common import *

class MyDataPoint():
    def __init__(self, _class, dist):
        self._class = _class
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other

    def __gt__(self, other):
        return self.dist > other

class MyPhoto():
    def __init__(self, photo, tensor=None):
        self.photo = photo
        self.tensor = tensor

class MyPerson():
    ssn = ""
    name = ""
    photos = None

    def __init__(self, ssn, name, photos=None):
        if photos is None:
            photos = []
        self.photos = photos
        self.ssn = ssn
        self.name = name

class FDL(DatabaseConnection):
    device = None
    tensor_dir = ""
    ppl = {}

    def __init__(self, device, database_photos_dir="C:\\Users\\pecko\\PycharmProjects\\MyPyFDL\\DatabasePhotos"):
        super().__init__()
        self.device = device

        try:
            os.mkdir(database_photos_dir)
        except:
            print("Error making directory for photos from the database.\n{}".format(database_photos_dir))
        finally:
            self.database_photos_dir = database_photos_dir

    def createAllTensors(self, photo_class_dir):
        workers = 0 if os.name == 'nt' else 4

        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        def collate_fn(x):
            return x[0]

        dataset = datasets.ImageFolder(photo_class_dir)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

        aligned = []
        names = []
        for x, y in loader:
            x_aligned, prob = mtcnn(x, return_prob=True)
            if x_aligned is not None:
                #print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

        aligned = torch.stack(aligned).to(self.device)
        embeddings = resnet(aligned).detach().cpu()

        seen = []
        counter = 0
        for i, x in enumerate(embeddings):
            if names[i] in seen:
                counter += 1
            else:
                counter = 0
                seen.append(names[i])

            name_ssn = names[i].split('$')
            name = name_ssn[0]
            ssn = name_ssn[1]

            tensor_parent_path = os.path.join(photo_class_dir, str(names[i]))
            tensor_path = os.path.join(tensor_parent_path, str(counter)+".pt")
            torch.save(x, tensor_path)

            photo_bin = convertToBinaryData(os.path.join(tensor_parent_path, str(counter)+".jpg"))
            tensor_bin = convertToBinaryData(tensor_path)

            p = MyPhoto(photo_bin, tensor_bin)
            if names[i] in self.ppl:
                self.ppl[names[i]].photos.append(p)
            else:
                self.ppl[names[i]] = MyPerson(ssn, name, [p])

    def insertPeople(self):
        if len(self.ppl) < 1:
            print("Nothing to insert.")
            return

        for k, v in self.ppl.items():
            if not self.checkPerson(v.ssn):
                self.insertPerson(v.ssn, v.name)
            for p in v.photos:
                self.insertPhoto(v.ssn, p)

    def createDirectoriesForResults(self, database_results):
        counter = 0
        seen = []
        for ssn, name, tensor in database_results:
            code = getPersonClass(name, ssn)
            path = os.path.join(self.database_photos_dir, code)

            if code in seen:
                counter += 1
            else:
                try:
                    os.mkdir(path)
                except:
                    print("Error trying to create a directory for a person in the database.\n{}".format(path))
                counter = 0
                seen.append(code)

            convert_data(tensor, os.path.join(path, str(counter)+".pt"))

    def createDirectoriesFromDatabase(self):
        result = self.getAllPeople()
        self.createDirectoriesForResults(result)

    '''This will only print the distances for the first image of every person.'''
    def printDistanceTable(self, unclear_image_class_dir):
        embeddings = []
        names = []
        try:
            for d in next(os.walk(self.database_photos_dir))[1]:
                tensor_file = os.path.join(d, "0.pt")
                tensor_path = os.path.join(self.database_photos_dir, tensor_file)

                embeddings.append(torch.load(tensor_path).tolist())
                names.append(d)
        except:
            print("Error trying to get directories from database photos directory.")
        unclear_image_path = os.path.join(os.path.join(unclear_image_class_dir, getPersonClass("Unclear", 0)), "0.pt")
        names.append("UnknownFace")
        embeddings.append(torch.load(unclear_image_path).tolist())
        t = torch.tensor(embeddings)
        dists = [[(e1 - e2).norm().item() for e2 in t] for e1 in t]
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 150)
        print(pd.DataFrame(dists, columns=names, index=names))

    def print_knn(self, unclear_image_class_dir):
        unclear_image_path = os.path.join(os.path.join(unclear_image_class_dir, getPersonClass("Unclear", 0)), "0.pt")
        uc = torch.load(unclear_image_path)
        points, classes = self.get_points(uc)
        knn_result = self.knn(points, classes)
        print("I think the person you're looking for is: {}".format(knn_result))

    def get_points(self, uc):
        points = []
        classes = {}
        for d in next(os.walk(self.database_photos_dir))[1]:
            path = os.path.join(self.database_photos_dir, d)
            for f in next(os.walk(path))[2]:
                tensor_path = os.path.join(path, f)
                t = torch.load(tensor_path)
                dist = (uc - t).norm().item()
                points.append(MyDataPoint(d, dist))
                classes[d] = 1
        return points, classes

    def knn(self, points, classes):
        k = self.choose_k(len(classes))
        return self.most_common(merge_sort(points)[:k])

    def most_common(self, points):
        ans = {}
        for dp in points:
            if dp._class in ans: ans[dp._class] += 1
            else: ans[dp._class] = 1
        return max(ans, key=ans.get)

    def choose_k(self, n):
        MAX = n+100
        for k in range(math.floor(math.sqrt(n)), MAX):
            if k%2 != 0 and n%k != 0:
                return k
        print("Could not find a value for K. Using 3.")
        return 3