import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

from DatabaseConnection import DatabaseConnection


class FDL(DatabaseConnection):
    ppl = {}
    device = None
    tensor_dir = ""

    def __init__(self, ppl, device, database_photos_dir="C:\\Users\\pecko\\PycharmProjects\\MyPyFDL\\DatabasePhotos"):
        super().__init__()
        self.ppl = ppl
        self.device = device

        try:
            os.mkdir(database_photos_dir)
        except:
            print("Error making directory for photos from the database.\n{}".format(database_photos_dir))
        finally:
            self.database_photos_dir = database_photos_dir

    def convert_data(self, data, file_name):
        with open(file_name, 'wb') as file:
            file.write(data)

    def convertToBinaryData(self, filename):
        # Convert digital data to binary format
        with open(filename, 'rb') as file:
            binaryData = file.read()
        return binaryData

    def createDirectoriesForResults(self, database_results):
        for ssn, name, clearPhoto, tensorOfClearPhoto in database_results:
            path = os.path.join(self.database_photos_dir, name)
            try:
                os.mkdir(path)
            except:
                print("Error trying to create a directory for a person in the database.\n{}".format(path))
            self.convert_data(tensorOfClearPhoto, os.path.join(path, "0.pt"))
            self.convert_data(clearPhoto, os.path.join(path, "0.jpg"))

    def createDirectoriesFromDatabase(self):
        result = self.getAllPeople()
        self.createDirectoriesForResults(result)

    def insertAllPeopleIntoDatabase(self):
        for k, v in self.ppl.items():

            '''TODO: create a loop to go through all of the photos in the person_path and add them to the one-to-many table
            make these arrays!
            '''
            clearPhotoBinary = self.convertToBinaryData(os.path.join(v.person_path, "0.jpg"))
            clearPhotoTensorBinary = self.convertToBinaryData(os.path.join(v.person_path, "0.pt"))

            self.insertPerson(v.ssn, v.name, clearPhotoBinary, clearPhotoTensorBinary)

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

        tensorPath = ""
        for i, x in enumerate(embeddings):
            tensor_parent_path = os.path.join(photo_class_dir, str(names[i]))
            tensor_file_name = "0.pt"
            tensor_path = os.path.join(tensor_parent_path, tensor_file_name)
            torch.save(x, tensor_path)



    def printDistanceTable(self, unclear_image_class_dir):
        embeddings = []
        names = []
        x = 0
        try:
            for d in next(os.walk(self.database_photos_dir))[1]:
                '''TODO: add a loop here to get all the photos of the person from the database, not just one.'''
                tensor_file = os.path.join(d, "0.pt")
                tensor_path = os.path.join(self.database_photos_dir, tensor_file)

                '''TODO: turn this into a 3D array of shape (<number-of-people>, <number-of-photos>, <tensor>)'''
                embeddings.append(torch.load(tensor_path).tolist())
                names.append(d)
        except:
            print("Error trying to get directories from database photos directory.")

        unclear_image_path = os.path.join(os.path.join(unclear_image_class_dir, "Unclear"), "0.pt")
        names.append("UnknownFace")
        embeddings.append(torch.load(unclear_image_path).tolist())

        t = torch.tensor(embeddings)

        dists = [[(e1 - e2).norm().item() for e2 in t] for e1 in t]
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 150)
        print(pd.DataFrame(dists, columns=names, index=names))
