'''
CREATE TABLE `facialdatabase1`.`faces` (
  `ssn` VARCHAR(9) NOT NULL,
  `name` VARCHAR(50) NOT NULL,
  `clearPhoto` LONGBLOB NOT NULL,
  `tensorOfClearPhoto` LONGBLOB NOT NULL,
  `ClearPhotoNorm` DECIMAL(16,16) NOT NULL,
  PRIMARY KEY (`ssn`));
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import mysql.connector
from mysql.connector import errorcode
from torch import linalg as LA

class MyInfo():
    imageFolder = ""
    conn = None
    def __init__(self):
        pass

class MyPerson():
    ssn = ""
    name = ""
    clearPhoto = ""
    tensorOfClearPhoto = ""
    clearPhotoNorm = ""
    def __init__(self, ssn, name, clearPhotoPath):
        self.ssn = ssn
        self.name = name
        self.clearPhoto = clearPhotoPath

info = MyInfo()
ppl = {}

def main():
    #1. ask user for a clear facial photo to insert, and name of person in photo
    flag = " "
    while flag:
        flag = input("Enter 1 enter a user to the DB. Enter 2 to quit")
        if int(flag) != 2:
            getInfoFromUser()
        else:
            break

    #2. create all tensors from photos in InsertPhotos and put them in tensors folder
    createAllTensorsInInsertPhotos()
    # Insert all tensors from tensors folder into the DB. Insert norm of the tensor. Insert name of person. Insert the photo.
    insertAllPeopleIntoDatabase()
    #3. ask user for an unclear facial photo to search
    #4. create tensor from this photo. Query DB for entries
    #5. Query DB for all entries that have norm distance < 1.0 and return everything on them

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def insertRow(ssn, name, clearPhoto, tensorOfClearPhoto, clearPhotoNorm):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='facialdatabase1',
                                             user='root',
                                             password='4201')

        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO faces
                          (ssn, name, clearPhoto, tensorOfClearPhoto) VALUES (%s,%s,%s,%s)"""

        clearPhotoBinary = convertToBinaryData(clearPhoto)
        tensorOfClearPhotoBinary = convertToBinaryData(tensorOfClearPhoto)

        insert_blob_tuple = (ssn, name, clearPhotoBinary, tensorOfClearPhotoBinary)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def insertAllPeopleIntoDatabase():
    for k, v in ppl.items():
        insertRow(v.ssn, v.name, v.clearPhoto, v.tensorOfClearPhoto, v.clearPhotoNorm)

def createAllTensorsInInsertPhotos():
    workers = 0 if os.name == 'nt' else 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(info.imageFolder)
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

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    x = 0
    tensorPath = ""
    for i in embeddings:
        tensorPath = "C:\\Users\\pecko\\PycharmProjects\\pythonProject1\\tensors2\\tensor_" + str(names[x]) + str(x) + ".pt"
        torch.save(i, tensorPath)
        ppl[names[x]].tensorOfClearPhoto = tensorPath
        ppl[names[x]].clearPhotoNorm = LA.vector_norm(i, ord=2).item()
        x += 1


def getInfoFromUser():
    ssn = input("What is the ssn of the person you would like to add to the DB?: ")
    personName = input("What is the name of the person you would like to add to the DB?: ")
    photoName = input("What is the name of the photo? (must be in camera roll folder): ")

    directory = personName

    parent_dir = "C:\\Users\\pecko\\PycharmProjects\\pythonProject1\\InsertPhotos"
    info.imageFolder = parent_dir

    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    #C:\\Users\\pecko\\Pictures\\Camera Roll\\WIN_20210910_11_02_15_Pro.jpg
    os.rename("C:\\Users\\pecko\\Pictures\\Camera Roll\\"+photoName, path+'\\1.jpg')

    ppl[personName] = MyPerson(ssn, personName, path+'\\1.jpg')

if __name__ == "__main__": main()