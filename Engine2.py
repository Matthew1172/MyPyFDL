import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd
import os
import mysql.connector
from mysql.connector import errorcode
from torch import linalg as LA

class MyInfo():
    unclearImageFolder = ""
    def __init__(self):
        pass

class MyUnclearPhoto():
    dataPath = ""
    unclearImagePath = []
    unclearImagePathDirectory = ""
    result = []
    directories = []
    ssn = []
    def __init__(self):
        pass

info = MyInfo()
up = MyUnclearPhoto()

def main():
    #1. ask user for a clear facial photo to insert, and name of person in photo
    #getInfoFromUser()
    #2. create all tensors from photos in InsertPhotos and put them in tensors folder
    #createAllTensorsInInsertPhotos()
    # Insert all tensors from tensors folder into the DB. Insert norm of the tensor. Insert name of person. Insert the photo.
    #insertAllPeopleIntoDatabase()
    #3. ask user for an unclear facial photo. Create a directory called Unclear with user supplied photo in it.
    getUnclearInfoFromUser()
    #4. Query DB for all entries.
    queryMyDatabaseForAllEntries()
    # Create all directories for queried results
    createDirectoriesForResults()
    # Create clear tensors in directories created in the prev step from DB for all entries. Save all tensors in ppl dict.
    createClearTensors()
    # Create clear photos from DB in directories for all entries.
    #createClearPhotos()

    #5. Perform distance operation on all entries. Print distances below 1.0
    createAllTensorsInUnclear()

    printDistanceTable()

def getUnclearInfoFromUser():
    photoName = input("What is the os name of the unclear photo? (must be in camera roll folder): ")
    directory = "UserUnclearPhotoDirectory"
    parent_dir = "C:\\Users\\pecko\\PycharmProjects\\pythonProject1\\Unclear"
    up.dataPath = parent_dir

    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    up.unclearImagePathDirectory = path

    path = os.path.join(path, directory+"2")
    os.mkdir(path)

    #C:\\Users\\pecko\\Pictures\\Camera Roll\\WIN_20210910_11_02_15_Pro.jpg
    os.rename("C:\\Users\\pecko\\Pictures\\Camera Roll\\"+photoName, path+"\\1.jpg")

    up.unclearImagePath.append(path+"\\1.jpg")


def queryMyDatabaseForAllEntries():
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='facialdatabase1',
                                             user='root',
                                             password='4201')

        cursor = connection.cursor()
        sql_insert_blob_query = """ SELECT * FROM faces"""

        cursor.execute(sql_insert_blob_query)
        up.result = cursor.fetchall()

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



def createAllTensorsInUnclear():
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

    dataset = datasets.ImageFolder(up.unclearImagePathDirectory)
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
    for i in embeddings:
        name = up.unclearImagePathDirectory+"\\"+str(x)+".pt"
        torch.save(i, name)
        x += 1


def createDirectoriesForResults():
    for ssn in [i[0] for i in up.result]:
        path = os.path.join(up.dataPath, ssn)
        os.mkdir(path)
        up.directories.append(path)
        up.ssn.append(ssn)

def createClearTensors():
    x = 0
    for tensorOfClearPhoto in [i[3] for i in up.result]:
        convert_data(tensorOfClearPhoto, up.directories[x]+"\\1.pt")
        x+=1

def convert_data(data, file_name):
    with open(file_name, 'wb') as file:
        file.write(data)

def printDistanceTable():
    embeddings = []
    names = []
    x = 0
    for d in up.directories:
        names.append(up.ssn[x])
        embeddings.append(torch.load(d+"\\1.pt").tolist())
        x+=1

    names.append("UnknownFace")


    embeddings.append(torch.load(up.unclearImagePathDirectory + "\\0.pt").tolist())

    t=torch.tensor(embeddings)

    dists = [[(e1 - e2).norm().item() for e2 in t] for e1 in t]
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    print(pd.DataFrame(dists, columns=names, index=names))

if __name__ == "__main__": main()