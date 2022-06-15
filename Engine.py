'''TODO: create a new SQL table that saves all the photos and tensors of a given person. SSN is unique key'''

'''
CREATE TABLE `facialdatabase1`.`faces` (
  `ssn` VARCHAR(9) NOT NULL,
  `name` VARCHAR(50) NOT NULL,
  `clearPhoto` LONGBLOB NOT NULL,
  `tensorOfClearPhoto` LONGBLOB NOT NULL,
  `ClearPhotoNorm` DECIMAL(16,16) NOT NULL,
  PRIMARY KEY (`ssn`));
'''

import UserApi
import MyPyFDL
from UserApi import *
import torch

db_parent_dir = os.path.join(os.getcwd(), "InsertPhotos")
unclear_dir = os.path.join(os.getcwd(), "UnclearPhotos")
database_photos_dir = os.path.join(os.getcwd(), "DatabasePhotos")

def main():
    ua = UserApi.PersonInterface(db_parent_dir=db_parent_dir, unclear_dir=unclear_dir)

    # 1. ask user for a clear facial photo to insert, and name of person in photo
    ua.REPL()

    #2. create all tensors from photos in InsertPhotos and put them in tensors folder
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    FDL = MyPyFDL.FDL(ua.ppl, device, database_photos_dir=database_photos_dir)
    FDL.createAllTensors(db_parent_dir)
    FDL.insertAllPeopleIntoDatabase()

    #3. ask user for an unclear facial photo
    ua.getUnclearInfo()

    #4. create tensor from this photo. Query DB for entries
    FDL.createAllTensors(unclear_dir)

    #5. Query DB for all entries
    FDL.createDirectoriesFromDatabase()

    #6. Print the distance table
    FDL.printDistanceTable(unclear_dir)

if __name__ == "__main__": main()