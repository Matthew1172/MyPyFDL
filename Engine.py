'''
CREATE TABLE `facialdatabase1`.`person_info` (
  `ssn` VARCHAR(9) NOT NULL,
  `name` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`ssn`));

CREATE TABLE `facialdatabase1`.`person_photos` (
  `ssn` VARCHAR(9) NOT NULL,
  `photo` LONGBLOB NULL,
  `tensor` LONGBLOB NULL,
  CONSTRAINT `ssn`
    FOREIGN KEY (`ssn`)
    REFERENCES `facialdatabase1`.`person_info` (`ssn`)
    ON DELETE CASCADE
    ON UPDATE NO ACTION);
'''

import UserApi
import MyPyFDL
from UserApi import *
import torch

import shutil

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
    FDL = MyPyFDL.FDL(device, database_photos_dir=database_photos_dir)

    if len(next(os.walk(db_parent_dir))[1]) > 0:
        FDL.createAllTensors(db_parent_dir)
        FDL.insertPeople()

    #3. ask user for an unclear facial photo
    ua.getUnclearInfo()

    #4. create tensor from this photo
    FDL.createAllTensors(unclear_dir)

    #5. Query DB for all entries
    FDL.createDirectoriesFromDatabase()

    #6. Print the distance table
    FDL.printDistanceTable(unclear_dir)

    FDL.print_knn(unclear_dir)

    #remove temp directory
    remDir(db_parent_dir)
    remDir(unclear_dir)
    remDir(database_photos_dir)


def remDir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__": main()