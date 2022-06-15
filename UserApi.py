import os

class MyPerson():
    ssn = ""
    name = ""
    person_path = ""

    def __init__(self, ssn, name, person_path):
        self.ssn = ssn
        self.name = name
        self.person_path = person_path

class PersonInterface():
    ppl = {}

    def __init__(self,
                 db_parent_dir="C:\\Users\\pecko\\PycharmProjects\\MyPyFDL\\InsertPhotos",
                 unclear_dir="C:\\Users\\pecko\\PycharmProjects\\MyPyFDL\\UnclearPhotos",
                 camera_roll_dir="C:\\Users\\pecko\\Pictures\\Camera Roll"):
        self.camera_roll_dir = camera_roll_dir

        try:
            os.mkdir(db_parent_dir)
        except:
            print("Error making directory for photos to insert into database.\n{}".format(db_parent_dir))
        finally:
            self.db_parent_dir = db_parent_dir

        try:
            os.mkdir(unclear_dir)
        except:
            print("Could not create directory for unclear photo.\n{}".format(unclear_dir))
        finally:
            self.unclear_dir = unclear_dir

    def getInfoFromUser(self):
        #get info from user
        ssn = input("What is the ssn of the person you would like to add to the DB?: ")
        person_name = input("What is the name of the person you would like to add to the DB?: ")
        '''TODO: create a loop here to give an option to add many photos of the same person.'''
        photo_name = input("What is the name of the photo? (must be in camera roll folder): ")

        #get the path for the user supplied photo
        photo_camera_roll_path = os.path.join(self.camera_roll_dir, photo_name)

        #create a new directory after the person name in the temporory directory to be inserted into the DB
        photo_db_dir = os.path.join(self.db_parent_dir, person_name)
        try:
            os.mkdir(photo_db_dir)
        except:
            print("Error making directory for person to insert.")

        '''TODO: create a loop here that numbers all photos given in the loop and adds them to an array.'''
        photo_db_path = os.path.join(photo_db_dir, "0.jpg")

        '''TODO: Create a loop here to move all the pictures given in the REPL loop from the camera roll to the temp dir.'''
        #move the pictures from the camera roll path into the temporary directory
        os.rename(photo_camera_roll_path, photo_db_path)

        #append to dictionary of people
        self.ppl[person_name] = MyPerson(ssn, person_name, photo_db_dir)

    def REPL(self):
        flag = " "
        while flag:
            flag = input("Enter 1 enter a user to the DB. Enter 2 to quit")
            if int(flag) != 2:
                self.getInfoFromUser()
            else:
                break

    def getUnclearInfo(self):
        '''TODO: create a loop here to get many unclear photos of the same person.'''
        unclear_photo_roll_name = input("What is the os name of the unclear photo? (must be in camera roll folder): ")
        unclear_photo_roll_path = os.path.join(self.camera_roll_dir, unclear_photo_roll_name)

        unclear_class_dir = os.path.join(self.unclear_dir, "Unclear")
        try:
            os.mkdir(unclear_class_dir)
        except:
            print("Error making Unclear class folder inside of the unclear directory.\n{}".format(unclear_class_dir))

        '''TODO: create a loop here that numbers all unclear photos given in the loop, and moves them into the unclear class dir.'''
        unclear_photo_class_path = os.path.join(unclear_class_dir, "0.jpg")

        #move the unclear photo from the camera roll into the unclear photo class path.
        os.rename(unclear_photo_roll_path, unclear_photo_class_path)

