import mysql.connector
from mysql.connector import errorcode


class DatabaseConnection():
    connection = None

    def __init__(self, host='localhost', database='facialdatabase1', user='root', password='4201'):
        try:
            self.connection = mysql.connector.connect(host=host,
                                                 database=database,
                                                 user=user,
                                                 password=password)

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist.")
            else:
                print(err)

    def kill(self):
        if self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed.")
        else:
            print("No connection to kill.")


    '''TODO: change this to also query the one-to-many relationship table of all the photos for each person'''
    def getAllPeople(self):
        try:
            cursor = self.connection.cursor()
            sql_insert_blob_query = """ SELECT * FROM faces"""

            cursor.execute(sql_insert_blob_query)
            result = cursor.fetchall()

            self.connection.commit()
            cursor.close()

            return result
        except:
            print("Something went wrong when fetching all people from the database.")

    '''TODO: change this to add all of the pictures in the person_path into the one-to-many SQL table of person photos'''
    def insertPerson(self, ssn, name, clearPhotoBinary, clearPhotoTensorBinary):
        try:
            cursor = self.connection.cursor()
            sql_insert_blob_query = """ INSERT INTO faces
                              (ssn, name, clearPhoto, tensorOfClearPhoto) VALUES (%s,%s,%s,%s)"""

            insert_blob_tuple = (ssn, name, clearPhotoBinary, clearPhotoTensorBinary)
            result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)

            self.connection.commit()
            cursor.close()
        except:
            print("Something went wrong when inserting a person row into the database.")
