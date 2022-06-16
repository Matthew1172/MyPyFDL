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
            self.connection.autocommit = False

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
            sql = """
            SELECT person_info.ssn, person_info.name, person_photos.tensor 
            FROM person_info 
            JOIN person_photos 
            ON person_info.ssn = person_photos.ssn"""

            cursor.execute(sql)
            result = cursor.fetchall()

            self.connection.commit()
            cursor.close()

            return result
        except:
            print("Something went wrong while fetching all people from the database.")

    def checkPerson(self, ssn):
        try:
            cursor = self.connection.cursor()
            sql = """ SELECT * FROM person_info WHERE ssn = %s"""

            sql_search = (int(ssn), )
            cursor.execute(sql, sql_search)
            result = cursor.fetchall()

            self.connection.commit()
            cursor.close()

            if len(result) > 0: return True
            return False
        except:
            print("Something went wrong while fetching a person from the database.\nSSN: {}.".format(ssn))

    def insertPerson(self, ssn, name):
        try:
            cursor = self.connection.cursor()
            sql_insert_blob_query = """ INSERT INTO person_info
                              (ssn, name) VALUES (%s,%s)"""
            insert_blob_tuple = (ssn, name)
            result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)

            self.connection.commit()
            cursor.close()
        except:
            print("Something went wrong when inserting a person's info into the database.\nSSN: {}\nNAME: {}".format(ssn, name))

    def insertPhoto(self, ssn, photo):
        try:
            cursor = self.connection.cursor()

            sql_insert_blob_query = """ INSERT INTO person_photos
                              (ssn, photo, tensor) VALUES (%s,%s,%s)"""
            insert_blob_tuple = (ssn, photo.photo, photo.tensor)
            result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)

            self.connection.commit()
            cursor.close()
        except:
            print("Something went wrong when inserting a person's photo into the database.\nSSN: {}".format(ssn))