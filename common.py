def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def convert_data(data, file_name):
    with open(file_name, 'wb') as file:
        file.write(data)

def getPersonClass(name, ssn):
    return name+"$"+str(ssn)