

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


def merge_sort(a):
    if len(a) < 2: return a
    mid = len(a)//2
    return merge(
        merge_sort(a[:mid]),
        merge_sort(a[mid:])
    )


def merge(a, b):
    def merge_iter(a, b, ans):
        if len(a) < 1 and len(b) < 1: return ans
        if len(a) < 1: return ans + b
        if len(b) < 1: return ans + a
        if a[0] < b[0]:
            ans.append(a[0])
            return merge_iter(a[1:], b, ans)
        else:
            ans.append(b[0])
            return merge_iter(a, b[1:], ans)
    return merge_iter(a, b, [])