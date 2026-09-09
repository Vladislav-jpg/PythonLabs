myDict = {"first one": "we cam do it"}

def biggestDict(**kwargs):
    myDict.update(kwargs)
    return myDict

print(biggestDict(key1="fsja", key2="fjash"))