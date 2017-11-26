import sklearn

def cls2str(x):
    if(x<10):
        return str(x)
    elif(x<36):
        return chr(x+55)
    else:
        return chr(x+61)


