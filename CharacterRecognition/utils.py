import sklearn

def cls2str(x):
    if(x<11):
        return str(x-1)
    elif(x<37):
        return chr(x+54)
    else:
        return chr(x+60)

def index2str(x):
    return cls2str(1+x)

