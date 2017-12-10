def cls2str(x):
    if (x < 11):
        return str(x - 1)
    elif (x < 37):
        return chr(x + 54)
    else:
        return chr(x + 60)


def index2str(x):
    return cls2str(1 + x)


def test():
    assert cls2str(1) == '0'
    assert cls2str(5) == '4'
    assert cls2str(10) == '9'
    assert cls2str(11) == 'A'
    assert cls2str(32) == 'V'
    assert cls2str(36) == 'Z'
    assert cls2str(43) == 'g'
    assert cls2str(54) == 'r'
    assert cls2str(62) == 'z'


def test2():
    assert index2str(0) == '0'
    assert index2str(61) == 'z'
    assert index2str(35) == 'Z'
