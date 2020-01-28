def RemoveListBrace(xstr):
    if xstr is None:
        return None
    res = ""
    for l in xstr:
        if l != '[' and l != ']':
            res += l
    return res

def ConvertAsType(var, dtype):
    if var == "None":
        return None
    if type(dtype) == bool:
        if var == "True":
            return True
        else:
            return False
    elif type(dtype) == int:
        return int(var)
    elif type(dtype) == float:
        return float(var)
    else:
        return var


def Str2List(xstr, delimiter=',', dtype=int):
    if xstr is None:
        return None
    xlist = xstr[1:-1].split(',')
    for i in range(len(xlist)):
        try:
            xlist[i] = dtype(xlist[i])
        except Exception:
            print(xlist)
            raise Exception
    return xlist


def ModelAssign(kwargs, key, default=None):
    if key not in kwargs.keys():
        return default
    else:
        return ConvertAsType(kwargs[key], default)


def StripRedundancy(X):
    res = ""
    for i in range(len(X)):
        if X[i] != '\'' and X[i] != '\"' and X[i] != ' ':
            res += X[i]
    return res


def SplitExpr(expr):
    values = expr.split("=")
    values[0] = StripRedundancy(values[0])
    values[1] = StripRedundancy(values[1])
    return values[0], values[1]

def ParseBlock(fp):
    ltype = None
    param_dict = {}
    line = fp.readline()
    if line == "":
        return ltype, param_dict
    while line == "\n":
        line = fp.readline()
    assert line[0] == '[', "First expression of a block must be a layer definition. e.g. [Dense]"
    ltype = line[1:-2]
    line = fp.readline()
    while line != "\n" and line != "":
        key, value = SplitExpr(line[:-1])
        param_dict[key] = value
        line = fp.readline()
    return ltype, param_dict
