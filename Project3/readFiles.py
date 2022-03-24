
def readVars(fileName):
    name = "./rlfap/var"
    name = name + fileName
    name = name + ".txt"
    fp = open(name, "r")    # open the file
    varCount = int(fp.readline())   # read the first line that contains the number of the variables
    dict = {}
    for i in range(0, varCount):    # read every variable
        tempLine = fp.readline()    # read the line
        splitted = tempLine.split(" ")  # split the line that constains two numbers
        if(len(splitted) != 2):
            print("WRONG FILE INPUT")
            return None
        dict[int(splitted[0])] = int(splitted[1])   # add the first number (variable) as key, and the second number (domain) as value
    fp.close()  # after reading all the variables, close the file
    return dict     # return the dictionary


def readDoms(fileName):
    name = "./rlfap/dom"
    name = name + fileName
    name = name + ".txt"
    fp = open(name, "r")    # open the file
    domCount = int(fp.readline())   # read the first line that contains the number of domains
    dict = {}
    for i in range(0, domCount):    # for every domain
        tempLine = fp.readline()    # read the line
        splitted = tempLine.split(" ")  # split the line (a list with every number of the line is returned in splitted variable)
        domain = int(splitted[0])   # the first number of the list is the domain
        values = [int(i) for i in splitted[2:]]  # add every other number of the list to the values of the domain
        # the second number of every line is how many values the domain contains, so we start from index 2 instead of index 1
        dict[domain] = values   # add the values corresponding to the domain
    fp.close()  # after reading all domains, close the file
    return dict     # return the dictionary with the domains and their values.


def readCtrs(fileName):
    name = "./rlfap/ctr"
    name = name + fileName
    name = name + ".txt"
    fp = open(name, "r")    # open the file
    ctrCount = int(fp.readline())   # read the first lien of the file that contains the number of the constraints
    ctrDict = {}
    neighbors = {}
    for i in range(0, ctrCount):    # for every constraint
        tempLine = fp.readline()    # read the line
        splitted = tempLine.split(" ")  # split the line
        if(len(splitted) != 4):
            print("WRONG FILE INPUT")
            return None
        # as keys we are inserting tuples with the two variables that are involved in the constaint. We are adding the constraint in both ways (var1,var2) and (var2,var1)
        # as value we are inserting a tuple with the symbol of the constraint and tha value for example (>,255)
        ctrDict[(int(splitted[0]), int(splitted[1]))] = (splitted[2], int(splitted[3]))
        ctrDict[(int(splitted[1]), int(splitted[0]))] = (splitted[2], int(splitted[3]))
        # we are also creating the neighbors for every variable
        if int(splitted[0]) not in neighbors:   # if the variable1 has no other neighbors
            neighbors[int(splitted[0])] = []    # initialize neighbors with an empty list
        neighbors[int(splitted[0])].append(int(splitted[1]))    # insert var2 to the neighbors of var1, as there is a constaint involving these two variables
        if int(splitted[1]) not in neighbors:   # if the variable2 has no other neighbors
            neighbors[int(splitted[1])] = []    # initialize neighbors with an empty list
        neighbors[int(splitted[1])].append(int(splitted[0]))    # insert var1 to the neighbors of var2, as there is a constaint involving these two variables
    fp.close()  # after reading all the constraints, close the file
    # finaly we have created two dictionaries
    # ctrDict contains all the constrainst (with keys the two variables that each constraint involves)
    # neighbors has every variable as keys, and for each key/variable a list with it's neighbors as value
    return ctrDict, neighbors
