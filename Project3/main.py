import csp
import readFiles
import sys
import time


method = sys.argv[1]    # read from input which method we will use
fileName = sys.argv[2]  # read from input from which file we will read the problem

varDomains = readFiles.readVars(fileName)   # call readVars to read var file
# varDomains is a dictionary with variables as keys and the domain that the variable belongs to as value of the key
domainValues = readFiles.readDoms(fileName)     # call readDoms to read dom file
# domainValues is a dictionary with domains as keys and the values of each domain as values
constrains, neighbors = readFiles.readCtrs(fileName)    # call readDoms to read ctr file
# constrains is a dictionary for the constraints (see readCtrs for more info)
# neighbors is a dictionary with every variable as keys, and for each variable correspond a list with it's neighbors

values = {}
for key in varDomains.keys():
    values[key] = domainValues[varDomains[key]]     # create a new dictionary with variables as keys and a list with it's allowed values as item for this key/variable
    if key not in neighbors:   # if a variable has no neighbors, add to the dictionary the variable as a key and an empty list as a value
        neighbors[key] = []


def checkConstrains(var1, value1, var2, value2):    # function that checks the constraints for our rfls problem, this function is given as a paremeter to the CSP class
    if value1 is None or value2 is None:
        return True
    if (var1, var2) in constrains:  # check if there is a constraint involving these two variables
        k = constrains[(var1, var2)][1]  # the right part of the constraint
        symbol = constrains[(var1, var2)][0]    # the symbol of the constraint (>,<,=)
        if symbol == '=':
            return abs(value1-value2) == k      # if the constraint is ok, return true, otherwise return false.  |var1-var2|=k
        elif symbol == '>':
            return abs(value1-value2) > k   # |var1-var2|>k
        elif symbol == '<':
            return abs(value1-value2) < k       # |var1-var2|<k
    elif (var2, var1) in constrains:    # check if a constraint exists with the opposite sequence of the variables.
        # The opposite sequence of the variables is the same because we are looking for absolute difference
        k = constrains[(var2, var1)][1]
        symbol = constrains[(var2, var1)][0]
        if symbol == '=':
            return abs(value1-value2) == k
        elif symbol == '>':
            return abs(value1-value2) > k
        elif symbol == '<':
            return abs(value1-value2) < k
    return True


# initialize the problem, calling the initializer of CSP class
rlfsProblem = csp.CSP(varDomains.keys(), values, neighbors, checkConstrains, constrains)
# call the right method depenting of the input of the user
if method == "fc":
    t0 = time.time()    # count the time spent until we solve the problem
    print(csp.backtracking_search(rlfsProblem, csp.dom_wdeg, csp.lcv, csp.forward_checking))
    t1 = time.time() - t0
    print("Expanded nodes: ", rlfsProblem.nassigns)    # print the number of node expanded
    print("Checks: ", rlfsProblem.checksCount)      # print the number of checks done by the inference function
    print("Time elapsed: ", t1, "seconds")
elif method == "mac":
    t0 = time.time()
    print(csp.backtracking_search(rlfsProblem, csp.dom_wdeg, csp.lcv, csp.mac))
    t1 = time.time() - t0
    print("Expanded nodes: ", rlfsProblem.nassigns)
    print("Checks: ", rlfsProblem.checksCount)
    print("Time elapsed: ", t1, "seconds")
elif method == "fc-cbj":
    t0 = time.time()
    print(csp.cbj_search(rlfsProblem, csp.dom_wdeg, csp.lcv, csp.forward_checking))
    t1 = time.time() - t0
    print("Expanded nodes: ", rlfsProblem.nassigns)
    print("Checks: ", rlfsProblem.checksCount)
    print("Time elapsed: ", t1, "seconds")
elif method == "min-conf":
    t0 = time.time()
    print(csp.min_conflicts(rlfsProblem))
    print("Expanded nodes: ", rlfsProblem.nassigns)
    t1 = time.time() - t0
    print("Time elapsed: ", t1, "seconds")
else:
    print("Wrong method!")
