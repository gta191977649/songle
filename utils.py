def findMaxIndex(array):
    current = [0,0]

    for i,v in enumerate(array):
        if v > current[0]:
            current[0] = v
            current[1] = i
    return current[1]