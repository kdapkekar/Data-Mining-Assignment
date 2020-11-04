import csv
import pandas as pd
import math
import numpy as np

def entropy(length_of_D,C):
    #finding probability for each class
    entropy = 0
    P = []
    try:
        for val in C:
            p = float(val)/length_of_D
            P.append(p)
    except:
        print('Divide by Zero error')
    if len(P)>0:
        result = all(prob == P[0] for prob in P)
        if result == True:
            return 1
        else:
            try:
                for value in P:
                    entropy += (-value*(math.log(value,2)))
            except:
                print('Log zero which is infinity')
        #print('Entropy for',length_of_D,'is',entropy)
    return entropy

def getColumn():
    column= int(input("Enter Column from (0 to 9)"))
    if column<0 or column>9:
        getColumn()
    else:
        return column

def getValue(D, col):
    #value = Data[index].values[int(len(Data)/2)]
    Data = D[0]
    i = (len(Data)/2)
    value = (Data.at[Data.index[i], Data.columns[col]])
    #value = Data.get_value(int(len(Data)/2),index)
    return value

def weighted_entropy(D_Y,C_Y,D_N,C_N):
    N = D_Y + D_N
    print('---------------')
    try:
        W_E = ((float(D_Y)/N) * (entropy(D_Y,C_Y)) + (float(D_Y)/N) * (entropy(D_N,C_N)))
    except:
        print('Divide by Zero error')
    #print('Weighted entropy for',D_Y,'and',D_N,'is',W_E)
    return W_E

def IG(D, col, value):
    """Compute the Information Gain of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Information Gain for the given split
	"""
    att = D[0]
    cl = D[1]
    D_Y = 0
    D_N = 0
    Class_y1=0
    Class_N1=0
    Class_N2 = 0
    Class_y2 = 0
    for row in range(0,len(att)):
        if att.at[att.index[row],(att.columns[col])]<=value:
            D_Y += 1
            if cl[row] == 0:
                Class_y1 += 1
            else:
                Class_y2 += 1
        else:
            D_N += 1
            if cl[row] == 0:
                Class_N1 += 1
            else:
                Class_N2 += 1
    C = []
    C_Y = []
    C_Y.append(Class_y1)
    C_Y.append(Class_y2)
    C_N = []
    C_N.append(Class_N1)
    C_N.append(Class_N2)
    for c1,c2 in zip(C_Y,C_N):
        c = c1 + c2
        C.append(c)
    #print(D_Y, C_Y)
    #print(D_N,C_N)
    N = D_Y + D_N
    l = len(D[0])
    info_gain = entropy(l,C) - weighted_entropy(D_Y,C_Y,D_N,C_N)
    #print(info_gain)
    return info_gain

def Prob_gini_index_cart(length_of_D,freq_of_class_list):
    prob = 0
    P = []
    try:
        for c in freq_of_class_list:
            p = math.pow((float(c)/length_of_D),2)
            P.append(p)
            prob += p
    except:
        print('Divide by Zero error')
    return prob, P

def G(D, index, value):
    """Compute the Gini index of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Gini index for the given split
	"""
    C = []
    att = D[0]
    cl = D[1]
    D_Y = 0
    D_N = 0
    Class_y1 = 0
    Class_N1 = 0
    Class_N2 = 0
    Class_y2 = 0
    for row in range(0, len(att)):
        if att.at[att.index[row], (att.columns[index])] <= value:
            D_Y += 1
            if cl[row] == 0:
                Class_y1 += 1
            else:
                Class_y2 += 1
        else:
            D_N += 1
            if cl[row] == 0:
                Class_N1 += 1
            else:
                Class_N2 += 1
    C_Y = []
    C_Y.append(Class_y1)
    C_Y.append(Class_y2)
    C_N = []
    C_N.append(Class_N1)
    C_N.append(Class_N2)
    for c1,c2 in zip(C_Y,C_N):
        c = c1 + c2
        C.append(c)
    N = D_Y + D_N
    a ,x = Prob_gini_index_cart(N,C)
    b ,y = Prob_gini_index_cart(D_Y,C_Y)
    c ,z = Prob_gini_index_cart(D_N,C_N)
    gini_index = 1-a
    weighted_gini_index = (float(D_Y)/N)*(1-b) + (float(D_N)/N)*(1-c)
    return weighted_gini_index

def CART(D, index, value):
    """Compute the CART measure of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the CART measure for the given split
	"""
    att = D[0]
    cl = D[1]
    D_Y = 0
    D_N = 0
    Class_y1 = 0
    Class_N1 = 0
    Class_N2 = 0
    Class_y2 = 0
    for row in range(0, len(att)):
        if att.at[att.index[row], (att.columns[index])] <= value:
            D_Y += 1
            if cl[row] == 0:
                Class_y1 += 1
            else:
                Class_y2 += 1
        else:
            D_N += 1
            if cl[row] == 0:
                Class_N1 += 1
            else:
                Class_N2 += 1
    C_Y = []
    C_Y.append(Class_y1)
    C_Y.append(Class_y2)
    C_N = []
    C_N.append(Class_N1)
    C_N.append(Class_N2)
    N = D_Y + D_N
    a , b = Prob_gini_index_cart(D_Y,C_Y)
    c , d = Prob_gini_index_cart(D_N,C_N)
    diff_of_prob_of_classes = 0
    for i,j in zip(b,d):
        k = abs(i - j)
        diff_of_prob_of_classes += k
    CART = 2 * (float(D_Y)/N) * (float(D_N)/N) * abs(diff_of_prob_of_classes)
    return CART
def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

	Args:
		D: A dataset, tuple (X, y) where X is the data, y the classes
		criterion: one of "IG", "GINI", "CART"

	Returns:
		A tuple (i, value) where i is the index of the attribute to split at value
	"""
# functions are first class objects in python, so let's refer to our desired criterion by a single name
    minValue = 1
    maxValue = 0
    Value = 0
    index_of_att = 0
    att = np.array(D[0])
    criterion = criterion.lower()
    if criterion == "ig":
        # MAX RESULT
        for i, col in enumerate(att):
            for j, value in enumerate(col):
                measureValue = IG(D, j, value)
                if measureValue > maxValue:
                    maxValue = measureValue
                    Value = value
                    index_of_att = j

    elif criterion == "gini":
        # MIN GINI
        for i, col in enumerate(att):
            for j, value in enumerate(col):
                measureValue = G(D, j, value)
                if measureValue < minValue:
                    minValue = measureValue
                    Value = value
                    index_of_att = j


    else:
        # MAX CART
        for i, col in enumerate(att):
            for j, value in enumerate(col):
                measureValue = CART(D, j, value)
            #print("index", index, "measureValue", measureValue)
                if measureValue > maxValue:
                    maxValue = measureValue
                    Value = value
                    index_of_att = j


    return index_of_att, Value

def load(filename):
    """Loads filename as a dataset. Assumes the last column is classes, and
	observations are organized as rows.

	Args:
		filename: file to read

	Returns:
		A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
		where X[i] comes from the i-th row in filename; y is a list or ndarray of
		the classes of the observations, in the same order
	"""

    train_dataset = pd.read_csv(filename)
    #print(train_dataset)
    df = pd.DataFrame(train_dataset)
    #train data set with attributes only
    X = df[df.columns[0:10]]
    #Classes of train dataset
    y = df[df.columns[10]]
    y = y.values.tolist()
    #classes,number_of_classes
    classes = list(set(y))
    number_of_classes = len(classes)
    #print(number_of_classes)
    C = []
    for i in classes:
        c = y.count(i)
        C.append(c)
    print(X)
    return X,y

def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""
    indexIg, valueIg = bestSplit(train, "IG")
    print("Best Index", indexIg, "Best Split Value", valueIg)
    predicted_classes = []
    #dataset of test
    t = test[0]
    #classes of test
    t1 = test[1]
    # dataset of train
    t2 = train[0]
    # classes of train
    t3 = train[1]
    for row in range(0,len(t)):
        if t.at[t.index[row],t.columns[indexIg]] <= valueIg:
            predicted_classes.append(0)
        else:
            predicted_classes.append(1)
    print("Classified using IG", predicted_classes)
    return predicted_classes

def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""
    indexGini, valueGini = bestSplit(train, "GINI")
    print("Best Index", indexGini, "Best Split Value", valueGini)
    predicted_classes = []
    # dataset of test
    t = test[0]
    # classes of test
    t1 = test[1]
    # dataset of train
    t2 = train[0]
    # classes of train
    t3 = train[1]
    for row in range(0, len(t)):
        if t.at[t.index[row], t.columns[indexGini]] <= valueGini:
            predicted_classes.append(0)
        else:
            predicted_classes.append(1)
    print("Classified using GINI", predicted_classes)
    return predicted_classes

def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""
    indexCart, valueCart = bestSplit(train, "CART")
    print("Best Index", indexCart, "Best Split Value", valueCart)
    predicted_classes = []
    # dataset of test
    t = test[0]
    # classes of test
    t1 = test[1]
    # dataset of train
    t2 = train[0]
    # classes of train
    t3 = train[1]
    for row in range(0, len(t)):
        if t.at[t.index[row], t.columns[indexCart]] <= valueCart:
            predicted_classes.append(1)
        else:
            predicted_classes.append(0)
    print("Classified using CART", predicted_classes)
    return predicted_classes

def main():
    """This portion of the program will run when run only when main() is called.
	This is good practice in python, which doesn't have a general entry point
	unlike C, Java, etc.
	This way, when you <import HW2>, no code is run - only the functions you
	explicitly call.
	"""
    with open('train') as csv_file:
        D = load(csv_file)
    #print(D)
    global D1
    with open('test') as csv_file:
        D1 = load(csv_file)
    #print(D1)
    print(len(D[0]))

    #a
    Col = getColumn()
    print('Given Index', Col)
    value = getValue(D, Col)
    print('Given Value', value)
    Info_Gain = IG(D, Col, value)
    print('Information Gain', Info_Gain)
    Gini_index = G(D,Col,value)
    print('Gini Index',Gini_index)
    Cart = CART(D,Col,value)
    print('Classification and Regression tree value is',Cart)

    criterion = raw_input("----Enter your criterion: IG GINI or CART----")
    '''bestIndex, splitValue = bestSplit(D, criterion)
    print("-----------")
    print("Best Index", bestIndex, "Best Split Value", splitValue)'''

    #Test dataset and classify
    if criterion.lower() == 'ig':
        P1 = classifyIG(D, D1)
        print('Predicted class values using Information gain', P1)
    elif criterion.lower() == 'gini':
        P2 = classifyG(D, D1)
        print('Predicted class values using Gini Index', P2)
    elif criterion.lower() == 'cart':
        P3 = classifyCART(D, D1)
        print('Predicted class values using CART', P3)
    else:
        print('Invalid criterion')



if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	"""
    main()