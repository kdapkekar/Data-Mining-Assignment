import string
from collections import Counter
#1
def test_print():
    print "hello world!"

#2
item_list = [2,1,2,3,4,3,2,1]
item_set = {2,1,2,3,4,3,2,1}
def list_set_length(item_list,item_set):
    print(len(item_list),len(item_set))

#3
def set_intersect():
    S = {1,4,7}
    T = {1,2,3,4,5,6}
    I = {s for s in S if s in T}
    print(I)

#4
def three_tuples():
    S = {-4,-2,1,2,5,0}
    tup1 = [(i,j,k) for i in S for j in S for k in S if i+j+k == 0]
    print(tup1)


#5a
def dict_init():
    my_dict = {'Hopper':'Grace', 'Einstein':'Albert','Turing':'Alan','Lovelace':'Ada'}

#5b

dlist = [{1:'ABC',2:'PQR',3:'XYZ'},{1:'LMN',2:'DEF',3:'IJK'},{1:'APPLE',2:'STRAWBERRY',3:'ORANGE'}]
def dict_find(dlist,k):
    found = [dict1[k] if k in dict1 else "not present" for dict1 in dlist]
    print(found)

#6
def file_line_count():
    fname = 'stories_small'
    count = 0
    with open(fname,'r') as f:
        for line in f:
            count += 1
    print(count)


#7a
def make_inverse_index(strlist):
    dictionary = dict()
    docs = [individuallist.split() for individuallist in strlist]
    docs1 = [(c for c in s if c.isalpha()) for s in docs]
    for (i, doc) in enumerate(docs1):
        for element in doc:
            if element in dictionary.keys():
                if not element.isdigit():
                    dictionary[element].update({i})
            else:
                dictionary[element] = {i}
    return dictionary

#7b
def or_search(inverseindex, query):
    doc_num = {j for i in query if i in inverseindex for j in inverseindex[i]}
    return doc_num

#7c
def and_search(inverseindex, query):
    doc_nums = []
    for key in query:
        if key not in inverseindex:
            return "Word is Not Present"
        else:
            doc_nums.append(inverseindex[key])
    return set.intersection(*doc_nums)

#7d
def most_similar(inverseindex,query):
    words = []
    temp_set = []
    for i in query:
        for j in inverseindex[i]:
            temp_set.append(j)
    res = [key for key, value in Counter(temp_set).most_common()]
    return str(res)

if __name__ == '__main__':
    test_print()
    list_set_length(item_list,item_set)
    set_intersect()
    three_tuples()
    dict_init()
    dict_find(dlist,1)
    file_line_count()

    f = open('stories_small','r')
    data = [line.lower() for line in f]

    inverse_index = make_inverse_index(data)
    print(inverse_index)

    res = or_search(inverse_index,['croatian','ornate','vnnvb'])
    print(res)
    res1 = and_search(inverse_index,['academy', 'dancer'])
    print(res1)
    res2 = most_similar(inverse_index,['brings','glass','reports'])
    print(res2)