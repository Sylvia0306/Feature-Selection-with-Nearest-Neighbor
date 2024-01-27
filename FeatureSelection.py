### Project 2 ###
from fileinput import filename
import os
import numpy as np
import time


def readFile( filename ) :
    # read the file and return it as list
    # note that the data is the float-point numbers
    data = []
    itemList = []
    f = open( filename, 'r' )
    for line in f :
        for item in line.split() :
            itemList.append( item )
        data.append( itemList )
        itemList = []
    f.close()
    return data


# calculate the accuracy
def accuracy_calculate( dataset, predict_class ):
    accurate = 0

    for i in range(len(dataset)):
        if float(dataset[i][0]) == predict_class[i]:
            accurate += 1
    
    accuracy = accurate/len(dataset)
    return accuracy


def get_test_dataset(dataset, current_set, feature_to_add):
    test_dataset = []
    temp = []
    testset = []

    # initialize a list test_dataset with all value equals to 0
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            temp.append(0)
        test_dataset.append(temp)
        temp = []

    testset = current_set.copy()
    testset.append(0)

    if feature_to_add != 0:
        # testset is a list that stores whitch features will use
        testset.append(feature_to_add)

    # add all the data of the feature we want to test into a list called test_dataset
    for j in range(len(testset)):
        testset_index = testset[j]
        for index in range(len(dataset)):
            test_dataset[index][testset_index] = dataset[index][testset_index]
    
    return test_dataset


def leave_one_out_cross_validation(dataset, current_set, feature_to_add):
    predict_class = []
    minimum_index = 0
    index = 0
    minimum = 100
    distance=0
    instances = len(dataset)
    features = len(dataset[0])-1

    test_dataset = get_test_dataset(dataset, current_set,feature_to_add)

    k=10

    # k_fold
    # assume k=10, len(test_dataset)=1000, k_fold_len=100
    # fold1:0-99, fold2:100-199, fold3:200-299, fold4:300-399...fold10:900-999
    k_fold_len = len(test_dataset)/k
    k_fold_start_index = 0

    for k in range(k):
        k_fold_end_index = k_fold_start_index+k_fold_len
       
        k_fold_testset = test_dataset[int(k_fold_start_index):int(k_fold_end_index)]

        temp = test_dataset.copy()
        del temp[int(k_fold_start_index):int(k_fold_end_index+1)]
        k_fold_trainset = temp

        k_fold_start_index += k_fold_len

        for instance in k_fold_testset:
            minimum = 100

            # calaulate the distance between instance and ith k_fold_trainset
            for i in range(len(k_fold_trainset)):
                distance = 0
                index = i

                # calculate distance of features
                for j in range(features):
                    distance += pow((float(instance[j+1])-float(k_fold_trainset[i][j+1])), 2)
                Euclidean_distance =  distance ** 0.5

                if Euclidean_distance < minimum:
                    minimum = Euclidean_distance
                    minimum_index = index

            # store the predict result in a list
            predict_class.append(float(k_fold_trainset[minimum_index][0]))
    
    accuracy = accuracy_calculate(dataset, predict_class)

    return accuracy


def get_string(list):
    temp_string = ""

    for i in range(len(list)):
        temp_string += str(list[i])
        temp_string += " "

    return temp_string


def forward_selection(dataset):
    features = len(dataset[0])-1
    current_set = []
    feature_to_add = 0
    max_accuracy = 0
    max_accuracy_index = 0
    last_max_accuracy = 0
    max_level_maximum_accuracy = 0
    accuracy_features = ""
    maximum_accuracy_features = ""
    count = 2

    # there are features number of level in the search tree
    for index in range(features):

        max_accuracy = 0
        # find the feature with the maximum accuracy
        for i in range(features):
            feature_to_add = i+1

            if feature_to_add not in current_set:

                accuracy = leave_one_out_cross_validation(dataset, current_set, feature_to_add)
                print("feature {current_set}{feature_to_add} accuracy is {accuracy}".format(current_set = get_string(current_set), feature_to_add = feature_to_add, accuracy = accuracy))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_accuracy_index = feature_to_add
                    accuracy_features = get_string(current_set)+str(feature_to_add)

        if max_accuracy > last_max_accuracy:
            maximum_accuracy_features = accuracy_features
            max_level_maximum_accuracy = max_accuracy

        if max_accuracy < last_max_accuracy and count >= 0:
            print("accuracy has decrease!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if count <= 0:
                print("accuracy has decrease over 3 times")
                break
            count -= 1
     
        print("using feature {current_set}{max_accuracy_index} has the maximum accuracy".format(current_set = get_string(current_set), max_accuracy_index = max_accuracy_index))
        current_set.append(max_accuracy_index)
        last_max_accuracy = max_accuracy
        print(" ")
        print("current set: {current_set}".format(current_set = current_set))
        

    print("finished search! the best feature was {features} with accuracy {accuracy}".format(features = maximum_accuracy_features, accuracy = max_level_maximum_accuracy))

def backward_elimination(dataset):
    features = len(dataset[0])-1
    current_set = []
    feature_to_add = 0
    max_accuracy = 0
    max_accuracy_index = 0
    last_max_accuracy = 0
    max_level_maximum_accuracy = 0
    accuracy_features = ""
    maximum_accuracy_features = ""
    count = 2

    max_set = []

    # initialize current_set by add all features into it
    for f in range(features):
        temp = f+1
        current_set.append(temp)

    # there are features number of level in the search tree
    for index in range(features):

        max_accuracy = 0
        # find the maximum accuracy by delete a feature
        for i in range(features):

            feature_to_del = i+1

            if feature_to_del in current_set:
                temp_set = current_set.copy()

                temp_set.remove(feature_to_del)

                accuracy = leave_one_out_cross_validation(dataset, temp_set, feature_to_add)
                print("feature {current_set}accuracy is {accuracy}".format(current_set = get_string(temp_set), accuracy = accuracy))

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_set = temp_set.copy()
                    accuracy_features = get_string(temp_set)

        if max_accuracy > last_max_accuracy:
            maximum_accuracy_features = accuracy_features
            max_level_maximum_accuracy = max_accuracy

        if max_accuracy < last_max_accuracy and count >= 0:
            print("accuracy has decrease!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if count <= 0:
                print("accuracy has decrease over 3 times")
                break
            count -= 1

        print("using feature {current_set}has the maximum accuracy".format(current_set = get_string(max_set)))
        current_set = max_set
        last_max_accuracy = max_accuracy
        print(" ")
        print("current set: {current_set}".format(current_set = current_set))

    print("finished search! the best feature was {features} with accuracy {accuracy}".format(features = maximum_accuracy_features, accuracy = max_level_maximum_accuracy))


filename = "real_world_data.txt"
dataset = readFile( filename )

print("\nWelcome to Yin-Yu and Tsung-Wei's feature selection algorithm.\n")
print("Type the number of algorithm you want to run.\n 1) Forward Selection\n 2) Backward Elimination\n")

option = int(input())

start = time.time()

if option == 1:
    forward_selection(dataset)

elif option == 2:
    backward_elimination(dataset)

end = time.time()

exeTime = round((end-start),4)
print("execution time was {t} second".format(t = exeTime))

