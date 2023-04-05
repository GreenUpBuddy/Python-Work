# Homework 2
# name: JD Sawyer
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import numpy as np
import operator
from collections import Counter
import pandas as pd

class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)

# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree

def calcEntropy(labels, target):
    entropy = 0
    counter = Counter(labels.loc[:,target])
    for num in counter:
        entropy += -1.0 * (counter[num] / len(labels)) * math.log(counter[num] / len(labels), 2)
    return entropy

def extract(examples, attr, val):
    newExample = examples.loc[examples[attr] == val]
    del newExample[attr]
    return newExample

def id3(examples, target, attributes):

    examplesList = examples.loc[:,target].tolist()

    if examplesList.count(examplesList[0]) == len(examplesList):
        return DecisionNode(examplesList[0])
    elif len(attributes) == 0:  
        exampleCount = {}
        for ex in examplesList:
            if ex not in exampleCount.keys(): 
                exampleCount[ex] = 1
            exampleCount[ex] += 1
        sortedVal = sorted(exampleCount.items(), key= operator.itemgetter(1) ,reverse= True)
        return DecisionNode(sortedVal[0][0])
    else:      
        finalGain = 0.0
        attrIndex = 0
        
        for nums in range(len(attributes)):
            entropy = calcEntropy(examples, target)
            counter = Counter(examples.loc[:, attributes[nums]])
            newEntropy = 0.0

            for key in counter:
                newEntropy += (counter[key] / sum(counter.values())) * calcEntropy(extract(examples, attributes[nums], key), target)  
            
            newGain = (entropy - newEntropy)
            if newGain > finalGain:
                finalGain = newGain
                attrIndex = nums

        topAt = attributes[attrIndex]
        tree = DecisionNode(topAt)
        counterNew = Counter(examples.loc[:, topAt])
        values = []

        for key in counterNew:
            if key not in values:
                values.append(key)
        
        for value in values:
            newExample = extract(examples, topAt, value)
            newAt = attributes[:]
            newAt.remove(topAt)
            child = id3(newExample, target, newAt)
            tree.children[value] = child

    #tree = funTree()
    return tree

####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))