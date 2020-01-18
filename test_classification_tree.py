## Run the tests by running
## pytest -v test_classification_tree.py
## All test functions must start with test_.
##echo $PATH
##export PATH="/Users/yuehanxiao/anaconda3/bin:$PATH"

import pytest
import classification_tree
import sql_tree
import numpy as np
import psycopg2
import secret
conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

cur = conn.cursor()

def test():
    # if responses are all 0 or all 1.
    #Making sure the tree ends there.
    test_data = np.array([[25,36,78,0], [10, 20, 15, 0], [99, 50, 4, 0]])
    def Gini_index(p):
        return p*(1-p)
    node_test = classification_tree.Node(test_data, Gini_index, [])
    assert node_test.right is None
    assert node_test.left is None

def test2():
    # Split feature is the same value.
    # making sure don't choose the split variable, which only has one value for all rows
    test_data = np.array([[5, 2, 34, 0], [38, 20, 21, 0], [72, 20, 10, 0], [52, 20, 15, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = classification_tree.Node(test_data, Gini_index, [])
    #According to the acknowledged node_test's split variables, it has two left split variables, none right split variable
    assert node_test.split_variable != 'b'
    assert node_test.left.split_variable != 'b'
    assert node_test.left.left.split_variable == None
    assert node_test.right.split_variable == None

def test3():
    # missing values
    # If there is missing values inside the dataset, throw an error.
    test_data = np.array([[2, None, 0], [3, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 1]])
    def Gini_index(p):
        return p*(1-p)
    with pytest.raises(ValueError):
        classification_tree.Node(test_data, Gini_index, [])

def test4():
    # impurity function, boundray
    #making sure the p-value is between 0 and 1
    def Cross_entropy(p):
        if p == 0 or p ==1:
            return 0
        else:
            return -p * np.log(p) - (1-p) * np.log(1-p)
    test_data = np.array([[5, 2, 34, 0], [38, 20, 21, 0], [72, 20, 10, 0], [52, 20, 15, 1]])
    node_test = classification_tree.Node(test_data, Cross_entropy, [])
    assert classification_tree.Node.impurity(node_test, 1, Cross_entropy) == 0
    assert classification_tree.Node.impurity(node_test, 0, Cross_entropy) == 0

def test5_R1():
    #making sure the split point is in the first column and be the right point.
    test_data = np.array([[5, 0], [38, 0], [72, 0], [52, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = classification_tree.Node(test_data, Gini_index, [])
    assert node_test.split_variable == 0
    assert node_test.split_point == 72
    assert node_test.left.split_variable == 0
    assert node_test.left.split_point == 52
    assert node_test.right.split_variable == None
    assert node_test.right.split_point == None

def test6_R2():
    test_data = np.array([[5, 34, 1], [38, 7, 1], [72, 15, 0], [52, 29, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = classification_tree.Node(test_data, Gini_index, [])
    assert node_test.split_variable == 0
    assert node_test.split_point == 72
    assert node_test.left.split_variable == None
    assert node_test.right.split_variable == None

def test7_R3():
    test_data = np.array([[16, 24, 34, 1], [38, 20, 21, 0], [14, 20, 10, 1], [52, 20, 15, 1], [32, 59, 1, 0], [34, 21, 69, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = classification_tree.Node(test_data, Gini_index, [])
    assert node_test.split_variable == 1
    assert node_test.split_point == 59
    assert node_test.left.split_variable == 0
    assert node_test.left.split_point == 34
    assert node_test.left.left.split_variable == None
    assert node_test.left.left.split_point == None
    assert node_test.left.right.split_variable == 0
    assert node_test.left.right.split_point == 38
    assert node_test.left.right.right.split_variable == None
    assert node_test.left.right.left.split_variable == None
    assert node_test.right.split_variable == None

def test8_reponse():
    test_data = np.array([[2, 1, 3], [3, 1, 4], [2, 1, 5], [2, 1, 6], [2, 1, 17]])
    test_data1 = np.array([[2, 1, 0], [3, 1, 4], [2, 1, 0], [2, 1, 0], [2, 1, 0]])
    test_data2 = np.array([[2, 1, 1], [3, 1, 4], [2, 1, 0], [2, 1, 0], [2, 1, 0]])
    test_data3 = np.array([[2, 1, 3], [3, 1, 3], [2, 1, 3], [2, 1, 3], [2, 1, 3]])
    def Gini_index(p):
        return p*(1-p)
    with pytest.raises(ValueError):
        classification_tree.Node(test_data, Gini_index, [])
    with pytest.raises(ValueError):
        classification_tree.Node(test_data1, Gini_index, [])
    with pytest.raises(ValueError):
        classification_tree.Node(test_data2, Gini_index, [])
    with pytest.raises(ValueError):
        classification_tree.Node(test_data3, Gini_index, [])

def test9():
    # making sure dataframe and SQL data base produce the same tree
    def Gini_index(p):
        return p*(1-p)
    test_data2 = np.array([[5, 2, 34, 0], [38, 20, 21, 0], [72, 20, 10, 0], [52, 20, 15, 1]])
    node_test = classification_tree.Node(test_data2, Gini_index, [])
    node_test2 = sql_tree.SQLTree(cur, 'test_data2', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])
    assert node_test.past_split == node_test2.past_split
