## Run the tests by running
## pytest -v test_classification_tree.py
## All test functions must start with test_.
## echo $PATH
## export PATH="/Users/yuehanxiao/anaconda3/bin:$PATH"


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
    # Making sure the tree ends there.
    def Gini_index(p):
        return p*(1-p)
    node_test = sql_tree.SQLTree(cur, 'cancer_zero_test', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])
    assert node_test.right is None
    assert node_test.left is None


def test2():
    # Split feature is the same value.
    # making sure don't choose the split variable, which only has one value for all rows
    def Gini_index(p):
        return p*(1-p)
    node_test = sql_tree.SQLTree(cur, 'test_data', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])
    # According to the acknowledged node_test's split variables, it has two left split variables, none right split variable
    assert node_test.split_variable != 'b'
    assert node_test.left.split_variable != 'b'
    assert node_test.left.left.split_variable is None
    assert node_test.right.split_variable is None


def test3():
    # missing values
    # If there is missing values inside the dataset, throw an error.
    def Gini_index(p):
        return p*(1-p)
    with pytest.raises(ValueError):
        sql_tree.SQLTree(cur, 'test_data1', ['column_1', 'column_2'], 'response', Gini_index, [])


def test4():
    # making sure dataframe and SQL data base produce the same tree
    def Gini_index(p):
        return p*(1-p)
    test_data2 = np.array([[5, 2, 34, 0], [38, 20, 21, 0], [72, 20, 10, 0], [52, 20, 15, 1]])
    node_test = classification_tree.Node(test_data2, Gini_index, [])
    node_test2 = sql_tree.SQLTree(cur, 'test_data2', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])
    assert node_test.past_split == node_test2.past_split


def test5_R1():
    # making sure the split point is in the first column and be the right point.
    # test_data3 = np.array([[5, 0], [38, 0], [72, 0], [52, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = sql_tree.SQLTree(cur, 'test_data3', ['column_1'], 'response', Gini_index, [])
    assert node_test.split_variable == 'column_1'
    assert node_test.split_point == 72
    assert node_test.left.split_variable == 'column_1'
    assert node_test.left.split_point == 52
    assert node_test.right.split_variable is None
    assert node_test.right.split_point is None


def test6_R2():
    # test_data4 = np.array([[5, 34, 1], [38, 7, 1], [72, 15, 0], [52, 29, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = sql_tree.SQLTree(cur, 'test_data4', ['column_1', 'column_2'], 'response', Gini_index, [])
    assert node_test.split_variable is None
    assert node_test.split_point is None


def test7_R3():
    # test_data5 = np.array([[16, 24, 34, 1], [38, 20, 21, 0], [14, 20, 10, 1], [52, 20, 15, 1], [32, 59, 1, 0], [34, 21, 69, 1]])
    def Gini_index(p):
        return p*(1-p)
    node_test = sql_tree.SQLTree(cur, 'test_data5', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])
    assert node_test.split_variable == 'column_1'
    assert node_test.split_point == 16
    assert node_test.left.split_variable is None
    assert node_test.left.split_point is None


def test8():
    # test the smallest value in each column is not the split point
    def Gini_index(p):
        return p*(1-p)
    predictors = ['column_1', 'column_2', 'column_3', 'column_4']
    node_test = sql_tree.SQLTree(cur, 'test_table', predictors, 'response', Gini_index, [])
    for predictor in predictors:
        cur.execute('''SELECT min(''' + predictor + ''')'''
                    '''FROM test_table;''')
        test = cur.fetchone()[0]
        assert test not in node_test.past_split
