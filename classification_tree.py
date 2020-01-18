
#!/usr/bin/env python3

## Your code goes here.
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from functools import reduce


class Node:
    def __init__(self, dataset, impurity_function, past_split):
        """pass in the whole dataset and build a tree
        Parameters
        ---------
        past_split: a list
                    store the past split variable and split point
        dataset: matrix like
                 using pandas.dataframe format
        split_variable: character
                        variables that calculates to be the best to split the tree
        split_point: numeric
                     point that calculates to be the best split point of the selected variable
        right: node
               right side node, which is greater than the split_point
        left: node
              left side node, which is smaller than the split_point
        count_zero: int
                   number of 0 in the outcomes
        count_one: int
                   number of 1 in the outcomes
        prob_value_zero: float
                         the probability of getting outcome 1
        prob_value_one: float
                        the probability of getting outcome 0
        Returns
        ----------
        The whole tree: linked list
        """
        # turn the dataset into a pandas.dataframe
        dataset = pd.DataFrame(dataset)
        if dataset.isnull().values.any() == True:
            raise ValueError('incorrect input, dataset should not have missing value')
        if len(set(dataset.iloc[:,-1])) > 2:
            raise ValueError('incorrect input, responose should be binary value')
        if len(set(dataset.iloc[:, -1])) == 2:
            if np.unique(dataset.iloc[:,-1])[0] != 0 or np.unique(dataset.iloc[:,-1])[1] != 1:
                raise ValueError('incorrect input, responses should be binary value')
        if len(set(dataset.iloc[:, -1])) ==1:
            if (0 in np.unique(dataset.iloc[:,-1])) == False and (1 in np.unique(dataset.iloc[:,-1])) == False:
                raise ValueError('incorrect input, responses should be binary value')

        self.split_variable = None
        self.split_point = None
        self.left = None
        self.right = None
        self.count_zero = 0
        self.count_one = 0
        self.impurity_function = impurity_function
        self.leaf = True
        self.past_split = past_split.copy()
        # Assume last column is outcomes
        for count in dataset.iloc[:, -1]:
            if count == 1:
                self.count_one = self.count_one + 1
            else:
                self.count_zero = self.count_zero + 1
        if self.count_zero + self.count_one == 0:
            self.prob_value_zero = 0
        else:
            self.prob_value_zero = self.count_zero/(self.count_zero + self.count_one)
        if self.count_zero+self.count_one == 0:
            self.prob_value_one = 0
        else:
            self.prob_value_one = self.count_one/(self.count_zero+self.count_one)
        if self.prob_value_zero < 1 and self.prob_value_one < 1 and len(dataset) > 2:
            # if it doesn't meet the above condition, this is when the tree should stop from building.
            # put dataset into the best_split function and get the split_variable and split_point.
            self.split_variable, self.split_point = self.best_split(dataset, impurity_function)
            if self.split_variable != None and self.split_point != None:
            # assign the returned value of the best_split function to the self.split_variable and self.split_point.
                left_dataset, right_dataset = self.get_data(self.split_variable, self.split_point, dataset)
            # call get_data function and pass the above split variable and split point that we get for the (root) node into the get_data function to get
            # new dataset (subset) for the right node and the left node
            # passing the dataset that you get from the get_data function as the left dataset and right dataset.
            # the right and left node are going to init again and get their own children, these two lines below behave like a recursion
                self.leaf = False
                left_past_split = self.past_split.copy()
                left_past_split.append(str(self.split_variable) + str(self.split_point) + '<')
                self.left = Node(left_dataset, self.impurity_function, left_past_split)
                right_past_split = self.past_split.copy()
                right_past_split.append(str(self.split_variable) + str(self.split_point) + '>=')
                self.right = Node(right_dataset, self.impurity_function, right_past_split)

    def impurity(self, p_value, impurity_function):
        """using this function to find the maximum impurity value
        Parameter:
        ----------
        impurity_function: function
                           the different types of impurity function that users choose by themselves in order to find the max impurity among variables.
        p-value: float
                 each separated dataset for selected variables has its corresponding p_value, the number of 1 in the separated dataset for current node
        Return:
        ----------
        impurity_center_value: numeric
                               impurity value for the current node

        """
        return impurity_function(p_value)

    def impurity_reduction(self, p_value, left_dataset, right_dataset, impurity_function):
        """ using this function to find the maximum impurity value
        Parameter:
        ----------
        impurity_function: function
                           the different types of impurity function that users choose by themselves in order to find the max impurity among variables.
        p-value: float
                 each separated dataset for selected variables has its corresponding p_value, the number of 1 in the separated dataset
        Return:
        ----------
        max_impurity: numeric
                      max impurity value
        """
        if p_value > 1 or p_value < 0:
            raise ValueError('p_value out of the boundray')
        # p_value here is the probability of one
        # impurity_function is the equation
        impurity_center=self.impurity(p_value, impurity_function)
        count_one_left = 0
        count_one_right = 0
        # get the left dataset's p_value
        for count_left in left_dataset.iloc[:, -1]:
            if count_left == 1:
                count_one_left = count_one_left + 1
        if len(left_dataset) == 0:
            p_value_left = 0
        else:
            p_value_left=count_one_left/len(left_dataset)
        # get the right dataset's p-value
        for count_right in right_dataset.iloc[:, -1]:
            if count_right == 1:
                count_one_right = count_one_right+1
        if len(right_dataset) == 0:
            p_value_right = 0
        else:
            p_value_right = count_one_right/len(right_dataset)
        if p_value_left == 0 or 1 - p_value_left == 0:
            impurity_left = 0
        else:
            impurity_left = self.impurity(p_value_left, impurity_function)
        if p_value_right == 0 or 1 - p_value_right == 0:
            impurity_right = 0
        else:
            impurity_right = self.impurity(p_value_right, impurity_function)
        prob_left = len(left_dataset)/(len(left_dataset) + len(right_dataset))
        prob_right = len(right_dataset)/(len(left_dataset) + len(right_dataset))
        max_impurity_reduction = impurity_center - prob_left * impurity_left - prob_right * impurity_right
        return max_impurity_reduction

    def best_split(self, dataset, impurity_function):
        """ Using this function to get the best split variable and split point for the tree.
        Parameters
        ----------
        dataset: pandas.dataframe formatting
                 either the whole dataset at the beginning or the sub dataset from the get_data function
        Return
        ---------
        split_variable: character
                        split variable of current node
        split_point: numeric
                     split point of current node
        """
        final_lst = defaultdict(list)
        for variable in dataset.columns[:-1]:
            # Making sure doesn't choose the Y column.
            # this is actually the examined split_variable
            # order the dataset for each variable.
            impurity_lst = []
            for test_splitpoint in dataset[variable][::2]:
                left_dataset, right_dataset = self.get_data(variable, test_splitpoint, dataset)
                # the range is 2
                # store impurity_value for each split in a list and append every time for new split and its corresponding impurity value also as a list,
                # nested list
                impurity_value = self.impurity_reduction(self.prob_value_one,left_dataset, right_dataset, impurity_function)
                impurity_lst.append([test_splitpoint, impurity_value])
            max_test_splitpoint = max(impurity_lst, key=lambda x: x[1])
            # According to the max impurity of the selected variable with corresponding split point, append this list into a dictionary for future comparison
            # between all variables
            final_lst[variable].append(max_test_splitpoint)
        # find the maximum impurity value of all varibles
        final = max(final_lst, key = lambda k : final_lst[k][0][1])
        #make sure it doesn't have max reduction for the split variable is 0 and for
        #other variables are negative's situation.
        for test in final_lst.keys():
            if test != final:
                if final_lst[test][0][1] <=0 and final_lst[final][0][1]==0:
                    return None, None
        split_variable = final
        split_point = final_lst[final][0][0]
        # checking if split feature has the same value
        for key in final_lst.keys():
            if len(np.unique(dataset[split_variable])) == 1:
            # split feature has the same value
                final_lst[split_variable][0][1] = 0
                final = max(final_lst, key = lambda k : final_lst[k][0][1])
                split_variable = final
                split_point = final_lst[final][0][0]
        return split_variable, split_point

    def get_data(self, split_variable, split_point, dataset):
        """ delete the lines from the whole dataset based on the split_point in order to get sub dataset for left and right sided nodes
            this function already helps to sort the dataset
        Parameters:
        ----------
        split_variable: character
                        the split_variable that we get from best_split function
        split_point: numeric
                     the split_point that we get from best_split function
        Return
        ----------
        left dataset: pandas.dataframe
                      the split_variable's values that are samller than split_point
        right dataset: pandas.dataframe
                       the split_variable's vlaue that are bigger than split_point
        """
        # removing all the lines that have the selected variable's value greater than split_point
        left_dataset = dataset[~(dataset[split_variable] >= split_point)]
        # removing all the lines that have the selected variable's value smaller than split point
        right_dataset = dataset[~(dataset[split_variable] < split_point)]
        return left_dataset, right_dataset
        # pass the left and right dataset back to the init function and continue to build a tree.'''

    def prune(self, node, min_error_alpha, whole_dataset):
        """Prune the tree until it doens't need to be pruned.
        Parameter:
        ----------
        node: the tree
              the tree that I built previously
        min_error_alpha: int
                         the returned value of cross_validation function
        Return:
        ----------
        return the modified tree
        """
        # if the left node of current node doesn't have leaf
        if node.left.leaf == False:
            self.prune(node.left, min_error_alpha, whole_dataset)
        # if the right node of current node doesn't have leaf
        if node.right.leaf == False:
            self.prune(node.right, min_error_alpha, whole_dataset)
        if node.left.leaf == True and node.right.leaf == True:
            # call g(t) function to get the minium of the g(t) which is the alpha stare and then comapre with alpha which
            # get from the cross validation function if the returned alpha star is smaller than alpha then prune the tree
            # otherwise don't prune the tree'''
            alpha_star = node.G_T(whole_dataset)
            if alpha_star < min_error_alpha:
                # prune the tree
                node.left = None
                node.right = None
        return
        # return the modified tree

    def G_T(self, whole_dataset):
        """Calculate the alpha_star in order to compare with alpha and to determin whether the node should be pruned or not.
        Parameter:
        ----------
        whole_dataset: pandas.dataframe
                       whole dataset, without any changes
        Return:
        ----------
        alpha_star: numeric
                    using the equation to get the alpha_star of each node
        """
        # We are on the node and we traverse down to this node through prune function.
        # We are on the node that is above its children. Only this one node
        num_rows = len(whole_dataset)
        if self.count_zero > self.count_one:
            misclassified_t = self.prob_value_one
        else:
            misclassified_t = self.prob_value_zero
        all_points_t = (self.count_one + self.count_zero)/num_rows
        R_t = misclassified_t * all_points_t
        if self.left.count_zero > self.left.count_one:
            misclassified_left = self.left.prob_value_one
        else:
            misclassified_left = self.left.prob_value_zero
        if self.right.count_zero > self.right.count_one:
            misclassified_right = self.right.prob_value_one
        else:
            misclassified_right = self.right.prob_value_zero
        all_points_left = (self.left.count_one + self.left.count_zero)/num_rows
        all_points_right = (self.right.count_one+self.right.count_zero)/num_rows
        R_left = misclassified_left * all_points_left
        R_right = misclassified_right * all_points_right
        R_Tt = R_left + R_right
        alpha_star = R_t + R_Tt
        # alpha_star is what we get based on the g_t and we need to compare the alpha_star with the alpha, which we get from the cross validation'''
        return alpha_star

    def query_node(self, row):
        """Traverse through the tree and get the prediction.
        Each row of test_dataset will be compared with the split_variable's split point in order to
        decide which path, left or right, should go in order to get the predicted outcomes
        Parameter:
        ----------
        row: list
             each row of the test_dataset
        node: tree
              The tree that we built previously
        Return:
        ----------
        prediction:
        """
        # on the leaf
        if self.right is None and self.left is None:
            if self.count_one > self.count_zero:
                prediction = 1
                return prediction
            else:
                prediction = 0
                return prediction
        # The top node's split_variable and split_point
        if row[self.split_variable] < self.split_point:
            # recursive function to query_node again
            return self.left.query_node(row)
        if row[self.split_variable] >= self.split_point:
            return self.right.query_node(row)

    def show(self, level=0):
        """Print out the tree in an appealing way."""

        print(" " * level, self.split_variable, ": ", self.split_point, sep="")

        if self.left is not None:
            print(" " * level, "left:", sep="")
            self.left.show(level + 2)

        if self.right is not None:
            print(" " * level, "right:", sep="")
            self.right.show(level + 2)

    def get_data_new(self, dataset):
        '''use past split to get data'''
        for split in self.past_split:
            if split[-1] == '<':
                dataset = dataset[~(dataset[split[0]] >= split[1])]
            else:
                dataset = dataset[~(dataset[split[0]] < split[1])]
        return dataset



    def is_valid(self, dataset):
        '''Checking the validation of a classification tree
        Parameter:
        ----------
        Node: tree
        Tree that we previously built
        dataset: pandas.data.frame
                 the whole dataset
        '''
        # 1st valid test: no node is empty
        if self.leaf == False:
            assert self.split_variable != None
        else:
            assert self.count_one > 0 or self.count_zero > 0
        # 2nd valid test: all data points in the left child should have xj < s and all
        # points in the right child should have xj> s.
        if self.leaf == False:
            new_dataset = self.get_data_new(dataset)
            assert (new_dataset[self.split_variable] < self.split_point).all() == True
            assert (new_dataset[self.split_variable] < self.split_point).all() == True
            return self.left.is_valid(dataset) + self.right.is_valid(dataset)
        else:
            # 3rd valid test, parents has the same number of the children's number of dataset.
            leaves_dataset_length = len(self.get_data_new(dataset))
            return leaves_dataset_length


class Tree:

    def __init__(self, dataset, impurity_function):
        self.Tnode = Node(dataset, impurity_function, [])
        self.prune(dataset, impurity_function)

    def prune(self, dataset, impurity_function):
        """Prune the tree until it doens't need to be pruned.
        Parameter:
        ----------
        node: the tree
              the tree that I built previously
        min_error_alpha: int
                         the returned value of cross_validation function
        Return:
        ----------
        return the modified tree
        """
        # if the left node of current node doesn't have leaf
        min_error_alpha = self.cross_validation(dataset, impurity_function)
        self.Tnode.prune(self.Tnode, min_error_alpha, dataset)

    def cross_validation(self, whole_dataset, impurity_function):
        """Providing a alpha_range, i.e(1,10), calculate the test set error for each alpha and choose the
        alpha that minimizes the test set error.
        Parameter:
        ----------
        alpha_range: list
                     it is chose by outselves
        whole_dataset: pandas.dataframe
                   the original dataset with all the variables and rows.
        Return:
        ----------
        min_error_alpha: int
                     alpha that minimize the test set error
        """
    # whole_dataset = pd.DataFrame(dataset)
    # Assume the dataset is random
    # 5 is the number of folds that set by oursleves, I will set it as 5
        split = len(whole_dataset)//5#note to be revised
        alpha = []
        for i in list(range(1,11)):
            check = 0.0000000001 * (10**i)
            alpha.append(check)
        all_alpha_error = defaultdict(list)
        for alpha_range in alpha:
            avg_train_tree_error = []
            for k in range(5):
            # note to be revised, must have dataset greater than 5 rows, otherwise cannot split into 5 folds.
            # get the number of rows of the whole dataset
                test_dataset = whole_dataset[split*k : split*(k+1)]
            # delete the rows which are test_dataset
                train_dataset = whole_dataset.drop(whole_dataset.index[split*k : split*(k+1)]).copy()
                train_tree = Node(train_dataset, impurity_function, [])
                train_tree.prune(train_tree, alpha_range, whole_dataset) # train_tree = self
                count_wrong = 0
                for x in range(len(test_dataset)):
                # get every row index
                # call query_node function to get the predicted result
                    row = test_dataset.iloc[x]
                # count the number of wrong and correct predictions based on the predicted results
                    if row[-1] != train_tree.query_node(row):
                        count_wrong = count_wrong + 1
                avg_train_tree_error.append(count_wrong)
        #average the 5 errors of the same alpha but different training trees
            alpha_error = reduce(lambda a, b: a + b, avg_train_tree_error) / len(avg_train_tree_error)
            all_alpha_error[alpha_range].append(alpha_error)
        min_error_alpha = min(all_alpha_error, key = all_alpha_error.get)
        return min_error_alpha

    def query_node(self, dataset):
        """Aiming for benchmark. Traverse through the tree and get the prediction.
        Each row of test_dataset will be compared with the split_variable's split point in order to
        decide which path, left or right, should go in order to get the predicted outcomes
        Parameter:
        ----------
        row: list
             each row of the test_dataset
        node: tree
              The tree that we built previously
        Return:
        ----------
        prediction:
        """
        prediction = []
        for index in list(dataset.index):
            row = dataset.iloc[index, :-1]
            prediction_row = Node.query_node(self.Tnode, row)
            prediction.append(prediction_row)
        return prediction


class tree_to_forest(Node):

    def __init__(self, whole_dataset, k, impurity_function):
        self.k = k
        super().__init__(whole_dataset, impurity_function, [])

    def best_split(self, dataset, impurity_function):
        """ Using this function to get the best split variable and split point for the tree.
        Parameters
        ----------
        dataset: pandas.dataframe formatting
                 either the whole dataset at the beginning or the sub dataset from the get_data function
        Return
        ---------
        split_variable: character
                        split variable of current node
        split_point: numeric
                     split point of current node
        """
        final_lst = defaultdict(list)
        Y = dataset.iloc[:, -1]
        data_k = dataset.iloc[:, :-1].sample(n=self.k, axis = 1) # dataset is the random data we import in _init_
        data_k['y'] = Y
        for variable in data_k.columns[:-1]:
            impurity_lst = []
            for test_splitpoint in dataset[variable][::2]:
                left_dataset, right_dataset = self.get_data(variable, test_splitpoint, dataset)
                impurity_value = self.impurity_reduction(self.prob_value_one,left_dataset, right_dataset, impurity_function)
                impurity_lst.append([test_splitpoint, impurity_value])
            max_test_splitpoint = max(impurity_lst, key=lambda x: x[1])
            final_lst[variable].append(max_test_splitpoint)
        final = max(final_lst, key = lambda k : final_lst[k][0][1])
        for test in final_lst.keys():
            if test != final:
                if final_lst[test][0][1] <=0 and final_lst[final][0][1]==0:
                    return None, None
        split_variable = final
        split_point = final_lst[final][0][0]
        for key in final_lst.keys():
            if len(np.unique(dataset[split_variable])) == 1:
            # split feature has the same value
                final_lst[split_variable][0][1] = 0
                final = max(final_lst, key = lambda k : final_lst[k][0][1])
                split_variable = final
                split_point = final_lst[final][0][0]
        return split_variable, split_point 

class Random_Forest():

    def __init__(self, whole_dataset, k, n_tree, impurity_function):
        self.forest = []
        for T in range(n_tree):
            n = random.randint(1, len(whole_dataset))
            random_data = whole_dataset.sample(n=n)
            tree = tree_to_forest(random_data, k, impurity_function)
            self.forest.append(tree)


    def prediction(self, user_dataset):
        forest_prediction = []
        for tree in self.forest:
            if tree is None:
                forest_prediction.append(list(user_dataset[:, -1]))
            else:
                single_tree_prediction = []
                for x in range(len(user_dataset)):
                    #assume user's dataset has y column
                    row = user_dataset.iloc[x][:-1]
                    prediction = tree.query_node(row)
                    single_tree_prediction.append(prediction)
                forest_prediction.append(single_tree_prediction)
        forest_prediction = pd.DataFrame(forest_prediction)
            # every row is a single tree's predictino so we need to get the mode for every column
            # in order to get same row's prediction for the whole forest
        outcome = forest_prediction.mode()
        return outcome



#Node(test, Gini_index).show()
#Tree(test, Gini_index)
#foo = Node(test, gini)
#Tree
#foo.show()
#Forest(test, 5, 2)

#node = Node(test, Gini_index)
#Node.is_valid(node, node, test)
#Tree(test, Gini_index)
#test_forest = Random_Forest(test, 2, 5, Gini_index)
#sth = test_forest.prediction(test)
#print(sth)
#Node(test, Gini_index)
#Tnode.prune(Tnode, 0.1, test)
#Node.query_node(Node(test, Gini_index), row)

#Node(test, Gini_index)
#node.is_valid(test)
'''
predict =  pd.DataFrame([{'a':-0.28696,'b':3.1784,'c':-3.5767,'d':-3.1896,'y':1}, {'a':-0.11996,'b':6.8741,'c':0.91995,'d':-0.6694,'y':0}, {'a':0.93584,'b':8.8855,'c':-1.6831,'d':-1.6599,'y':0}])
test = pd.DataFrame([{'a':2.2517,'b':-5.1422,'c':4.2916,'d':-1.2487,'y':0}, {'a':5.504,'b':10.3671,'c':-4.413,'d':-4.0211,'y':0}, {'a':2.8521,'b':9.171,'c':-3.6461,'d':-1.2047,'y':0},
{'a':1.1676,'b':9.1566,'c':-2.0867,'d':-0.80647,'y':0,}, {'a':2.6104,'b':8.0081,'c':-0.23592,'d':-1.7608,'y':0}, {'a':0.32444,'b':10.067,'c':-1.1982,'d':-4.1284,'y':0}, {'a':-1.3971,'b':3.3191,'c':-1.3927,'d':-1.9948,'y':1},
{'a':0.39012,'b':-0.14279,'c':-0.031994,'d':0.35084,'y':1}, {'a':-1.6677,'b':-7.1535,'c':7.8929,'d':0.96765,'y':1}, {'a':-3.8483,'b':-12.8047,'c':15.6824,'d':-1.281,'y':1},
{'a':-3.5681,'b':-8.213,'c':10.083,'d':0.96765,'y':1}, {'a':-2.2804,'b':-0.30626,'c':1.3347,'d':1.3763,'y':1}, {'a':-1.7582,'b':2.7397,'c':-2.5323,'d':-2.234,'y':1}, {'a':-0.89409,'b':3.1991,'c':-1.8219,'d':-2.9452,'y':1},
{'a':-2.7143,'b':11.4535,'c':2.1092,'d':-3.9629,'y':0}, {'a':3.8244,'b':-3.1081,'c':2.4537,'d':0.52024,'y':0}, {'a':2.7961,'b':2.121,'c':1.8385,'d':0.38317,'y':0},
{'a':3.5358,'b':6.7086,'c':-0.81857,'d':0.47886,'y':0}, {'a':-0.7056,'b':8.7241,'c':2.2215,'d':-4.5965,'y':0}])
def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)

#Random_Forest(test, 2, 5, Gini_index)
'''


#Tnode = Tree(test, Gini_index)
#Tree.query_node(Tnode, predict)
'''
test = pd.DataFrame([{'a':2.2517,'b':-5.1422,'c':4.2916,'d':-1.2487,'y':0}, {'a':5.504,'b':10.3671,'c':-4.413,'d':-4.0211,'y':0}, {'a':2.8521,'b':9.171,'c':-3.6461,'d':-1.2047,'y':0},
{'a':1.1676,'b':9.1566,'c':-2.0867,'d':-0.80647,'y':0,}, {'a':2.6104,'b':8.0081,'c':-0.23592,'d':-1.7608,'y':0}, {'a':0.32444,'b':10.067,'c':-1.1982,'d':-4.1284,'y':0}, {'a':-1.3971,'b':3.3191,'c':-1.3927,'d':-1.9948,'y':1},
{'a':0.39012,'b':-0.14279,'c':-0.031994,'d':0.35084,'y':1}, {'a':-1.6677,'b':-7.1535,'c':7.8929,'d':0.96765,'y':1}, {'a':-3.8483,'b':-12.8047,'c':15.6824,'d':-1.281,'y':1},
{'a':-3.5681,'b':-8.213,'c':10.083,'d':0.96765,'y':1}, {'a':-2.2804,'b':-0.30626,'c':1.3347,'d':1.3763,'y':1}, {'a':-1.7582,'b':2.7397,'c':-2.5323,'d':-2.234,'y':1}, {'a':-0.89409,'b':3.1991,'c':-1.8219,'d':-2.9452,'y':1},
{'a':-2.7143,'b':11.4535,'c':2.1092,'d':-3.9629,'y':0}, {'a':3.8244,'b':-3.1081,'c':2.4537,'d':0.52024,'y':1}, {'a':2.7961,'b':2.121,'c':1.8385,'d':0.38317,'y':0},
{'a':3.5358,'b':6.7086,'c':-0.81857,'d':0.47886,'y':1}, {'a':-0.7056,'b':8.7241,'c':2.2215,'d':-4.5965,'y':0}, {'a':2.7961,'b':2.121,'c':1.8385,'d':0.38317,'y':1}, {'a':2,'b':2.121,'c':1,'d':0.5,'y':1},
{'a':2.7,'b':3,'c':5,'d':0.8,'y':1}, {'a':0.961,'b':0.121,'c':1.99,'d':0.38317,'y':1}, {'a':5.2,'b':2.2,'c':1.8,'d':0.3,'y':1}])
'''
'''
test = pd.DataFrame([{'a': 25, 'b': 45, 'c':78, 'y':0}, {'a': 10, 'b': 20, 'c':15, 'y':0}, {'a': 99, 'b': 50, 'c':4, 'y':0},
{'a': 98, 'b': 4, 'c':55, 'y':1}, {'a': 18, 'b': 13, 'c':16, 'y':0}, {'a': 9, 'b': 10, 'c':39, 'y':1}, {'a': 29, 'b': 58, 'c':77, 'y':0},
{'a': 41, 'b': 72, 'c':71, 'y':1}, {'a': 87, 'b': 23, 'c':12, 'y':1}, {'a': 18, 'b': 34, 'c':27, 'y':1}])


def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)
'''
'''
Node(test, Gini_index, [])
'''
'''
Node(iris_data, Gini_index, []).show()
'''
'''
test = pd.DataFrame([{'a': 25, 'b': 36, 'c':78, 'y':0}, {'a': 10, 'b': 20, 'c':15, 'y':0}, {'a': 99, 'b': 50, 'c':4, 'y':0},
{'a': 98, 'b': 4, 'c':55, 'y':1}, {'a': 20, 'b': 40, 'c':68, 'y':1}])

user_dataset = pd.DataFrame([{'a': 25, 'b': 45, 'c':78, 'y':0}, {'a': 10, 'b': 20, 'c':15, 'y':0}, {'a': 99, 'b': 50, 'c':4, 'y':0},
{'a': 98, 'b': 4, 'c':55, 'y':1}, {'a': 18, 'b': 13, 'c':16, 'y':0}, {'a': 9, 'b': 10, 'c':39, 'y':1}, {'a': 29, 'b': 58, 'c':77, 'y':0},
{'a': 41, 'b': 72, 'c':71, 'y':1}, {'a': 87, 'b': 23, 'c':12, 'y':1}, {'a': 18, 'b': 34, 'c':27, 'y':1}])

row = {'a': 36, 'b': 36, 'c':78, 'y':0}
alpha_range = [1e-09, 1e-08, 1.0000000000000001e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0]

test_data = pd.DataFrame([{'a': None, 'b': 20, 'c':34, 'y':0}, {'a': 38, 'b': 20, 'c':21,'y':0}, {'a': 72, 'b': 20, 'c':10,'y':0}, {'a': 52, 'b': 20, 'c':15,'y':1}])
#row1 =  {'a': 10, 'b': 20, 'c':15, 'y':0}
#row3 ={'a': 20, 'b': 40, 'c':68, 'y':1}
test_data = np.array([[2, 1, 3], [3, 1, 4], [2, 1, 5], [2, 1, 6], [2, 1, 17]])
'''

#tree_to_forest(test, 2, Gini_index)
#Random_Forest(test, 2, 5, Gini_index)
#def Gini_index(p):
#        return p*(1-p)
#test_data4 = np.array([[5, 34, 1], [38, 7, 1], [72, 15, 0], [52, 29, 1]])
#node_test = Node(test_data4, Gini_index, []).show()
'''
test = pd.DataFrame([{'a':2.2517,'b':-5.1422,'c':4.2916,'d':-1.2487,'y':0}, {'a':5.504,'b':10.3671,'c':-4.413,'d':-4.0211,'y':0}, {'a':2.8521,'b':9.171,'c':-3.6461,'d':-1.2047,'y':0},
{'a':1.1676,'b':9.1566,'c':-2.0867,'d':-0.80647,'y':0,}, {'a':2.6104,'b':8.0081,'c':-0.23592,'d':-1.7608,'y':0}, {'a':0.32444,'b':10.067,'c':-1.1982,'d':-4.1284,'y':0}, {'a':-1.3971,'b':3.3191,'c':-1.3927,'d':-1.9948,'y':1},
{'a':0.39012,'b':-0.14279,'c':-0.031994,'d':0.35084,'y':1}, {'a':-1.6677,'b':-7.1535,'c':7.8929,'d':0.96765,'y':1}, {'a':-3.8483,'b':-12.8047,'c':15.6824,'d':-1.281,'y':1},
{'a':-3.5681,'b':-8.213,'c':10.083,'d':0.96765,'y':1}, {'a':-2.2804,'b':-0.30626,'c':1.3347,'d':1.3763,'y':1}, {'a':-1.7582,'b':2.7397,'c':-2.5323,'d':-2.234,'y':1}, {'a':-0.89409,'b':3.1991,'c':-1.8219,'d':-2.9452,'y':1},
{'a':-2.7143,'b':11.4535,'c':2.1092,'d':-3.9629,'y':0}, {'a':3.8244,'b':-3.1081,'c':2.4537,'d':0.52024,'y':1}, {'a':2.7961,'b':2.121,'c':1.8385,'d':0.38317,'y':0},
{'a':3.5358,'b':6.7086,'c':-0.81857,'d':0.47886,'y':1}, {'a':-0.7056,'b':8.7241,'c':2.2215,'d':-4.5965,'y':0}, {'a':2.7961,'b':2.121,'c':1.8385,'d':0.38317,'y':1}, {'a':2,'b':2.121,'c':1,'d':0.5,'y':1},
{'a':2.7,'b':3,'c':5,'d':0.8,'y':1}, {'a':0.961,'b':0.121,'c':1.99,'d':0.38317,'y':1}, {'a':5.2,'b':2.2,'c':1.8,'d':0.3,'y':1}])
def Gini_index(p):
       return p*(1-p)
Node(test, Gini_index, []).show()
'''