import classification_tree
import secret
import pandas as pd
import psycopg2
from collections import defaultdict
from sklearn.datasets import load_breast_cancer

conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

cur = conn.cursor()


class SQLTree(classification_tree.Node):
    def __init__(self, cur, table, predictors, response, impurity_function, past_split):
        """Pass in the sql database name and build a tree
        Parameters
        ---------
        cur: the connection with the sql database
        table: the table name that created in sql database
                string
        predictors: a list of column names in sql database
                    list
        response: a string, Y
        impurity_function: a function
                            user input
        past_split: a list
                    store the past split variable and split point
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
        The whole tree
        """
        # testing whether the original dataset has missing values or not
        test_missing = cur.execute('''SELECT *
                                       FROM ''' + table +
                                   ''' WHERE ''' + ' IS NULL OR '.join(predictors) +
                                   ''' IS NULL OR ''' + response + ''' IS NULL;''')
        test_missing = cur.fetchone()
        if test_missing is not None:
            raise ValueError('incorrect input, dataset should not have missing value')
        self.table = table
        self.cur = cur
        self.predictors = predictors
        self.split_variable = None
        self.split_point = None
        self.left = None
        self.right = None
        self.impurity_function = impurity_function
        self.leaf = True
        self.response = response
        self.past_split = past_split.copy()
        if self.past_split == []:
            self.count_one = self.cur.execute('''SELECT COUNT(*) ''' +
                                              '''FROM ''' + self.table +
                                              ''' WHERE ''' + response + '''= 1;''')
            self.count_one = self.cur.fetchone()[0]
            self.count_zero = self.cur.execute('''SELECT COUNT(*) ''' +
                                               '''FROM ''' + self.table +
                                               ''' WHERE ''' + response + '''= 0;''')
            self.count_zero = self.cur.fetchone()[0]
            self.prob_value_one = self.cur.execute('''SELECT fraction_ones(''' + response + ''')''' +
                                                   '''FROM ''' + self.table + ';')
            self.prob_value_one = self.cur.fetchone()[0]
            self.prob_value_zero = 1 - self.prob_value_one
        else:
            self.count_one = self.cur.execute('''SELECT COUNT(''' + response + ''')''' +
                                              '''FROM ''' + self.table +
                                              ''' WHERE ''' + ' AND '.join(self.past_split) +
                                              ''' AND ''' + response + '=' + '1' + ''';''')
            self.count_one = self.cur.fetchone()[0]
            self.count_zero = self.cur.execute('''SELECT COUNT(''' + response + ''')''' +
                                               '''FROM ''' + self.table +
                                               ''' WHERE ''' + ' AND '.join(self.past_split) +
                                               ''' AND ''' + response + '''= 0;''')
            self.count_zero = self.cur.fetchone()[0]
            self.prob_value_one = self.cur.execute('''SELECT fraction_ones(''' + response + ''')''' +
                                                   '''FROM ''' + self.table +
                                                   ''' WHERE ''' + ' AND '.join(self.past_split) + ';')
            self.prob_value_one = self.cur.fetchone()[0]
            self.prob_value_zero = 1 - self.prob_value_one
        if self.prob_value_zero < 1 and self.prob_value_one < 1:
            self.split_variable, self.split_point = self.best_split(impurity_function)
            print(self.split_variable, self.split_point)
            if self.split_variable is not None and self.split_point is not None:
                self.split_point = self.split_point[0]
                self.leaf = False
                left_past_split = self.past_split.copy()
                left_past_split.append(str(self.split_variable) + ' < ' + str(self.split_point))
                self.left = SQLTree(self.cur, self.table, self.predictors, self.response, self.impurity_function, left_past_split)
                right_past_split = self.past_split.copy()
                right_past_split.append(str(self.split_variable) + ' >= ' + str(self.split_point))
                self.right = SQLTree(self.cur, self.table, self.predictors, self.response, self.impurity_function, right_past_split)

    def best_split(self, impurity_function):
        """ Using this function to get the best split variable and split point for the tree.
        Parameters
        ----------
        Impurity_function: Gini_index/Cross_entropy/Bayes_error/User_input
        Return
        ---------
        split_variable: character
                        split variable of current node
        split_point: numeric
                     split point of current node
        """
        final_lst = defaultdict(list)
        for variable in self.predictors:
            impurity_lst = []
            test_column = self.cur.execute('SELECT DISTINCT ' + str(variable) +
                                           ' FROM ' + self.table + ';')
            test_column = list(self.cur.fetchall())
            for test_splitpoint in test_column:
                past_split_left = self.past_split + [str(variable) + ' < ' + str(test_splitpoint[0])]
                past_split_right = self.past_split + [str(variable) + ' >= ' + str(test_splitpoint[0])]
                test_zero_left = self.cur.execute('SELECT COUNT(*)' +
                                                  ' FROM ' + self.table +
                                                  ' WHERE ' + ' AND '.join(past_split_left) + ';')
                test_zero_left = self.cur.fetchone()[0]
                test_zero_right = self.cur.execute('SELECT COUNT(*)' +
                                                   ' FROM ' + self.table +
                                                   ' WHERE ' + ' AND '.join(past_split_right) + ';')
                test_zero_right = self.cur.fetchone()[0]
                if str(variable) + ' >= ' + str(test_splitpoint[0]) not in self.past_split and str(variable) + ' < ' + str(test_splitpoint[0]) not in self.past_split:
                    if self.past_split == []:
                        impurity_center = self.cur.execute('SELECT fraction_ones(' + self.response + ')' +
                                                           ' FROM ' + self.table + ';')
                        impurity_center = self.cur.fetchone()[0]
                    else:
                        impurity_center = self.cur.execute('SELECT fraction_ones(' + self.response + ')' +
                                                           ' FROM ' + self.table +
                                                           ' WHERE ' + ' AND '.join(self.past_split) + ';')
                        impurity_center = self.cur.fetchone()[0]
                    if test_splitpoint[0] == min(test_column)[0]:
                        impurity_left = 0
                    else:
                        if test_zero_left != 0 and test_zero_right != 0:
                            impurity_left = self.cur.execute('SELECT fraction_ones(' + self.response + ')' +
                                                             ' FROM ' + self.table +
                                                             ' WHERE ' + ' AND '.join(past_split_left) + ';')
                            impurity_left = self.cur.fetchone()[0]

                            impurity_right = self.cur.execute('SELECT fraction_ones(' + self.response + ')' +
                                                              ' FROM ' + self.table +
                                                              ' WHERE ' + ' AND '.join(past_split_right) + ';')
                            impurity_right = self.cur.fetchone()[0]
                            len_left_dataset = self.cur.execute('SELECT COUNT(*)' +
                                                                ' FROM ' + self.table +
                                                                ' WHERE ' + ' AND '.join(past_split_left) + ';')
                            len_left_dataset = self.cur.fetchone()[0]
                            len_right_dataset = self.cur.execute('SELECT COUNT(*)' +
                                                                 ' FROM ' + self.table +
                                                                 ' WHERE ' + ' AND '.join(past_split_right) + ';')
                            len_right_dataset = self.cur.fetchone()[0]
                            prob_left = len_left_dataset/(len_left_dataset + len_right_dataset)
                            prob_right = len_right_dataset/(len_left_dataset + len_right_dataset)
                            max_impurity_reduction = impurity_center - prob_left * impurity_left - prob_right * impurity_right
                            impurity_lst.append([test_splitpoint, max_impurity_reduction])
            # Concer Case: Where the impurity list doesn't contain anything
            if impurity_lst == []:
                return None, None
            max_test_splitpoint = max(impurity_lst, key=lambda x: x[1])
            final_lst[variable].append(max_test_splitpoint)
        final = max(final_lst, key = lambda k: final_lst[k][0][1])
        # Coner Case: if splitted dataset is empty, we stop building the tree
        if final == min(test_column):
            return None, None
        # Coner Case: making sure it doesn't have max reduction for the split variable is 0 and for
        # other variables are negative's situation.
        for test in final_lst.keys():
            if test != final:
                if final_lst[test][0][1] <=0 and final_lst[final][0][1]==0:
                    return None, None
        split_variable = final
        split_point = final_lst[final][0][0]
        # Checking if the feature has the same values
        len_variable = self.cur.execute('SELECT COUNT( DISTINCT ' + split_variable + ')'
                                 'FROM ' +  self.table + ';')
        len_variable = self.cur.fetchone()[0]
        for key in final_lst.keys():
            if len_variable == 1:
                final_lst[split_variable][0][1] = 0
                final = max(final_lst, key = lambda k : final_lst[k][0][1])
                split_variable = final
                split_point = final_lst[final][0][0]
        return split_variable, split_point

    def show(self, level=0):
        """Print out the tree in an appealing way."""
        print(" " * level, self.split_variable, ": ", self.split_point, sep="")

        if self.left is not None:
            print(" " * level, "left:", sep="")
            self.left.show(level + 2)

        if self.right is not None:
            print(" " * level, "right:", sep="")
            self.right.show(level + 2)

    def is_valid(self):
        '''Checking the validation of a classification tree
        Parameter:
        ----------
        Node: tree
        Tree that we previously built
        dataset: pandas.data.frame
                 the whole dataset
        '''
        # 1st valid test: no node is empty
        if self.leaf is False:
            assert self.split_variable is not None
        else:
            assert self.count_one > 0 or self.count_zero > 0
        # 2nd valid test: test parent node's dataset has the same length as its
        # children's node's dataset
        if self.leaf is False:
            if self.past_split == []:
                leaves_left_length = self.cur.execute('''SELECT COUNT(*)
                                                      FROM ''' + self.table +
                                                      ''' WHERE ''' + str(self.split_variable) + ' < ' + str(self.split_point) +
                                                      ''';''')
                leaves_left_length = self.cur.fetchone()[0]
                leaves_right_length = self.cur.execute('''SELECT COUNT(*)
                                                       FROM ''' + self.table +
                                                       ''' WHERE ''' + str(self.split_variable) + ' >= ' + str(self.split_point) +
                                                       ''';''')
                leaves_right_length = self.cur.fetchone()[0]
                parent_length = self.cur.execute('''SELECT COUNT(*)
                                                  FROM ''' + self.table + ''';''')
                parent_length = self.cur.fetchone()[0]
                assert parent_length == leaves_left_length + leaves_right_length
                return self.left.is_valid() + self.right.is_valid()
            else:
                leaves_left_length = self.cur.execute('''SELECT COUNT(*)
                                                      FROM ''' + self.table +
                                                      ''' WHERE ''' + ' AND '.join(self.past_split) +
                                                      '''AND''' + str(self.split_variable) + ' < ' + str(self.split_point) +
                                                      ''';''')
                leaves_left_length = self.cur.fetchone()[0]
                leaves_right_length = self.cur.execute('''SELECT COUNT(*)
                                                       FROM ''' + self.table +
                                                       ''' WHERE ''' + ' AND '.join(self.past_split) +
                                                       '''AND''' + str(self.split_variable) + ' >= ' + str(self.split_point) +
                                                       ''';''')
                leaves_right_length = self.cur.fetchone()[0]
                parent_length = self.cur.execute('''SELECT COUNT(*)
                                                 FROM ''' + self.table +
                                                 ' AND '.join(self.past_split) + ''';''')
                parent_length = self.cur.fetchone()[0]
                assert parent_length == leaves_left_length + leaves_right_length
                return self.left.is_valid() + self.right.is_valid()
        else:
            leaves_dataset_length = self.cur.execute('''SELECT COUNT(*)
                                                     FROM ''' + self.table +
                                                     ''' WHERE ''' + ' AND '.join(self.past_split))
            leaves_dataset_length = self.cur.fetchone()[0]
            return leaves_dataset_length

    def prune(self, min_error_alpha):
        """Prune the tree until it doens't need to be pruned.
        Parameter:
        ----------
        min_error_alpha: int
                         user input
        Return:
        ----------
        return the modified tree
        """
        # if the left node of current node doesn't have leaf
        if self.left is not None and self.right is not None:
            if self.left.leaf is False:
                self.left.prune(min_error_alpha)
        # if the right node of current node doesn't have leaf
            if self.right.leaf is False:
                self.right.prune(min_error_alpha)
            if self.left.leaf is True and self.right.leaf is True:
                alpha_star = self.G_T()
                if alpha_star < min_error_alpha:
                    # prune the tree
                    self.left = None
                    self.right = None
            return
            # return the modified tree

    def G_T(self):
        """Calculate the alpha_star in order to compare with alpha and to determin whether the node should be pruned or not.
        Parameter:
        ----------
        Return:
        ----------
        alpha_star: numeric
                    using the equation to get the alpha_star of each node
        """
        # We are on the node and we traverse down to this node through prune function.
        # We are on the node that is above its children. Only this one node
        num_rows = self.cur.execute('SELECT COUNT(*)' +
                                    'FROM' + self.table + ';')
        num_rows = self.cur.fetchone()[0]
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
        if self.right is None and self.left is None:
            if self.count_one > self.count_zero:
                prediction = 1
                return prediction
            else:
                prediction = 0
                return prediction
        if row[self.split_variable] < self.split_point:
            return self.left.query_node(row)
        if row[self.split_variable] >= self.split_point:
            return self.right.query_node(row)


def min_error_alpha(alpha):
    return alpha


def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)


# creating 'test_bigger test' and test_table' tables in sql database
# testing dataset, loading from sklean
cancer_whole = load_breast_cancer()
cancer = pd.DataFrame(cancer_whole.data)
cancer_result = pd.DataFrame(cancer_whole.target)
cancer = cancer.loc[:, 0:3]
cancer_test = pd.concat([cancer, cancer_result], axis = 1, ignore_index = True)
cancer_small_test = cancer_test.iloc[121:132, :]
cancer_small_test.columns = ['column_1', 'column_2', 'column_3', 'column_4', 'response']
cancer_bigger_test = cancer_test.iloc[0:50, :]
cancer_bigger_test.columns = ['column_1', 'column_2', 'column_3', 'column_4', 'response']

# Insert DataFrame records one by one from python to SQL.
# The following codes, transforming data from python to SQL is learnt through the link below.
# https://www.dataquest.io/blog/sql-insert-tutorial/

'''
for i,row in cancer_bigger_test.iterrows():
    sql = "INSERT INTO test_bigger_table (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
    cur.execute(sql, tuple(row))
    # the connection is not autocommitted by default, so we must commit to save our changes
    conn.commit()
# Insert DataFrame recrds one by one.
for i,row in cancer_small_test.iterrows():
    sql = "INSERT INTO test_table (" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
    cur.execute(sql, tuple(row))
    # the connection is not autocommitted by default, so we must commit to save our changes
    conn.commit()
'''

# Different verified cases

# SQLTree(cur, 'test_bigger_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, []).show()
# SQLnode = SQLTree(cur, 'test_bigger_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])
# min_error_alpha(0.7)
# SQLnode.prune(1)
# SQLnode.show()
# node_test2 = SQLTree(cur, 'test_data2', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])
# print(node_test2.past_split)
# test = SQLTree(cur, 'test_data4', ['column_1', 'column_2'], 'response', Gini_index, [])
# SQLTree(cur, 'test_table',['column_1', 'column_2', 'column_3', 'column_4'] , 'response', Gini_index, []).best_split(Gini_index)
# test_tree = SQLTree(cur, 'test_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])
# test_tree.is_valid()
