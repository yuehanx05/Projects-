# Classification and Regression tree

> Links
[Classification and Regression tree: having data frame as dataset](https://github.com/yuehanx05/Projects-/blob/master/classification_tree.py)
[Classification and Regression tree: having sql database as dataset](https://github.com/yuehanx05/Projects-/blob/master/sql_tree.py)
[Classification and Regression tree test: data frame](https://github.com/yuehanx05/Projects-/blob/master/test_classification_tree.py)
[Classification and Regression tree test: sql database](https://github.com/yuehanx05/Projects-/blob/master/SQLtest.py)
[Classification tree benchmark code](https://github.com/yuehanx05/Projects-/blob/master/benchmark.py)
[Classification tree profile code](https://github.com/yuehanx05/Projects-/blob/master/Profile_code.py)
[Classification tree benchmark and profile results](https://github.com/yuehanx05/Projects-/blob/master/benchmark.txt)

> For classification_tree.py, using data frame to build and prune classification tree. Then, making random forest based on the tree that we built before. 

> The following functions are in the Class Node

### def __init__(self.dataset. Impurity_function, past_split): 
The init function needs users to input their desired dataset, which is a matrix, their desired impurity function, which can be Gini index, Bayes error, cross-entropy or any users defined function, and the past_split, which is an empty list [ ]. 
After having these inputs, the init function is going to help users to build a balanced tree with nodes and leaves. Each node contains the split point, split variable, its connection to the right child and left child, the number of zero and one of this node, and the probability of zero and one of this node. 

### def impurity(self, p_value, impurity_function):
The impurity function takes both p_value and impurity function in order to return the impurity center value, which is the parent node, for future calculations in impurity reduction function. 

### def impurity_reduction(self, p_value, left_dataset, right_dataset, impurity_function):
The impurity reduction function takes in p-value, left dataset, right dataset, and impurity function. In addition, the entered p-value should be between 0 and 1, otherwise, the function would throw an error. 
The maximum impurity value is calculated by using the impurity center value, which gets from the impurity function minus the probability of 1 in left dataset times the impurity value of the left dataset, which also gets from impurity function minus the probability of 1 in right dataset times the impurity value of the left dataset, which gets from impurity function. 
max_impurity_reduction = impurity_center - prob_left * impurity_left - prob_right * impurity_right

### def best_split(self, dataset, impurity_function):
The best split function takes dataset and impurity function as inputs and would return the best split variable and split point for each node. 
In order to find the best split variable and best split point for each node, this function would use the values returned by the impurity_reduction and the datasets returned by the get_data function. Each testing split variable and each split point would be corresponding with a impurity reduction value and for which split variable and split point who has the maximum impurity reduction would be the best split point and best split variable for that node. 

### def get_data(self, split_variable, split_point, dataset):
This get_data function inputs split variable, split point, and dataset and would return the sub dataset for left and right side nodes, which are also children nodes. 
The split variable and split point that input in this function is usually not the best split variable and split point but rather the testing split variable and split point that we examined in the best_split function. The way that we get the data is that we deduced the lines from the whole dataset based on the split point and split variable. 

### def prune(self, node, min_error_alpha, whole_dataset):
For prune function, it takes a node, the min_error alpha, and the whole dataset and it is going to return the modified tree after we prune. 
The min_error_alpha gets from the cross_validation function in the tree class which we will explain later. 
This function would also employ the G_T function to get the alpha star. If the alpha star is smaller than the min_error alpha, we would prune the tree. 

### def G_T(self, whole_dataset):
This G_T function takes the whole dataset and it would calculate the alpha start which would be used in the prune function that we mentioned above. 

### def query_node(self, row):
This query_node function takes every row of the data frame and traverse through the tree that we built and pruned before in order to get the prediction. 


> The following functions are in the Class Tree. 
### __init__(self, dataset, impurity_function):
This function takes the dataset and impurity function, which aims to build a tree and then call the prune function in this class. 
prune(self, dataset, impurity_function):
The prune function takes in the dataset and impurity function. This function would call the corss_validation function to get the minimum error alpha and then call the prune function in the Node class. 

### def cross_validation(self, whole_dataset, impurity_function):
The cross_Validation takes in the whole dataset and impurity function. This function would return the minimum error alpha which would be used in the prune function to compare with the alpha star to decide whether we need to prune a tree or not. 

### def query_node(self, dataset)
This query_node function takes the dataset and it aims for benchmark. Traverse through the tree and get the prediction. Each row of test_dataset will be compared with the split_variable's split point in order to decide which path, left or right, should go in order to get the predicted outcomes

> The following functions are in the Class tree_to_forest and it is a super function of Node Class. 
### __init__(self, whole_dataset, k, impurity_function):
This init function takes in the whole dataset, k, which is the number of variables that the users wish to select from all the covariates, and impurity function. This function aims to set the k as an attribute and inherit the functions in the Node class. 

### def best_split(self, dataset, impurity_function):
This best split takes in dataset and impurity function and it would return the best split point and best split variable for each node. This function is similar to the best_split function in the Node class, however, it doesn’t use the whole dataset nor all the variables to choose the best split point and best split variable but rather use the k number oof variables that users defined and then a random sample dataset. 

> The following function are included in the Class Forest 
### __init__(self, whole_dataset, k, n_tree, impurity_function):
This init function takes in whole dataset, k, which is the number of variables that users pick, n_tree, which is the number of trees that the users wish to have in the forest and the impurity function. This function would build a forest. 

### prediction(self, user_dataset):
This prediction function takes in the user’s dataset and it would return a list that has all the predictions that we calculated through our tree. 

> For sql_tree.py, it uses exactly the same algorithms as the classification_tree.py except some of the codes can only be applied to the SQL database rather than the data frame. One thing needs to notice is that sql_tree doesn’t have random forest nor the calculation of the minimum error alpha, which would be used in the prune function. Therefore, users need to identify their own minimum error alpha in order to prune the tree. 

> For both test_classification_tree.py and SQLtest.py, they are unit tests for the classification_tree.py and sql_tree.py.

> For benchmark.py, Profile_code.py, and benchmark.txt, they are measuring how fast the classification and regression tree code execute. They aim to find where the bottlenecks are and how to optimize them. 


