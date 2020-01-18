import profile
import classification_tree
import sql_tree
import pandas as pd
import random
import secret
import psycopg2
from sklearn import datasets
import pstats
from pstats import SortKey

conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)

cur = conn.cursor()
# 4 X 2 data set 
profile.run('''sql_tree.SQLTree(cur, 'test_data3', ['column_1'], 'response', Gini_index, [])''', 'profile')
p = pstats.Stats('profile')
p.strip_dirs().sort_stats('time').print_stats('profile')
p.print_stats()

# 4 X 4 data set 

profile.run('''sql_tree.SQLTree(cur, 'test_data', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])''', 'profile')
p = pstats.Stats('profile')
p.strip_dirs().sort_stats('time').print_stats('profile')
p.print_stats()

#13 X 5 data set

profile.run('''sql_tree.SQLTree(cur, 'test_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])''', 'profile')
p = pstats.Stats('profile')
p.strip_dirs().sort_stats('time').print_stats('profile')
p.print_stats()

#50 X 5 data set 

profile.run('''sql_tree.SQLTree(cur, 'test_bigger_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])''', 'profile')
p = pstats.Stats('profile')
p.strip_dirs().sort_stats('time').print_stats('profile')
p.print_stats()