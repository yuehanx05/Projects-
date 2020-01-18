#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:12:01 2019

@author: yuehanxiao
"""

import timeit
import sql_tree
import secret 
import psycopg2
import pandas as pd

# with 4 X 2 dataset 

def build_trees():
    setup_code= '''
import sql_tree
import pandas as pd
import secret
import psycopg2
conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

cur = conn.cursor()'''

    mycode = '''
def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)
Tnode = sql_tree.SQLTree(cur, 'test_data3', ['column_1'], 'response', Gini_index, [])'''

    times = timeit.repeat(setup = setup_code,
                      stmt = mycode,
                      repeat = 3,
                      number=10)
    print('Building trees time1: {}'.format(min(times)))

if __name__ == "__main__":
    build_trees()
    
# Building trees time1: 2.7732616900029825

# with 4 X 4 dataset 
def build_trees():
    setup_code= '''
import sql_tree
import pandas as pd
import secret
import psycopg2
conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

cur = conn.cursor()'''

    mycode = '''
def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)
Tnode = sql_tree.SQLTree(cur, 'test_data', ['column_1', 'column_2', 'column_3'], 'response', Gini_index, [])'''

    times = timeit.repeat(setup = setup_code,
                      stmt = mycode,
                      repeat = 3,
                      number=10)
    print('Building trees time1: {}'.format(min(times)))

if __name__ == "__main__":
    build_trees()
    
# Building trees time1: 4.682211845007259

# with 13 X 5 dataset 
def build_trees():
    setup_code= '''
import sql_tree
import pandas as pd
import secret
import psycopg2
conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

cur = conn.cursor()'''

    mycode = '''
def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)
Tnode = sql_tree.SQLTree(cur, 'test_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])'''

    times = timeit.repeat(setup = setup_code,
                      stmt = mycode,
                      repeat = 3,
                      number=10)
    print('Building trees time2: {}'.format(min(times)))

if __name__ == "__main__":
    build_trees()

# Building trees time: 16.638398843992036
    
# with 50 X 5 dataset 
    
def build_trees():
    setup_code= '''
import sql_tree
import pandas as pd
import secret
import psycopg2
conn = psycopg2.connect(host="sculptor.stat.cmu.edu", database=secret.DB_USER,
                        user=secret.DB_USER, password=secret.DB_PSSWD)

cur = conn.cursor()'''

    mycode = '''
def Gini_index(p):
    if p == 0 or p == 1:
        return 0
    else:
        return p*(1-p)
Tnode = sql_tree.SQLTree(cur, 'test_bigger_table', ['column_1', 'column_2', 'column_3', 'column_4'], 'response', Gini_index, [])'''

    times = timeit.repeat(setup = setup_code,
                      stmt = mycode,
                      repeat = 3,
                      number=10)
    print('Building trees time1: {}'.format(min(times)))

if __name__ == "__main__":
    build_trees()

# Building trees time1: 336.48836890098755
