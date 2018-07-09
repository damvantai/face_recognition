
# coding: utf-8

# In[1]:


import os
import sqlite3
import pandas as pd
from flask import Flask, g, render_template
from contextlib import closing
import pymysql as MySQLdb
import numpy as np
# In[66]:


colors = [
    "#80ff80", "#66c2ff", "#6666ff", "#b366ff",
    "#ffff66"]


# In[97]:


# Create our little application
app = Flask(__name__)


# In[98]:


# Configuration
app.config.update(dict(
    DATABASE = os.path.join("../data/people.db")
))


# In[99]:


def connect_db():
    return sqlite3.connect(app.config['DATABASE'])


# In[100]:


# def init_db():
#     with closing(connect_db()) as db:
#         with app.open_resource("")


# In[101]:


# @app.route('/')
# def chart():
#     con = sqlite3.connect("../data/people.db")
#     df = pd.read_sql_query("select * from people_inroom", con)
#     df1 = pd.read_sql_query("select * from people_unknown", con)
    
#     print(df.to_json())
#     con.close()
    
#     name_list = df['name'].values.tolist()
#     gender_list = df['gender'].values.tolist()
#     old_list = df['old'].values.tolist()
#     total_in_room = len(name_list)
#     total_male_in_room = gender_list.count(1)
#     total_female_in_room = total_in_room - total_male_in_room
    
#     gender = ["Male", "Female"]
#     values_gender = [total_male_in_room, total_female_in_room]
# #     all_values_old = 
#     num_0 = len([i for i in old_list if (i < 20)])
#     num_20_30 = len([i for i in old_list if (i >= 20 and i < 30)])
#     num_30_40 = len([i for i in old_list if (i >= 30 and i < 40)])
#     num_40_50 = len([i for i in old_list if (i >= 40 and i < 50)])
#     num_50 = len([i for i in old_list if (i >= 50)])
#     old_labels = ["0-20", "20-30", "30-40", "40-50", "50-"]
#     old_values = [num_0, num_20_30, num_30_40, num_40_50, num_50]
# #     print(old_list)
# #     print(values_gender)
# #     print(name_list)


#     name_list_un = df1['name'].values.tolist()
#     gender_list_un = df1['gender'].values.tolist()
#     old_list_un = df1['old'].values.tolist()
#     total_in_room_un = len(name_list_un)
#     total_male_in_room_un = gender_list_un.count(1)
#     total_female_in_room_un = total_in_room_un - total_male_in_room_un

#     num_0_un = len([i for i in old_list_un if (i < 20)])
#     num_20_30_un = len([i for i in old_list_un if (i >= 20 and i < 30)])
#     num_30_40_un = len([i for i in old_list_un if (i >= 30 and i < 40)])
#     num_40_50_un = len([i for i in old_list_un if (i >= 40 and i < 50)])
#     num_50_un = len([i for i in old_list_un if (i >= 50)])
#     old_labels_un = ["0-20", "20-30", "30-40", "40-50", "50-"]
#     old_values_un = [num_0_un, num_20_30_un, num_30_40_un, num_40_50_un, num_50_un]

#     # return render_template('linegraph.html', labels=gender, values=values_gender, old_labels = old_labels, old_values=old_values, old_labels_un=old_labels_un, old_values_un=old_values_un, colors=colors)
#     return render_template('linegraph.html', labels=gender, values=values_gender, old_labels = old_labels, old_values=old_values, set = zip(old_values_un, old_labels_un, colors))
    
@app.route('/')
def chart():
    conn = MySQLdb.connect(host='localhost', 
                       port=3306, 
                       user='root', 
                       passwd='hanoi1994', 
                       db='people')

    cur = conn.cursor()

    labels_old = ["0-20", "20-30", "30-40", "40-50", "50-"]

    # in room
    num_people = np.zeros([2, 5], dtype=int)
    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 0 and old < 20")
    num_people[0][0] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 1 and old < 20")
    num_people[1][0] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 0 and old >= 20 and old < 30")
    num_people[0][1] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 1 and old >= 20 and old < 30")
    num_people[1][1] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 0 and old >= 30 and old < 40")
    num_people[0][2] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 1 and old >= 30 and old < 40")
    num_people[1][2] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 0 and old >= 40 and old < 50")
    num_people[0][3] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 1 and old >= 40 and old < 50")
    num_people[1][3] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 0 and old >= 50")
    num_people[0][4] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 1 and old >= 50")
    num_people[1][4] = cur.fetchone()[0]


    # people unknown
    num_people_unknown = np.zeros([2, 5], dtype=int)
    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 0 and old < 20")
    num_people_unknown[0][0] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 1 and old < 20")
    num_people_unknown[1][0] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 0 and old >= 20 and old < 30")
    num_people_unknown[0][1] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 1 and old >= 20 and old < 30")
    num_people_unknown[1][1] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 0 and old >= 30 and old < 40")
    num_people_unknown[0][2] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 1 and old >= 30 and old < 40")
    num_people_unknown[1][2] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 0 and old >= 40 and old < 50")
    num_people_unknown[0][3] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 1 and old >= 40 and old < 50")
    num_people_unknown[1][3] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 0 and old >= 50")
    num_people_unknown[0][4] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM people.people_unknown where gender = 1 and old >= 50")
    num_people_unknown[1][4] = cur.fetchone()[0]

    print(num_people)
    print(type(num_people[0][1]))

    print(num_people_unknown)
    print(type(num_people_unknown[0][1]))

    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 1")
    total_male_in_room = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM people.people_inroom where gender = 0")
    total_female_in_room = cur.fetchone()[0]
    cur.close()
    conn.close()

    return render_template('BarChart.html', num_people = num_people, num_people_unknown = num_people_unknown, total_male_in_room=total_male_in_room, total_female_in_room=total_female_in_room)
    # return render_template('linegraph.html', labels=gender, values=values_gender, old_labels = old_labels, old_values=old_values, set = zip(old_values_un, old_labels_un, colors))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1925)

