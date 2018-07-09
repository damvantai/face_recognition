import os
import sqlite3
import pandas as pd
from flask import Flask, g, render_template
from contextlib import closing
import pymysql as MySQLdb
import numpy as np
import pandas as pd
import json

colors = [
    "#80ff80", "#66c2ff", "#6666ff", "#b366ff",
    "#ffff66"]


# Create our little application
app = Flask(__name__)

# Configuration
app.config.update(dict(
    DATABASE = os.path.join("../data/people.db")
))

def connect_db():
    return sqlite3.connect(app.config['DATABASE'])
    
@app.route('/')
def chart():
    conn = MySQLdb.connect(host='localhost', 
                       port=3306, 
                       user='root', 
                       passwd='hanoi1994', 
                       db='people')

    cur = conn.cursor()

    # labels_hour = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    # old = ['0-20','20-30', '30-40', '40-50', '50-']
    num_people = np.zeros([7, 24], dtype=int)
    old_labels = np.array([0, 20, 30, 40, 50, 100])
    now = pd.datetime.now().hour

    # select print number people by hour and old
    for i in range(0, now+1):

        cur.execute("SELECT COUNT(DISTINCT history_come.name) FROM history_come, people_known WHERE (people_known.name = history_come.name) and hour(come_in) <= '%d' and (come_out is null or hour(come_out) >= '%d') and (people_known.gender = 1)" % (i, i))

        temp1 = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT history_come.name) FROM history_come, people_unknown WHERE (people_unknown.name = history_come.name) and hour(come_in) <= '%d' and (come_out is null or hour(come_out) >= '%d') and (people_unknown.gender = 1)" % (i, i))

        temp2 = cur.fetchone()[0]

        num_people[0][i] = temp1 + temp2

        cur.execute("SELECT COUNT(DISTINCT history_come.name) FROM history_come, people_known WHERE (people_known.name = history_come.name) and hour(come_in) <= '%d' and (come_out is null or hour(come_out) >= '%d') and (people_known.gender = 0)" % (i, i))

        temp1 = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT history_come.name) FROM history_come, people_unknown WHERE (people_unknown.name = history_come.name) and hour(come_in) <= '%d' and (come_out is null or hour(come_out) >= '%d') and (people_unknown.gender = 0)" % (i, i))

        temp2 = cur.fetchone()[0]

        num_people[1][i] = temp1 + temp2

        for j in range(len(old_labels)-2):

            cur.execute("SELECT COUNT(DISTINCT history_come.name) FROM history_come, people_known WHERE (people_known.name = history_come.name) and hour(come_in) <= '%d' and (come_out is null or hour(come_out) >= '%d') AND (people_known.old >= '%d') AND (people_known.old < '%d')" % (i, i, old_labels[j], old_labels[j+1]))

            temp1 = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT history_come.name) FROM history_come, people_unknown WHERE (people_unknown.name = history_come.name) and hour(come_in) <= '%d' and (come_out is null or hour(come_out) >= '%d') AND (people_unknown.old >= '%d') AND (people_unknown.old < '%d')" % (i, i, old_labels[j], old_labels[j+1]))

            temp2 = cur.fetchone()[0]

            num_people[j+2][i] = temp2 + temp1

    print(num_people)

    print(now)
    return render_template('LineChart.html', num_people = num_people, now = now)
    # return render_template('linegraph.html', labels=gender, values=values_gender, old_labels = old_labels, old_values=old_values, set = zip(old_values_un, old_labels_un, colors))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1926)

