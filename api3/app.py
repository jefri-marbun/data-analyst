from re import X
from tkinter import Y
import requests
from flask import Flask, render_template, jsonify
import json
import operator
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np



app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')



@app.route('/list-transaksi', methods=['GET'])
def transaksi():
    req = requests.get('http://apilp.ptppa.com/platform/carpool/apicpl_viewrequest.php?aimei=31A04E99-3C3B-48C3-A706-5B4576C4CDBC&alogin=rhera&atoken=1HTNHR$BFINCiZ/5zJdNSxcJ1Igh/&dtgl_pakai=2019-12-11&dtgl_pakai2=2019-12-11')

    data = json.loads(req.content)

    return render_template('transaksi.html', data=data['result'])



def data_kinerja(data, key):
    return data.get(key, None)

@app.route('/kinerja-driver', methods=['GET'])
def kinerja():
    res = requests.get('http://apilp.ptppa.com/platform/carpool/apicpl_viewrequest.php?aimei=31A04E99-3C3B-48C3-A706-5B4576C4CDBC&alogin=rhera&atoken=1HTNHR$BFINCiZ/5zJdNSxcJ1Igh/&dtgl_pakai=2019-12-11&dtgl_pakai2=2019-12-11') 
    data_json = res.json()
    done_status_pegawai = {}

    for i in data_json['result']:
        if i['astatus'] == 'Done':
            if data_kinerja(done_status_pegawai, i['anama_driver']):
                done_status_pegawai[i['anama_driver']] += 1
            else:
                done_status_pegawai[i['anama_driver']] = 1


    x = done_status_pegawai
    data = dict(sorted(x.items()))

    return render_template('kinerja.html', data=data)



@app.route('/graph-kinerja', methods=['GET'])
def graph():
    res = requests.get('http://apilp.ptppa.com/platform/carpool/apicpl_viewrequest.php?aimei=31A04E99-3C3B-48C3-A706-5B4576C4CDBC&alogin=rhera&atoken=1HTNHR$BFINCiZ/5zJdNSxcJ1Igh/&dtgl_pakai=2019-12-11&dtgl_pakai2=2019-12-11') 
    data_json = res.json()
    done_status_pegawai = {}

    for i in data_json['result']:
        if i['astatus'] == 'Done':
            if data_kinerja(done_status_pegawai, i['anama_driver']):
                done_status_pegawai[i['anama_driver']] += 1
            else:
                done_status_pegawai[i['anama_driver']] = 1


    x = done_status_pegawai

    labels = []
    values = []
    for k, v in x.items():
        labels.append((k))
        values.append((v))
    

    return render_template('graph.html', nama=labels, values=values)



@app.route('/read-csv')
def readcsv():
    df = pd.read_csv('file/response2.csv')   
    return df.to_html()


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

    

@app.route('/acc')
def acc():
    df = pd.read_csv('file/response2.csv', header=0, usecols=['arating', 'crelease'])   
    df.head()

    # define x & y
    feature_col = ['arating', 'crelease']
    x = df[feature_col]
    y = df['arating']

    # split x & y  into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # train a logistic regression model on training set
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    # make class prediction for the testing set
    y_pred_class = logreg.predict(x_test)

    # calculate accuracy
    acc = metrics.accuracy_score(y_test, y_pred_class)

    # return str(acc)
    return render_template('data-acc.html', data=str(acc))
    

if __name__ == "__main__":
    app.run(debug=True)