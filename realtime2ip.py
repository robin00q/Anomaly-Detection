from IPython.display import display
import time
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import mglearn
import elasticsearch
import tensorflow as tf
from drawnow import drawnow
import json
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import csv
import ipaddr, ipaddress
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Variable
nowtime = np.array([])
tStamp_com1 = np.array([])
count_com1 = np.array([])
count2_com1 = np.array([])
tStamp_com2 = np.array([])
count_com2 = np.array([])
count2_com2 = np.array([])

targetname = ['NORMAL', 'DOS', 'PORTSCAN']
normaldata_X = genfromtxt('C:\\Python\\Python37\\dataset\\timenormal.csv', delimiter=',')
normaldata_y = np.array([0]*len(normaldata_X))

dos_X = genfromtxt('C:\\Python\\Python37\\dataset\\timedos.csv', delimiter=',')
dos_y = np.array([1]*len(dos_X))

portscan_X = genfromtxt('C:\\Python\\Python37\\dataset\\timeport.csv', delimiter=',')
portscan_y = np.array([2]*len(portscan_X))

dataset_X = np.empty(shape=[0, 2])
dataset_y = np.empty(shape=[0, 1])
dataset_X = np.append(normaldata_X, portscan_X, axis=0)
dataset_X = np.append(dataset_X, dos_X, axis=0)
dataset_y = np.append(normaldata_y, portscan_y, axis=0)
dataset_y = np.append(dataset_y, dos_y, axis=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(dataset_X, dataset_y)
tree = DecisionTreeClassifier(max_depth = 4, random_state=0)
tree.fit(dataset_X, dataset_y)

##################################################
#plt.scatter(normaldata_X[:, 0], normaldata_X[:, 1])
#plt.scatter(portscan_X[:, 0], portscan_X[:, 1])
#plt.scatter(dos_X[:, 0], dos_X[:, 1])
##################################################


windowSize = 7
hitsData_com1 = np.empty(shape=[0, 6])
hitsPort_com1 = np.empty(shape=[0, 1])
hitsIP_com1 = np.empty(shape=[0, 1])
hitsData_com2 = np.empty(shape=[0, 6])
hitsPort_com2 = np.empty(shape=[0, 1])
hitsIP_com2 = np.empty(shape=[0, 1])

f1 = open('C:\\Python\\Python37\\dataset\\savedata.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f1)


#Subplots
plt.ion()
#f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(10,10))
f = plt.figure(figsize=(16,12))
ax1 = f.add_subplot(5, 2, 1)
ax2 = f.add_subplot(5, 2, 3)
ax3 = f.add_subplot(5, 2, 5)
ax4 = f.add_subplot(5, 4, 13)
ax5 = f.add_subplot(5, 4, 14)
ax6 = f.add_subplot(5, 2, 2)
ax7 = f.add_subplot(5, 2, 4)
ax8 = f.add_subplot(5, 2, 6)
ax9 = f.add_subplot(5, 4, 15)
ax10 = f.add_subplot(5, 4, 16)
ax11 = f.add_subplot(5, 5, 21)
ax12 = f.add_subplot(5, 5, 25)
#f = plt.figure(figsize=(8,8))
#ax1 = f.add_subplot(4, 1, 1)
#ax2 = f.add_subplot(4, 1, 2)
#ax3 = f.add_subplot(4, 1, 3)
#ax4 = f.add_subplot(4, 2, 7)
#ax5 = f.add_subplot(4, 2, 8)




#f.tight_layout()
f.subplots_adjust(hspace=0.7)

#GET data from elasticsearch
def getElasticdata(Addr):
    es_client = elasticsearch.Elasticsearch("localhost:9200")
    docs = es_client.search(index = 'packetflow', 
                           body={
                               "query":{
                                   "bool":{
                                        "must":[
                                            {
                                                "match":{
                                                    "dst_ip":Addr
                                                }
                                            },
                                            {
                                                "range":{
                                                    "@timestamp":{
                                                        "gte":"now-5s",
                                                        "lt":"now"
                                                    
                                                    }
                                                }
                                            }
                                        
                                        ]
                                    }  
                                
                               }
                           },
                          scroll='5s',
                          size=10000)

    return docs

#PARSE data
def parseData(docs, Datahit):
    #init port, ip
    hData = Datahit
    hPort = np.array([]) # port
    hIP = np.array([]) # ip
    for res in docs['hits']['hits']:
        now = res['_source']['@timestamp'][0:10]+res['_source']['@timestamp'][11:19]
        timestamp = time.mktime(datetime.strptime(now, '%Y-%m-%d%H:%M:%S').timetuple())
        temp = np.array([[timestamp, 
                          int(ipaddress.IPv4Address(res['_source']['src_ip'])), 
                          int(ipaddress.IPv4Address(res['_source']['dst_ip'])), 
                          int(res['_source']['src_port']), 
                          int(res['_source']['dst_port']), 
                          int(res['_source']['pkt_len'])]])
        
        hData = np.append(hData, temp, axis=0)
        hPort = np.append(hPort, [int(temp[0][4])], axis=0)
        hIP = np.append(hIP, [int(temp[0][1])], axis=0)

    return hData, hPort, hIP
    
#Count data
def countData(num_docs, c1, c2, tS, i):
    appender = np.array([num_docs])
    c1 = np.append(c1, appender, axis=0)
    c2 = np.append(c2, appender, axis=0)
    tsAppender = np.array([i])
    tS = np.append(tS, tsAppender, axis=0)
    i = i+5

    return c1, c2, tS, i

#MovingAverage
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

#FindEvents
def findEvents(c1, c2, tS, mov, std, hPort, hIP, which, targetname):
    t = ""
    events_x= np.array([])
    events_y= np.array([])
    for ii in range(len(c1)):
        if c1[ii] > mov[ii]+3*std:
            events_x = np.append(events_x, [tS[ii]], axis=0)
            events_y = np.append(events_y, [c1[ii]], axis=0)
            
        if (ii == len(c1)-1) :
            if c1[ii] > mov[ii]+3*std:
                if(len(hPort) != 0):
                    if(hPort[0] != 0):
                        wr.writerow([np.std(hPort.flatten()), np.std(hIP.flatten())])
                        y_pred = knn.predict([[np.std(hPort.flatten()), np.std(hIP.flatten())]])
                        y_pred2 = tree.predict([[np.std(hPort.flatten()), np.std(hIP.flatten())]])
                        print(targetname[int(y_pred)])
                        print(targetname[int(y_pred2)])
                        t = targetname[int(y_pred2)]
                        print(c1[len(c1)-1])
                        if(int(y_pred) != 0):
                            if(len(c1) >= 2):
                                c2[len(c1)-1] = int(c2[len(c1)-2])

    return events_x, events_y, c2, t



i = 0
j = 0
for k in range(20):
    start_time = time.time()
    #GET data from elasticsearch
    docs1 = getElasticdata("192.168.56.102")
    docs2 = getElasticdata("192.168.56.103")

    #PARSE data
    hitsData_com1, hitsPort_com1, hitsIP_com1 = parseData(docs1, hitsData_com1)
    hitsData_com2, hitsPort_com2, hitsIP_com2 = parseData(docs2, hitsData_com2)

    #Get data length
    num_docs1 = len(docs1['hits']['hits'])
    num_docs2 = len(docs2['hits']['hits'])

    #Count data
    count_com1, count2_com1, tStamp_com1, i = countData(num_docs1, count_com1, count2_com1, tStamp_com1, i)
    count_com2, count2_com2, tStamp_com2, j = countData(num_docs2, count_com2, count2_com2, tStamp_com2, j)
    
    #MovingAverage
    MOV_com1 = movingaverage(count2_com1, windowSize)
    MOV_com2 = movingaverage(count2_com2, windowSize)
    if i == 0: 
        MOV_com1[0] = count_com1[0]
        MOV_com2[0] = count_com2[0]
    
    #StandardDeviation
    STD_com1 = np.std(MOV_com1)
    STD_com2 = np.std(MOV_com2)


    #FindEvents
    events_x_com1, events_y_com1, count2_com1, text_com1 = findEvents(count_com1, count2_com1, tStamp_com1, MOV_com1, STD_com1, hitsPort_com1, hitsIP_com1, 111, targetname)
    events_x_com2, events_y_com2, count2_com2, text_com2 = findEvents(count_com2, count2_com2, tStamp_com2, MOV_com2, STD_com2, hitsPort_com2, hitsIP_com2, 222, targetname)
    
        
    #DRAW data
    flatPort_com1 = [hitsPort_com1.flatten()]
    flatIP_com1 = [hitsIP_com1.flatten()]
    flatPort_com2 = [hitsPort_com2.flatten()]
    flatIP_com2 = [hitsIP_com2.flatten()]

    
    ax1.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax9.clear()
    ax10.clear()
    ax11.clear()
    ax12.clear()

    
    ax1.plot(tStamp_com1[0:len(count_com1)], count_com1[0:len(count_com1)], c = 'r')
    ax1.plot(tStamp_com1[0:len(count_com1)], MOV_com1[0:len(count_com1)], c = 'b')    
    ax1.plot(tStamp_com1[0:len(count_com1)], 3*STD_com1+MOV_com1[0:len(count_com1)], c = 'g')
    ax1.scatter(events_x_com1, events_y_com1, c = 'r')
    
    ax6.plot(tStamp_com2[0:len(count_com2)], count_com2[0:len(count_com2)], c = 'r')
    ax6.plot(tStamp_com2[0:len(count_com2)], MOV_com2[0:len(count_com2)], c = 'b')    
    ax6.plot(tStamp_com2[0:len(count_com2)], 3*STD_com2+MOV_com2[0:len(count_com2)], c = 'g')
    ax6.scatter(events_x_com2, events_y_com2, c = 'r')
        
    ax2.scatter(hitsData_com1[:, 0], hitsData_com1[:, 4], c = 'g')
    ax3.scatter(hitsData_com1[:, 0], hitsData_com1[:, 1], c = 'r')

    ax7.scatter(hitsData_com2[:, 0], hitsData_com2[:, 4], c = 'g')
    ax8.scatter(hitsData_com2[:, 0], hitsData_com2[:, 1], c = 'r')

    
   
    if(len(hitsPort_com1) != 0):
        ax4.set_xlim([0, 2])
        ax4.set_ylim([0, 5000])
    ax4.boxplot(flatPort_com1)

    if(len(hitsPort_com2) != 0):
        ax9.set_xlim([0, 2])
        ax9.set_ylim([0, 5000])
    ax9.boxplot(flatPort_com2)

    if(len(hitsIP_com1) != 0):
        ax5.set_xlim([0, 2])
        ax5.set_ylim([-100000000000, 300000000000])
    ax5.boxplot(flatIP_com1)
    
    if(len(hitsIP_com2) != 0):
        ax10.set_xlim([0, 2])
        ax10.set_ylim([-100000000000, 300000000000])
    ax10.boxplot(flatIP_com2)

    ax1.title.set_text('IP 192.168.56.102 5-Sec Packet Count')
    ax2.title.set_text('UnixTime / Dst_Port distribution')
    ax3.title.set_text('UnixTime / Src_IP distribution')
    ax4.title.set_text('Dst_Port BoxPlot')
    ax5.title.set_text('Src_IP BoxPlot')
    ax6.title.set_text('IP 192.168.56.103 5-Sec Packet Count')
    ax7.title.set_text('UnixTime / Dst_Port distribution')
    ax8.title.set_text('UnixTime / Src_IP distribution')
    ax9.title.set_text('Dst_Port BoxPlot')
    ax10.title.set_text('Src_IP BoxPlot')
    ax11.text(0.1, 0.4, text_com1, fontsize=20)
    ax11.set_xticks([], [])
    ax11.set_yticks([], [])
    ax12.text(0.1, 0.4, text_com2, fontsize=20)
    ax12.set_xticks([], [])
    ax12.set_yticks([], [])
    
    plt.draw()
    plt.pause(0.01)
    time.sleep(5.0-(time.time() - start_time))
    

f1.close()



        


