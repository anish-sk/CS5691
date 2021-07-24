#!/usr/bin/env python
# coding: utf-8

# In[61]:


from catboost import Pool, CatBoostClassifier

import csv

from datetime import datetime

from geopy.distance import distance

from lightgbm import LGBMClassifier

import numpy as np
import os 
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit

from tqdm import tqdm

dire_data = "../../data"


# In[2]:


train = pd.read_csv(dire_data + "/train.csv")
test = pd.read_csv(dire_data + "/test.csv")


# In[3]:


train_bikers_set = np.array(train["biker_id"].drop_duplicates())
test_bikers_set = np.array(test["biker_id"].drop_duplicates())
req_bikers_set = np.union1d(train_bikers_set, test_bikers_set)
tours_set = np.array(pd.merge(test["tour_id"].drop_duplicates(), 
                              train["tour_id"].drop_duplicates(), how = 'outer'))
tours_set = tours_set.reshape((tours_set.shape[0],))


# In[4]:


biker = pd.read_csv(dire_data + "/bikers.csv")
biker = biker[biker.biker_id.isin(req_bikers_set)]
lldf = pd.read_csv(dire_data + "/locations.csv")
lldf = lldf[lldf.biker_id.isin(req_bikers_set)]
biker['latitude']  = lldf['latitude']
biker['longitude'] = lldf['longitude']
biker.to_csv("bikers_useful.csv", index = False)
biker=None
lldf=None


# In[5]:


biker_net = pd.read_csv(dire_data + "/bikers_network.csv")
biker_net = biker_net[biker_net.biker_id.isin(req_bikers_set)]
biker_net.to_csv("bikers_network_useful.csv", index = False)
biker_net=None


# In[6]:


tour_convoy = pd.read_csv(dire_data + "/tour_convoy.csv")
tour_convoy = tour_convoy[tour_convoy.tour_id.isin(tours_set)]
tour_convoy.to_csv("tour_convoy_useful.csv", index = False)
tour_convoy=None


# In[7]:


tours = pd.read_csv(dire_data + "/tours.csv")
tours = tours[tours.tour_id.isin(tours_set)]
tours.to_csv("tours_useful.csv", index = False)
tours=None


# In[8]:


biker = { key: row for key, *row in csv.reader(open("bikers_useful.csv", 'r'))}
print(biker.pop('biker_id'))
biker_net = { key: row for key, *row in csv.reader(open("bikers_network_useful.csv", 'r'))}
print(biker_net.pop('biker_id'))
tours = { key: row for key, *row in csv.reader(open("tours_useful.csv", 'r'))}
print(tours.pop('tour_id'))
tour_convoy = { key: row for key, *row in csv.reader(open("tour_convoy_useful.csv", 'r'))}
print(tour_convoy.pop('tour_id'))


# In[9]:


tour_convoy_full = { key: row for key, *row in csv.reader(open(dire_data + "/tour_convoy.csv", 'r'))}


# In[10]:


events_attended = {}
events_maybe = {}
events_invited = {}
events_notgoing = {}
for tid, row in tour_convoy_full.items():
    people_attended = row[0].split()
    people_maybe = row[1].split()
    people_invited = row[2].split()
    people_notgoing = row[3].split()
    for bid in people_attended:
        events_attended[bid] = events_attended.get(bid, 0) + 1
    for bid in people_maybe:
        events_maybe[bid] = events_maybe.get(bid, 0) + 1
    for bid in people_invited:
        events_invited[bid] = events_invited.get(bid, 0) + 1
    for bid in people_notgoing:
        events_notgoing[bid] = events_notgoing.get(bid, 0) + 1
        
events_attended_friends = {}
events_maybe_friends = {}
events_invited_friends = {}
events_notgoing_friends = {}
no_friends = {}
for bid, row in biker_net.items():
    friends = row[0].split()
    no_friends[bid] = len(friends)
    for friend in friends:
        events_attended_friends[bid] = events_attended_friends.get(bid, 0) + events_attended.get(friend, 0)
        events_maybe_friends[bid] = events_maybe_friends.get(bid, 0) + events_maybe.get(friend, 0)
        events_invited_friends[bid] = events_invited_friends.get(bid, 0) + events_invited.get(friend, 0)
        events_notgoing_friends[bid] = events_notgoing_friends.get(bid, 0) + events_notgoing.get(friend, 0)
        
#events_attended_friends


# In[11]:


events_attended_bt = {}
events_maybe_bt = {}
events_invited_bt = {}
events_notgoing_bt = {}
for tid, row in tour_convoy_full.items():
    people_attended = row[0].split()
    people_maybe = row[1].split()
    people_invited = row[2].split()
    people_notgoing = row[3].split()
    for bid in people_attended:
        events_attended_bt[bid] = events_attended_bt.get(bid,[]) + [tid]
    for bid in people_maybe:
        events_maybe_bt[bid] = events_maybe_bt.get(bid,[]) + [tid]
    for bid in people_invited:
        events_invited_bt[bid] = events_invited_bt.get(bid,[]) + [tid]
    for bid in people_notgoing:
        events_notgoing_bt[bid] = events_notgoing_bt.get(bid,[])+ [tid]
        
events_attended_friends_bt = {}
events_maybe_friends_bt = {}
events_invited_friends_bt = {}
events_notgoing_friends_bt = {}
no_friends = {}
for bid, row in tqdm(biker_net.items()):
    friends = row[0].split()
    no_friends[bid] = len(friends)
    for friend in friends:
        for tid in events_attended_bt.get(friend,[]):            
            events_attended_friends_bt[(bid,tid)] = events_attended_friends_bt.get((bid,tid), 0) + 1
        for tid in events_maybe_bt.get(friend,[]):
            events_maybe_friends_bt[(bid,tid)] = events_maybe_friends_bt.get((bid,tid), 0) + 1
        for tid in events_invited_bt.get(friend,[]):
            events_invited_friends_bt[(bid,tid)] = events_invited_friends_bt.get((bid,tid), 0) + 1
        for tid in events_notgoing_bt.get(friend,[]):
            events_notgoing_friends_bt[(bid,tid)] = events_notgoing_friends_bt.get((bid,tid), 0) + 1


# In[12]:


tours_full = pd.read_csv(dire_data + "/tours.csv").values
len_tf = tours_full.shape[0]
np.random.seed(42)
tours_full = tours_full[np.random.choice(len_tf, len_tf//10)]


# In[13]:


word_count = []
for row in tqdm(tours_full):
    w = []
    for i in range(9, 110):
        w.append(int(row[i]))
    word_count.append(w)
    
word_count = np.array(word_count)
 
kmeans = KMeans(n_clusters=30, random_state=0).fit(word_count)


# In[14]:


#Selecting top countries for bikers and putting rest as others
countries = {}
for bid, row in biker.items():
    countries[row[1]] = countries.get(row[1], 0) + 1
    
countries_list = list(countries.keys())
countries_list.sort(key = lambda x : countries[x], reverse = True)
top_10_countries = countries_list[:10]
country_val = {}
for i, country in enumerate(top_10_countries):
    country_val[country] = i
    
#Selecting top countries for tours and putting rest as others
tour_countries = {}
for bid, row in tours.items():
    tour_countries[row[5]] = tour_countries.get(row[5], 0) + 1
    
tour_countries_list = list(tour_countries.keys())
tour_countries_list.sort(key = lambda x : tour_countries[x], reverse = True)
tour_top_10_countries = tour_countries_list[:10]
tour_country_val = {}
for i, country in enumerate(tour_top_10_countries):
    tour_country_val[country] = i    
    
#Selecting top cities for tours and putting rest as others
tour_cities = {}
for bid, row in tours.items():
    tour_cities[row[2]] = tour_cities.get(row[2], 0) + 1
    
tour_cities_list = list(tour_cities.keys())
tour_cities_list.sort(key = lambda x : tour_cities[x], reverse = True)
tour_top_10_cities = tour_cities_list[:10]
tour_city_val = {}
for i, city in enumerate(tour_top_10_cities):
    tour_city_val[city] = i    

#Selecting top languages for bikers and putting rest as others
langs = {}
for bid, row in biker.items():
    langs[row[1]] = langs.get(row[0], 0) + 1
    
langs_list = list(langs.keys())
langs_list.sort(key = lambda x : langs[x], reverse = True)
top_10_langs = langs_list[:10]
lang_val = {}
for i, lang in enumerate(top_10_langs):
    lang_val[lang] = i


# In[15]:


lldf = pd.read_csv(dire_data + "/locations.csv")
bdf = pd.read_csv(dire_data + "/bikers.csv")
ladf = pd.DataFrame()
ladf['time_zone'] = bdf['time_zone']
ladf['latitude'] = lldf['latitude']
lodf = pd.DataFrame()
lodf['time_zone'] = bdf['time_zone']
lodf['longitude'] = lldf['longitude']
tz_mod_lat  = ladf.groupby('time_zone').apply(lambda x: x['latitude'].value_counts().idxmax())
tz_mod_long = lodf.groupby('time_zone').apply(lambda x: x['longitude'].value_counts().idxmax())

for bid, row in biker.items():
    if row[7] == '' and row[6] in tz_mod_lat:
        biker[bid][7] = tz_mod_lat[row[6]]
    if row[8] == '' and row[6] in tz_mod_long:
        biker[bid][8] = tz_mod_long[row[6]]     


# In[19]:


top_10_loc = { 
                 'ID': [-2.4833826, 117.8902853],
                 'US': [39.7837304, -100.4458825],
                 'GB': [54.7023545, -3.2765753],
                 'LA': [20.0171109, 103.378253],
                 'ES': [39.3262345, -4.8380649],
                 'FR': [46.603354, 1.8883335],
                 'GE': [32.3293809, -83.1137366],
                 'AR': [-34.9964963, -64.9672817],
                 'CN': [35.000074, 104.999927],
                 'RU': [64.6863136, 97.7453061]
}
for bid, row in biker.items():
    if row[7] == '' and row[1] in top_10_loc:
        biker[bid][7] = top_10_loc[row[1]][0]
    if row[8] == '' and row[1] in top_10_loc:
        biker[bid][8] = top_10_loc[row[1]][1]     


# In[20]:


biker_full = { key: row for key, *row in csv.reader(open(dire_data + "/bikers.csv", 'r'))}
print(biker_full.pop('biker_id'))


# In[21]:


biker_full_lat = {}
biker_full_long = {}
ladf['biker_id'] =bdf['biker_id']
lodf['biker_id'] =bdf['biker_id']

for index, row in ladf.iterrows():
    bid = row['biker_id']
#     if not pd.isna(row['latitude']):
#         biker_full_lat[bid] = row['latitude']
    if pd.isna(row['latitude']) and (row['time_zone'] in tz_mod_lat):
        biker_full_lat[bid] = tz_mod_lat[row['time_zone']]
    elif biker_full[bid][1] in top_10_loc:
        biker_full_lat[bid] = top_10_loc[biker_full[bid][1]][0]

for index, row in lodf.iterrows():
    bid = row['biker_id']
#     if not pd.isna(row['longitude']):
#         biker_full_long[bid] = row['longitude']
    if pd.isna(row['longitude']) and (row['time_zone'] in tz_mod_long):
        biker_full_long[bid] = tz_mod_long[row['time_zone']]
    elif biker_full[bid][1] in top_10_loc:
        biker_full_long[bid] = top_10_loc[biker_full[bid][1]][1]


# In[22]:


cc_set = {
            'LA': [20.0171109, 103.378253],
            'RU': [64.6863136, 97.7453061],
            'ID': [-2.4833826, 117.8902853],
            'LT': [55.3500003, 23.7499997],
            'VN': [13.2904027, 108.4265113],
            'CN': [35.000074, 104.999927],
            'BR': [-10.3333333, -53.2],
            'ES': [39.3262345, -4.8380649],
            'FR': [46.603354, 1.8883335],
            'UD': [39.7837304, -100.4458825],
            'US': [39.7837304, -100.4458825],
            'GE': [32.3293809, -83.1137366],
            'GB': [54.7023545, -3.2765753],
            'TW': [23.9739374, 120.9820179],
            'CA': [61.0666922, -107.9917071]
        }

com_coun = {**cc_set, **top_10_loc}

def get_nan_count():
    nan_count = 0
    for tid, row in tours.items():
        if row[6] == '' or row[7] == '':
            nan_count+=1
    return nan_count

def get_friend_org(org):
    for bid, row in biker_net.items():
        friends = row[0].split()
        if org in friends:
            return bid
    return ''


def repl_missing_tour_loc1():
    bikerlalist = set(biker_full_lat)
    bikerlolist = set(biker_full_long)
    for tid, row in tours.items():
        if row[6] == '':
            att_list = tour_convoy_full[tid][0].split()
            att_list = bikerlalist.intersection(att_list)
            inv_list = tour_convoy_full[tid][2].split()
            inv_list = bikerlalist.intersection(inv_list)
            bc_list = []
            for bi in att_list:
                bc_list.append(biker_full_lat[bi])
            for bi in inv_list:
                bc_list.append(biker_full_lat[bi])
            if len(bc_list) > 0:
                most_common_la = max(set(bc_list), key=bc_list.count)
                tours[tid][6] = most_common_la
        if row[7] == '':
            att_list = tour_convoy_full[tid][0].split()
            att_list = bikerlolist.intersection(att_list)
            inv_list = tour_convoy_full[tid][2].split()
            inv_list = bikerlolist.intersection(inv_list)
            bc_list = []
            for bi in att_list:
                bc_list.append(biker_full_long[bi])
            for bi in inv_list:
                bc_list.append(biker_full_long[bi])
            if len(bc_list) > 0:
                most_common_lo = max(set(bc_list), key=bc_list.count)
                tours[tid][7] = most_common_lo

def repl_missing_tour_loc2():
    for tid, row in tqdm(tours.items()):
        fren = ''
        if row[6] == '' or row[7] == '':
            fren = get_friend_org(row[0])
        if row[6] == '':
            if fren not in biker_full_lat:
                continue
            tours[tid][6] = biker_full_lat[fren]
        if row[7] == '':
            if fren not in biker_full_long:
                continue
            tours[tid][7] = biker_full_long[fren]


repl_missing_tour_loc1()

repl_missing_tour_loc2()

latlist = []
longlist = []
for tid, row in tours.items():
    if row[6] != '':
        latlist.append((float(row[6])))
    if row[7] != '':
        longlist.append((float(row[7])))
        
latmed = np.median(np.array(latlist))
longmed = np.median(np.array(longlist))

for tid, row in tours.items():
    if row[6] == '':
        tours[tid][6] = latmed
    if row[7] == '':
        tours[tid][7] = longmed


# In[23]:


#Number of tours each person has in train/test set
train_count = {}
test_count = {}
like_count = {}
dislike_count = {}
like_tour_count = {}
dislike_tour_count = {}
train_tour_count = {}
test_tour_count = {}

for index, row in train.iterrows():
    train_count[row["biker_id"]]   = train_count.get(row["biker_id"],0)+1
    like_count[row["biker_id"]]    = like_count.get(row["biker_id"],0)+row["like"]
    dislike_count[row["biker_id"]] = dislike_count.get(row["biker_id"],0)+row["dislike"]
    train_tour_count[row["tour_id"]]   = train_tour_count.get(row["tour_id"],0)+1
    like_tour_count[row["tour_id"]]    = like_tour_count.get(row["tour_id"],0)+row["like"]
    dislike_tour_count[row["tour_id"]] = dislike_tour_count.get(row["tour_id"],0)+row["dislike"]
    
    
for index, row in test.iterrows():
    test_count[row["biker_id"]] = test_count.get(row["biker_id"],0)+1
    test_tour_count[row["tour_id"]]   = test_tour_count.get(row["tour_id"],0)+1


# In[24]:


events_attended_dist = {}
events_maybe_dist = {}
events_invited_dist = {}
events_notgoing_dist = {}
for tid, row in tour_convoy_full.items():
    if tid == 'tour_id' or tid not in tours:
        continue
    people_attended = row[0].split()
    people_maybe = row[1].split()
    people_invited = row[2].split()
    people_notgoing = row[3].split()
    for bid in people_attended:
        events_attended_dist[bid] = events_attended_dist.get(bid, np.zeros(101)) + np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    for bid in people_maybe:
        events_maybe_dist[bid] = events_maybe_dist.get(bid, np.zeros(101)) + np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    for bid in people_invited:
        events_invited_dist[bid] = events_invited_dist.get(bid, np.zeros(101)) + np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    for bid in people_notgoing:
        events_notgoing_dist[bid] = events_notgoing_dist.get(bid, np.zeros(101)) + np.array(list(map(lambda x: int(x), tours[tid][8:109])))
        
events_attended_friends_dist = {}
events_maybe_friends_dist = {}
events_invited_friends_dist = {}
events_notgoing_friends_dist = {}
for bid, row in biker_net.items():
    friends = row[0].split()
    for friend in friends:
        events_attended_friends_dist[bid] = events_attended_friends_dist.get(bid, np.zeros(101)) + events_attended_dist.get(friend, np.zeros(101))
        events_maybe_friends_dist[bid] = events_maybe_friends_dist.get(bid, np.zeros(101)) + events_maybe_dist.get(friend, np.zeros(101))
        events_invited_friends_dist[bid] = events_invited_friends_dist.get(bid, np.zeros(101)) + events_invited_dist.get(friend, np.zeros(101))
        events_notgoing_friends_dist[bid] = events_notgoing_friends_dist.get(bid, np.zeros(101)) + events_notgoing_dist.get(friend, np.zeros(101))
        
for bid in events_attended_dist:
    events_attended_dist[bid] /= (events_attended[bid] + 1)
    
for bid in events_maybe_dist:
    events_maybe_dist[bid] /= (events_maybe[bid] + 1)
    
for bid in events_invited_dist:
    events_invited_dist[bid] /= (events_invited[bid] + 1)
    
for bid in events_notgoing_dist:
    events_notgoing_dist[bid] /= (events_notgoing[bid] + 1)
    
for bid in events_attended_friends_dist:
    events_attended_friends_dist[bid] /= (events_attended_friends[bid] + 1)
    
for bid in events_maybe_friends_dist:
    events_maybe_friends_dist[bid] /= (events_maybe_friends[bid] + 1)
    
for bid in events_invited_friends_dist:
    events_invited_friends_dist[bid] /= (events_invited_friends[bid] + 1)
    
for bid in events_notgoing_friends_dist:
    events_notgoing_friends_dist[bid] /= (events_notgoing_friends[bid] + 1)
    


# In[26]:


lang_attended = {}
lang_maybe = {}
lang_invited = {}
lang_notgoing = {}
lat_long = []
for bid, row in biker.items():
    if row[7] == '':
        row[7] = '0'
    if row[8] == '':
        row[8] = '0'
    lat_long.append([row[7],row[8]])
    
lat_long=np.array(lat_long)
kmeans_ll = KMeans(n_clusters=30, random_state=0).fit(lat_long)
kmeans_ll1 = KMeans(n_clusters=15, random_state=0).fit(lat_long)

llclus_attended = {}
llclus1_attended = {}
llclus_maybe = {}
llclus1_maybe = {}
llclus_invited = {}
llclus1_invited = {}
llclus_notgoing = {}
llclus1_notgoing = {}
for tid, row in tour_convoy_full.items():
    if tid == 'tour_id' or tid not in tours:
        continue
    people_attended = row[0].split()
    people_maybe    = row[1].split()
    people_invited  = row[2].split()
    people_notgoing = row[3].split()
    for bid in people_attended:
        if bid not in biker:
            continue
        lang_attended[(biker[bid][0], tid)] = lang_attended.get((biker[bid][0], tid), 0) + 1
        clu = kmeans_ll.predict(np.array([biker[bid][7:9]]))[0]
        clu1 = kmeans_ll1.predict(np.array([biker[bid][7:9]]))[0]
        llclus_attended[(clu,tid)] = llclus_attended.get((clu,tid),0) + 1
        llclus1_attended[clu1] = llclus1_attended.get(clu1,0) + 1
    for bid in people_maybe:
        if bid not in biker:
            continue
        lang_maybe[(biker[bid][0], tid)] = lang_maybe.get((biker[bid][0], tid), 0) + 1
        clu = kmeans_ll.predict(np.array([biker[bid][7:9]]))[0]
        clu1 = kmeans_ll1.predict(np.array([biker[bid][7:9]]))[0]
        llclus_maybe[(clu, tid)] = llclus_maybe.get((clu,tid),0) + 1
        llclus1_maybe[clu1] = llclus1_maybe.get(clu1,0) + 1
    for bid in people_invited:
        if bid not in biker:
            continue
        lang_invited[(biker[bid][0], tid)] = lang_invited.get((biker[bid][0], tid), 0) + 1
        clu = kmeans_ll.predict(np.array([biker[bid][7:9]]))[0]
        clu1 = kmeans_ll1.predict(np.array([biker[bid][7:9]]))[0]
        llclus_invited[(clu, tid)] = llclus_invited.get((clu,tid),0) + 1
        llclus1_invited[clu1] = llclus1_invited.get(clu1,0) + 1
    for bid in people_notgoing:
        if bid not in biker:
            continue
        lang_notgoing[(biker[bid][0], tid)] = lang_notgoing.get((biker[bid][0], tid), 0) + 1
        clu = kmeans_ll.predict(np.array([biker[bid][7:9]]))[0]
        clu1 = kmeans_ll1.predict(np.array([biker[bid][7:9]]))[0]
        llclus_notgoing[(clu,tid)] = llclus_notgoing.get((clu,tid),0) + 1
        llclus1_notgoing[clu1] = llclus1_notgoing.get(clu1,0) + 1
        
bt_attended = {}
bt_maybe = {}
bt_invited = {}
bt_notgoing = {}
for tid, row in tour_convoy_full.items():
    if tid == 'tour_id' or tid not in tours:
        continue
    people_attended = row[0].split()
    people_maybe    = row[1].split()
    people_invited  = row[2].split()
    people_notgoing = row[3].split()
    for bid in people_attended:
        bt_attended[(bid, tid)] = 1
    for bid in people_maybe:
        bt_maybe[(bid, tid)] = 1
    for bid in people_invited:
        bt_invited[(bid, tid)] = 1
    for bid in people_notgoing:
        bt_notgoing[(bid, tid)] = 1


# In[32]:


def get_like_count(bid):
    return like_count.get(bid,0)

def get_dislike_count(bid):
    return dislike_count.get(bid,0)

def get_like_tour_count(tid):
    return like_tour_count.get(tid,0)

def get_dislike_tour_count(tid):
    return dislike_tour_count.get(tid,0)

def get_no_friends_count(bid):
    return no_friends.get(bid,0)

def get_train_count(bid):
    return train_count.get(bid,0)

def get_test_count(bid):
    return test_count.get(bid,0)

def get_train_tour_count(tid):
    return train_tour_count.get(tid,0)

def get_test_tour_count(tid):
    return test_tour_count.get(tid,0)

def get_country(bid):
    return country_val.get(biker[bid][1], 10)

def get_lang(bid):
    return lang_val.get(biker[bid][0].upper(), 10)

def get_tour_country(tid):
    return tour_country_val.get(tours[tid][5], 10)

def get_tour_city(tid):
    return tour_city_val.get(tours[tid][2], 10)

def get_delta1(bid, tid, timestamp):
    return (abs(datetime.strptime(timestamp[:10], "%d-%m-%Y") - 
                datetime.strptime(tours[tid][1], "%d-%m-%Y")).total_seconds())/1e7

def get_delta2(bid, tid, timestamp):
    return (abs(datetime.strptime(timestamp[:10], "%d-%m-%Y") - 
                datetime.strptime(biker[bid][4], "%d-%m-%Y")).total_seconds())/1e7

def get_delta3(bid, tid):
    return (abs(datetime.strptime(biker[bid][4], "%d-%m-%Y") - 
                datetime.strptime(tours[tid][1], "%d-%m-%Y")).total_seconds())/1e7

def get_distance(bid, tid):
    return distance((biker[bid][7], biker[bid][8]), (tours[tid][6], tours[tid][7])).miles

def get_llclus(bid,tid):
    clus = kmeans_ll.predict(np.array([biker[bid][7:9]]))[0]
    return [llclus_attended.get((clus,tid),0),llclus_maybe.get((clus,tid),0),llclus_invited.get((clus,tid),0),llclus_notgoing.get((clus,tid),0)]

def get_llclus1(bid):
    clus1 = kmeans_ll1.predict(np.array([biker[bid][7:9]]))[0]
    return [llclus1_attended.get(clus1,0), llclus1_maybe.get(clus1,0),llclus1_invited.get(clus1,0),llclus1_notgoing.get(clus1,0)]

def get_top10_wc(tid):
    row = tours[tid]
    w = []
    for i in range(8, 109):
        w.append(int(row[i]))
    w = np.array(w)
    return kmeans.predict(np.array([w]))[0]

def get_sum_wc(tid):
    row = tours[tid]
    s = 0
    for i in range(8,18):
        s += int(row[i])
    return s

def get_gender(bid):
    return biker[bid][3] == 'male'
    
def get_age(bid):
    by = (biker[bid][2])
    if by == None or by == 'None':
        by = 1990
    return (2020 - int(by))

def get_friend_with_org(bid, tid):
    org = tours[tid][0]
    frens = biker_net[bid][0].split()
    return org in frens

def get_ymdw(tobj):
    tim = datetime.strptime(tobj, "%d-%m-%Y")
    return [tim.year, tim.month, tim.day, tim.weekday()]

def get_timezone(bid):
    tz = biker[bid][6]
    if tz == '':
        return 7.0
    return float(tz)/60

def get_sim_a(bid, tid):
    v1 = events_attended_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_m(bid, tid):
    v1 = events_maybe_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_i(bid, tid):
    v1 = events_invited_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_n(bid, tid):
    v1 = events_notgoing_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_af(bid, tid):
    v1 = events_attended_friends_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_mf(bid, tid):
    v1 = events_maybe_friends_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_if(bid, tid):
    v1 = events_invited_friends_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

def get_sim_nf(bid, tid):
    v1 = events_notgoing_friends_dist.get(bid, None)
    if v1 is None:
        return 0
    v2 = np.array(list(map(lambda x: int(x), tours[tid][8:109])))
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0 :
        return 0
    return np.dot(v1, v2)/(a*b)

fraction_in = {}
fraction_in1 = {}
fraction_in2 = {}
total_attendees = {}
total_maybe = {}
total_invited = {}
total_notgoing = {}
deltas = {}

for tid, row in tour_convoy.items():
    not_going = len(row[3].split())
    invited = len(row[2].split())
    maybe = len(row[1].split())
    attend = len(row[0].split())
    fraction_in[tid]  = (not_going)/(invited+1)
    fraction_in1[tid] = (attend)/(invited+1)
    fraction_in2[tid] = (maybe)/(invited+1)
    total_attendees[tid] = attend
    total_maybe[tid] = maybe
    total_invited[tid] = invited
    total_notgoing[tid] = maybe

for index, row in train.iterrows():
    a = row["biker_id"]
    b = row["tour_id"]
    c = row["timestamp"]
    curr_delta = get_delta1(a,b,c)
    if a not in deltas:
        deltas[a] = curr_delta
    else:
        deltas[a] = min(deltas[a], curr_delta)
        
for index, row in test.iterrows():
    a = row["biker_id"]
    b = row["tour_id"]
    c = row["timestamp"]
    curr_delta = get_delta1(a,b,c)
    if a not in deltas:
        deltas[a] = curr_delta
    else:
        deltas[a] = min(deltas[a], curr_delta)


# In[33]:


cluster_attend = {}
cluster_maybe = {}
cluster_invited = {}
cluster_notgoing = {}
for tid, row in tours.items():
    w = []
    for i in range(8, 109):
        w.append(int(row[i]))
    w = np.array(w)
    clus = kmeans.predict(np.array([w]))[0]
    cluster_attend[tid]   = cluster_attend.get(tid, 0) + total_attendees[tid]
    cluster_maybe[tid]    = cluster_maybe.get(tid, 0) + total_maybe[tid]
    cluster_invited[tid]  = cluster_invited.get(tid, 0) + total_invited[tid]
    cluster_notgoing[tid] = cluster_notgoing.get(tid, 0) + total_notgoing[tid]

cluster_attend_bt   = {}
cluster_maybe_bt    = {}
cluster_invited_bt  = {}
cluster_notgoing_bt = {}    
for bid, row in biker.items():
    for tid in events_attended_bt.get(bid,[]):
        if tid not in tours:
            continue
        w = list(map(int, tours[tid][8:109])) 
        w = np.array(w)
        clus = kmeans.predict(np.array([w]))[0]
        cluster_attend_bt[(bid,clus)] = cluster_attend_bt.get((bid,clus), 0) + 1
    for tid in events_maybe_bt.get(bid,[]):
        if tid not in tours:
            continue
        w = list(map(int, tours[tid][8:109])) 
        w = np.array(w)
        clus = kmeans.predict(np.array([w]))[0]
        cluster_maybe_bt[(bid,clus)] = cluster_maybe_bt.get((bid,clus), 0) + 1
    for tid in events_invited_bt.get(bid,[]):
        if tid not in tours:
            continue
        w = list(map(int, tours[tid][8:109])) 
        w = np.array(w)
        clus = kmeans.predict(np.array([w]))[0]
        cluster_invited_bt[(bid,clus)] = cluster_invited_bt.get((bid,clus), 0) + 1
    for tid in events_notgoing_bt.get(bid,[]):
        if tid not in tours:
            continue
        w = list(map(int, tours[tid][8:109])) 
        w = np.array(w)
        clus = kmeans.predict(np.array([w]))[0]
        cluster_notgoing_bt[(bid,clus)] = cluster_notgoing_bt.get((bid,clus), 0) + 1
        
def get_clus_bt(bid,tid):
    w = list(map(int, tours[tid][8:109])) 
    w = np.array(w)
    clus = kmeans.predict(np.array([w]))[0]
    return [cluster_attend_bt.get((bid,clus),0),cluster_maybe_bt.get((bid,clus),0),cluster_invited_bt.get((bid,clus),0),cluster_notgoing_bt.get((bid,clus),0)]

# In[34a]:

#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.decomposition import TruncatedSVD
#dtMatrix = []
#tid_to_ind = {}
#for index, (tid, row) in enumerate(tours.items()):
#    w = list(map(int, tours[tid][8:109]))
#    dtMatrix.append(w)
#    tid_to_ind[tid] = index
#dtMatrix = np.array(dtMatrix)
#tfidf = TfidfTransformer()
#tfidfMatrix = tfidf.fit_transform(dtMatrix).toarray()
#svd = TruncatedSVD(n_components = 10, random_state = 0)
#svdMatrix = svd.fit_transform(tfidfMatrix)

# In[34]:


bikers_useful = pd.read_csv("bikers_useful.csv")
tours_useful = pd.read_csv("tours_useful.csv")

train_x=pd.merge(train,bikers_useful,on='biker_id',how='left')
train_x=pd.merge(train_x,tours_useful,on='tour_id',how='left')

Y=train_x['like']
train_x.drop(columns=['like','dislike', 'biker_id_y'],inplace=True)

test_x=pd.merge(test,bikers_useful,on='biker_id',how='left')
test_x=pd.merge(test_x,tours_useful,on='tour_id',how='left')
test_x.drop(columns=['biker_id_y'],inplace=True)

for col in ['gender', 'country', 'city', 'state', 'pincode', 'area']:
    train_x[col].fillna('',inplace=True)
    test_x[col].fillna('',inplace=True)


# In[35]:


d = 0
def make_feature(bid, tid, timestamp, invited):
    f = [0.0]*85
    
    f[0] = get_delta1(bid, tid, timestamp)
    f[1] = fraction_in[tid]
    f[2] = total_attendees[tid]
    f[3] = get_distance(bid, tid)
    f[4] = events_attended_friends[bid]
    f[5] = events_maybe_friends[bid]
    f[6] = events_notgoing_friends[bid]
    f[7] = 1 if f[0] == 0 else deltas[bid]/f[0]
    f[8] = total_maybe[tid]
    
    f[9] = int(get_country(bid))               #cat
    
    f[10] = int(get_top10_wc(tid))             #cat
    
    f[11] = get_train_count(bid)
    f[12] = get_test_count(bid)
    f[13] = get_delta2(bid, tid, timestamp)
    f[14] = get_delta3(bid, tid)
    f[15] = total_notgoing[tid]
    f[16] = get_friend_with_org(bid, tid)
    f[17] = get_age(bid)
    
    t1 = get_ymdw(timestamp[:10])
    t1 += get_ymdw(biker[bid][4])
    t1 += get_ymdw(tours[tid][1])
    for j in range(12):
        f[18+j] = t1[j]                    #cat
    
    f[30] = get_tour_country(tid)          #cat   
    f[31] = get_tour_city(tid)             #cat     
    f[32] = get_lang(bid)                  #cat
    
    f[33] = get_no_friends_count(bid)
    f[34] = get_timezone(bid)
    f[35] = get_sum_wc(tid)
    f[36] = (t1[3] in [4,5,6])             #cat
    f[37] = (t1[7] in [4,5,6])             #cat
    f[38] = (t1[11] in [4,5,6])            #cat
    
    f[39] = get_like_count(bid)
    f[40] = get_dislike_count(bid)
    
    f[41] = get_train_tour_count(tid)
    
    f[42] = events_attended.get(bid,0)
    f[43] = events_maybe.get(bid,0)
    f[44] = events_notgoing.get(bid,0)
    
    f[45] = get_test_tour_count(tid)
    
    f[46] = get_sim_a(bid, tid)
    f[47] = get_sim_m(bid, tid)
    f[48] = get_sim_n(bid, tid)
    f[49] = get_sim_af(bid, tid)
    f[50] = get_sim_mf(bid, tid)
    f[51] = get_sim_nf(bid, tid)
    
    f[52] = t1[1]//4                       #cat
    f[53] = t1[5]//4                       #cat
    f[54] = t1[9]//4                       #cat
    
    f[55] = int(t1[9] == 12)               #cat
   
    f[56] = cluster_attend[tid]  
    f[57] = cluster_notgoing[tid]
    f[58] = cluster_maybe[tid]   
    
    f[59] = lang_attended.get((biker[bid][0], tid),0)
    f[60] = lang_notgoing.get((biker[bid][0], tid),0)
    f[61] = lang_maybe.get((biker[bid][0], tid),0)

    llclu1 = get_llclus1(bid)
    f[62] = llclu1[0]
    f[63] = llclu1[1]
    f[64] = llclu1[2]
    f[65] = llclu1[3]
    
    f[66] = events_attended_friends_bt.get((bid,tid),0)
    f[67] = events_maybe_friends_bt.get((bid,tid),0)
    f[68] = events_invited_friends_bt.get((bid,tid),0)
    f[69] = events_notgoing_friends_bt.get((bid,tid),0)
    
    f[70] = lang_invited.get((biker[bid][0], tid),0)
    f[71] = cluster_invited[tid]   
    
    f[72] = get_sim_i(bid, tid)
    f[73] = get_sim_if(bid, tid)
    
    f[74] = events_invited.get(bid,0)
    f[75] = events_invited_friends[bid]
    f[76] = total_invited[tid]
    
    llclu = get_llclus(bid,tid)
    f[77] = llclu[0]
    f[78] = llclu[1]
    f[79] = llclu[2]
    f[80] = llclu[3]  

    return f

X = []
for index, row in tqdm(train_x.iterrows()):
    a = row["biker_id_x"]
    b = row["tour_id"]
    c = row["timestamp"]
    d = row["invited"]
    curr_l = list(row.values)
    curr_l += make_feature(a, b, c, d)
    X.append(curr_l)
        
X = pd.DataFrame(X)

test_X = []
for index, row in tqdm(test_x.iterrows()):
    a = row["biker_id_x"]
    b = row["tour_id"]
    c = row["timestamp"]
    d = row["invited"]
    curr_l = list(row.values)
    curr_l += make_feature(a, b, c, d)
    test_X.append(curr_l)
        
test_X = pd.DataFrame(test_X)

for c in [3,8,13]:
    X[c] = (pd.to_datetime(X[c]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    test_X[c] = (pd.to_datetime(test_X[c]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

X.drop(columns=[0,1],inplace=True)
test_X.drop(columns=[0,1],inplace=True)

# In[47]:


cat_features = [0,1,2,3,4,5,6,7,11,12,13,14,15,128,129,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,155,156,157,171,172,173,174,182]
preds = np.zeros(test_X.shape[0])
for c in cat_features:
    if (c+2) not in X:
        continue
    col_type = X[c+2].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c+2] = X[c+2].astype('category')
        test_X[c+2] = test_X[c+2].astype('category')

fit_params1={ 
            "metric" : 'auc', 
            'random_state':0,
            'deterministic':True,
            'n_jobs':1,
            'num_leaves':27,
            'n_estimators':300,
            'reg_lambda':0.0000,
            'learning_rate':0.04500,
}

fit_params2={ 
            "metric" : 'auc', 
            'random_state':0,
            'deterministic':True,
            'n_jobs':1,
            'num_leaves':27,
            'n_estimators':450,
            'reg_lambda':0.0000,
            'learning_rate':0.04500,
}

fit_params3={ 
            "metric" : 'auc', 
            'random_state':0,
            'deterministic':True,
            'n_jobs':1,
            'num_leaves':27,
            'n_estimators':600,
            'reg_lambda':0.0000,
            'learning_rate':0.04500,
}

clf1 = LGBMClassifier(
        **fit_params1
    )
clf2 = LGBMClassifier(
        **fit_params2
    )
clf3 = LGBMClassifier(
        **fit_params3
    )

from sklearn.metrics import log_loss
n_bag = 175
sss = StratifiedShuffleSplit(n_splits=n_bag, test_size=0.1, random_state=0)
sss.get_n_splits(X, Y)

print(sss)       
ct = 0
for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)
    y1 = clf1.predict_proba(X_test)[:,1]
    y2 = clf2.predict_proba(X_test)[:,1]
    y3 = clf3.predict_proba(X_test)[:,1]
    y4 = clf1.predict_proba(test_X)[:,1]
    y5 = clf2.predict_proba(test_X)[:,1]
    y6 = clf3.predict_proba(test_X)[:,1]
    a = (1-log_loss(y_test, (y1+y2+y3)/3))
    if a > 0.57:
        preds = preds + (y4+y5+y6)
        ct+=1


# In[48]:


ts = {}
inv = {}
num_bt = {}
for index, row in test.iterrows():
    ts[(row["biker_id"], row["tour_id"])] = row["timestamp"]
    inv[(row["biker_id"], row["tour_id"])] = row["invited"]
    num_bt[(row["biker_id"], row["tour_id"])] = index
    
test_Y = preds/ct


# In[49]:


train_bikers_set = np.array(train["biker_id"].drop_duplicates())
bikers_out = []
tours_out = []
for biker1 in tqdm(test_bikers_set):
    idx = np.where(biker1==test["biker_id"]) 
    tour = list(test["tour_id"].loc[idx]) # for each unique biker in test data get all the events  
    score = {}
    for tou in tour:
        s = test_Y[num_bt[(biker1, tou)]]
        score[tou] = s
    tour.sort(key = lambda x : score[x], reverse = True)
    tour = " ".join(tour) # list to space delimited string
    bikers_out.append(biker1)
    tours_out.append(tour)


# In[50]:


sample_submission =pd.DataFrame(columns=["biker_id","tour_id"])
sample_submission["biker_id"] = bikers_out
sample_submission["tour_id"] = tours_out
sample_submission.to_csv("CS18B032_CS18B050_1.csv",index=False)


# In[51]:


d = 0
def make_feature(bid, tid, timestamp, invited):
    f = [0.0]*85
    
    f[0] = get_delta1(bid, tid, timestamp)
    f[1] = fraction_in[tid]
    f[2] = total_attendees[tid]
    f[3] = get_distance(bid, tid)
    f[4] = events_attended_friends[bid]
    f[5] = events_maybe_friends[bid]
    f[6] = events_notgoing_friends[bid]
    f[7] = 1 if f[0] == 0 else deltas[bid]/f[0]
    f[8] = total_maybe[tid]
    
    f[9] = int(get_country(bid))               #cat
    
    f[13] = get_delta2(bid, tid, timestamp)
    f[14] = get_delta3(bid, tid)
    f[15] = total_notgoing[tid]
    f[16] = get_friend_with_org(bid, tid)
    f[17] = get_age(bid)
    
    t1 = get_ymdw(timestamp[:10])
    t1 += get_ymdw(biker[bid][4])
    t1 += get_ymdw(tours[tid][1])
    for j in range(12):
        f[18+j] = t1[j]                    #cat
    
    f[30] = get_tour_country(tid)          #cat   
    f[31] = get_tour_city(tid)             #cat     
    f[32] = get_lang(bid)                  #cat
    
    f[33] = get_no_friends_count(bid)
    f[34] = get_timezone(bid)
    f[35] = get_sum_wc(tid)
    f[36] = (t1[3] in [4,5,6])             #cat
    f[37] = (t1[7] in [4,5,6])             #cat
    f[38] = (t1[11] in [4,5,6])            #cat
    
    f[42] = events_attended.get(bid,0)
    f[43] = events_maybe.get(bid,0)
    f[44] = events_notgoing.get(bid,0)
    
    f[66] = events_attended_friends_bt.get((bid,tid),0)
    f[67] = events_maybe_friends_bt.get((bid,tid),0)
    f[68] = events_invited_friends_bt.get((bid,tid),0)
    f[69] = events_notgoing_friends_bt.get((bid,tid),0)
    
    
    f[74] = events_invited.get(bid,0)
    f[75] = events_invited_friends[bid]
    f[76] = total_invited[tid]
    
    return f

X = []
for index, row in tqdm(train_x.iterrows()):
    a = row["biker_id_x"]
    b = row["tour_id"]
    c = row["timestamp"]
    d = row["invited"]
    curr_l = list(row.values)
    curr_l += make_feature(a, b, c, d)
    X.append(curr_l)
        
X = pd.DataFrame(X)

test_X = []
for index, row in tqdm(test_x.iterrows()):
    a = row["biker_id_x"]
    b = row["tour_id"]
    c = row["timestamp"]
    d = row["invited"]
    curr_l = list(row.values)
    curr_l += make_feature(a, b, c, d)
    test_X.append(curr_l)
        
test_X = pd.DataFrame(test_X)

for c in [3,8,13]:
    X[c] = (pd.to_datetime(X[c]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    test_X[c] = (pd.to_datetime(test_X[c]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

X.drop(columns=[0,1],inplace=True)
test_X.drop(columns=[0,1],inplace=True)


# In[56]:


cat_features = [0,1,2,3,4,5,6,7,11,12,13,14,15,128,129,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,155,156,157,171,172,173,174,182]
preds = np.zeros(test_X.shape[0])
for c in cat_features:
    if (c+2) not in X:
        continue
    col_type = X[c+2].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c+2] = X[c+2].astype('category')
        test_X[c+2] = test_X[c+2].astype('category')

fit_params={ 
            "metric" : 'auc', 
            'random_state':0,
            'deterministic':True,
            'n_jobs':1,
            'num_leaves':27,
            'n_estimators':300,
            'reg_lambda':0.0000,
            'learning_rate':0.05200,
            'colsample_bytree':0.95,
}
ct = 0
for csb in [0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00]:
    fit_params['colsample_bytree']=csb
    clf = LGBMClassifier(**fit_params)
    clf.fit(X,Y)
    preds+=clf.predict_proba(test_X)[:,1]
    ct+=1


# In[57]:


ts = {}
inv = {}
num_bt = {}
for index, row in test.iterrows():
    ts[(row["biker_id"], row["tour_id"])] = row["timestamp"]
    inv[(row["biker_id"], row["tour_id"])] = row["invited"]
    num_bt[(row["biker_id"], row["tour_id"])] = index
    
test_Y = preds/ct


# In[58]:


train_bikers_set = np.array(train["biker_id"].drop_duplicates())
bikers_out = []
tours_out = []
for biker1 in tqdm(test_bikers_set):
    idx = np.where(biker1==test["biker_id"]) 
    tour = list(test["tour_id"].loc[idx]) # for each unique biker in test data get all the events  
    score = {}
    for tou in tour:
        s = test_Y[num_bt[(biker1, tou)]]
        score[tou] = s
    tour.sort(key = lambda x : score[x], reverse = True)
    tour = " ".join(tour) # list to space delimited string
    bikers_out.append(biker1)
    tours_out.append(tour)


# In[59]:


sample_submission =pd.DataFrame(columns=["biker_id","tour_id"])
sample_submission["biker_id"] = bikers_out
sample_submission["tour_id"] = tours_out
sample_submission.to_csv("CS18B032_CS18B050_2.csv",index=False)


# In[ ]:


os.remove("bikers_useful.csv")
os.remove("bikers_network_useful.csv")
os.remove("tours_useful.csv")
os.remove("tour_convoy_useful.csv")

