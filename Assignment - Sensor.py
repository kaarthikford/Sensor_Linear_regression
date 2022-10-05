#!/usr/bin/env python
# coding: utf-8

#Import necessary packages for loading the data from mongodb
get_ipython().system('pip install pymongo')
get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')


#Import necessary packages
import pymongo
import pandas
import pprint
from pandas import datetime 


#Connect to the collection in mongodb
mongo_uri="mongodb://localhost:27017/"
client=pymongo.MongoClient(mongo_uri)
db=client.intern.sensor

#Query to get the structured data
db1=db.aggregate([{
    "$unwind": "$data"
},
    {
        "$match": {"data.timestamp": { "$gt": datetime(2018,1,1,0,0,15),"$lt": datetime(2018,1,2,0,0,15)} } },
  { "$group":{"_id": {"data":"$data.timestamp","val":"$data.value","sensor":"$sensor_name"} } 
  },
  {
      "$unwind": "$_id"
  },
    {"$out":"output"}
])


get_ipython().system('pip install ndjson')
# Python program to read
# json file
import ndjson

# Opening JSON file
f = ndjson.load(open('out.json'))



from pandas import json_normalize
fil = json_normalize(f) 
print(fil)


fil.to_csv("file.csv")

df = pandas.read_csv('output.csv',index_col=0)
print(df)


df1 = pandas.pivot_table(df,index='Date',columns='Sensor',values="Value")
print(df1)

df1.columns = [''.join(str(s) for s in col if s) for col in df1.columns]

df1

df1.fillna(method ='ffill', inplace = True)

#Preparing data for regression analysis
X = df1.drop("Pump Radial Bearing Vibration",1)   
y = df1["Pump Radial Bearing Vibration"]

#Splitting the training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)


# create linear regression object
from sklearn import datasets, linear_model, metrics
reg = linear_model.LinearRegression()
  
# train the model using the training sets
reg.fit(X_train, y_train)
  
# regression coefficients
print('Coefficients: ', reg.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))


predictions = reg.predict(X_test)


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))


Accuracy=r2_score(y_test,predictions)*100
print(" Accuracy of the model is %.2f" %Accuracy)
