# importing required libraries
import numpy as np
import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import lit
from sklearn.linear_model import SGDClassifier
from pyspark.sql.functions import array
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import pickle
from sklearn import model_selection
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,VectorAssembler #OneHotEncoderEstimator, 
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Word2Vec, RegexTokenizer
#from pyspark.ml.classification import LogisticRegression
import sklearn.linear_model as lm
from pyspark.sql import Row, Column
import sys

sc = SparkContext("local[2]", "APSU")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)
flag=0
#model=lm.LogisticRegression(warm_start=True)

def convert_jsn(data):
	jsn=json.loads(data)
	l=list()
	for i in jsn:
		rows=tuple(jsn[i].values())
		l.append(rows)
	return l 	

def convert_df(data):
	global model, model_lm, model_sgd, model_mlp, result2,result3, result1, x, y
	if data.isEmpty():
		return

	ss=SparkSession(data.context)
	data=data.collect()[0]
	#print(data)
	col=[f"feature{i}" for i in range(len(data[0]))]
	try:
		df=ss.createDataFrame(data,col)
	except:
		return
	#df.show()
	data2=[('ham','ham','ham')]
	newRow=ss.createDataFrame(data2, col)
	df=df.union(newRow)
	#df.show(11)
		
	
	print('\n\nDefining the  stages.................\n')
	df_new=df

	regex = RegexTokenizer(inputCol= 'feature1' , outputCol= 'tokens', pattern= '\\W')

	print("Regex done")

	
	remover2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	print("Stopwords done")

	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	
	#stage_3=CountVectorizer(inputCol="filtered_words", outputCol="vector", vocabSize=10000, minDF=5)
	
	
	print("Word2vec done")

	
	indexer = StringIndexer(inputCol="feature2", outputCol="categoryIndex", stringOrderType='alphabetAsc')

	print("Target column Done")
	
	#nsamples, nx, ny = x.shape
	#df_new = df_new.withColumn('vector').reshape((nsamples,nx*ny))
	
	pipeline=Pipeline(stages=[regex, remover2, stage_3, indexer])
	pipelineFit=pipeline.fit(df)
	dataset=pipelineFit.transform(df)
	#dataset.show(11)
	dataset=dataset.filter(dataset.feature1!='ham')
	new_df=dataset.select(['vector'])
	new_df_target=dataset.select(['categoryIndex'])
	#new_df.show()


	x=np.array(new_df.select('vector').collect())
	y=np.array(new_df_target.select('categoryIndex').collect())
	
	x = [np.concatenate(i) for i in x]

	
	model_lm=lm.LogisticRegression(warm_start=True)
	model_lm=model_lm.fit(x,y.ravel())
	print("u r a genius pt.1")
	result1=model_lm.score(x, y)
	print("Logistic Regression accuracy: ", result1)

	
	model_sgd=SGDClassifier(alpha=0.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
	#model=MultinomialNB()
	model_sgd.partial_fit(x,y.ravel(), classes=[0.0,1.0])
	#model.fit(x,y)
	print("u r a genius pt.2")
	result2=model_sgd.score(x, y)
	print("SGD Classifier accuacy: ",result2)
	
	model_mlp=MLPClassifier(random_state=1, max_iter=300)
	model_mlp.partial_fit(x,y.ravel(), classes=[0.0,1.0])
	print("u r a genius pt.3")
	result3=model_mlp.score(x, y)
	print("MLP Classifier accuacy: ",result3)	
	
	
	


lines = ssc.socketTextStream("localhost",6100).map(convert_jsn).foreachRDD(convert_df)



ssc.start() 
ssc.awaitTermination(150)
ssc.stop()

filename='model_lm_1000.sav'
pickle.dump(model_lm, open(filename, 'wb'))
print("LM Model saved successfully")

filename='model_sgd_1000.sav'
pickle.dump(model_sgd, open(filename, 'wb'))
print("LM Model saved successfully")

filename='model_mlp_1000.sav'
pickle.dump(model_mlp, open(filename, 'wb'))
print("LM Model saved successfully")

results=[result1, result2, result3]
names=['Logistic Regression', 'SGD Classifier', 'MLP Classifer']

#filename='final accuracies'
#file1=open(filename, 'w')
#file1.writelines(names)
#file1.writelines(results)
#file1.close()

plt.bar(names, results)
plt.show()


