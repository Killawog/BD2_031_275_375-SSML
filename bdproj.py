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

import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,VectorAssembler #OneHotEncoderEstimator, 
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
#from pyspark.ml.classification import LogisticRegression
from sklearn.linear_model import LogisticRegression
from pyspark.sql import Row, Column
import sys

sc = SparkContext("local[2]", "APSU")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

def convert_jsn(data):
	jsn=json.loads(data)
	l=list()
	for i in jsn:
		rows=tuple(jsn[i].values())
		l.append(rows)
	return l 	

def convert_df(data):
	if data.isEmpty():
		return

	ss=SparkSession(data.context)
	data=data.collect()[0]
	col=[f"feature{i}" for i in range(len(data[0]))]
	df=ss.createDataFrame(data,col)
	#df.show()
	
	print('\n\nDefining the  stages.................\n')
	df_new=df
	#df_new = df.withColumn("feature1", array(df["feature1"]))
	regex = RegexTokenizer(inputCol= 'feature1' , outputCol= 'tokens', pattern= '\\W')
	#df_new=regex.transform(df_new)
	#df=stage_1.fit(df).transform(df)
	#print("Regex done")

	
	remover2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	#df_new=remover2.transform(df_new)
	print("Stopwords done")

	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	#df_new=stage_3.transform(df_new)
	#df_new.show()
	#mod.getVectors().show()
	#df_new=stage_3.fit(df_new).transform(df_new)
	#df_new.show()
	
	
	print("Word2vec done")

	
	indexer = StringIndexer(inputCol="feature2", outputCol="categoryIndex")
	#df_new = indexer.fit(df_new).transform(df_new)
	#df_new.show()
	print("Target column Done")
	
	#x=df_new.select('vector').collect()
	#col='vector'
	#train=ss.createDataFrame(x, col)
	#nsamples, nx, ny = x.shape
	#df_new = df_new.withColumn('vector').reshape((nsamples,nx*ny))
	
	#x=np.array(df_new.select('vector').collect())
	
	pipeline=Pipeline(stages=[regex, remover2, stage_3, indexer])
	pipelineFit=pipeline.fit(df)
	dataset=pipelineFit.transform(df)
	dataset.show(5)
	
	#y=np.array(df_new.select('categoryIndex').collect())

	#x=df_new.select('vector').collect()
	#y=df_new["categoryIndex"]


	
	print(df_new.dtypes)

	#model = LogisticRegression(featuresCol= 'vector', labelCol= 'categoryIndex')
	#model=LogisticRegression(warm_start=True)
	#model.fit(x,y)
	#print("u r a genius")
	
	#print('\n\nStages Defined................................\n')
	#pipeline = Pipeline(stages= [stage_1, stage_2, stage_3])
	
	
	#print('\n\nFit the pipeline with the training data.......\n')
	#pipelineFit = pipeline.fit(df)
	
	#model=SGDClassifier(alpha=0.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
	#df_new = df_new.withColumn("vector", array(df_new["vector"]))

	
	

	
	#model.partial_fit(x,y, classes=[0.0,1.0])
	#model.fit(df_new)
	print("Hi")

	
	#X=mod
	#Y=df['categoryIndex']
	#model.partial_fit(X, Y, classes=[0.0,1.0])
	


lines = ssc.socketTextStream("localhost",6100).map(convert_jsn).foreachRDD(convert_df)



ssc.start() 
ssc.awaitTermination(100)
ssc.stop()




