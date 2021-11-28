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



def convert_jsn(data):
	jsn=json.loads(data)
	l=list()
	for i in jsn:
		rows=tuple(jsn[i].values())
		l.append(rows)
	return l 	

def convert_df(data):
	global model
	global x
	global y
	if data.isEmpty():
		return

	ss=SparkSession(data.context)
	data=data.collect()[0]
	col=[f"feature{i}" for i in range(len(data[0]))]
	try:
		df=ss.createDataFrame(data,col)
	except:
		return
	#df.show()
	
	print('\n\nDefining the  stages.................\n')
	df_new=df

	regex = RegexTokenizer(inputCol= 'feature1' , outputCol= 'tokens', pattern= '\\W')

	print("Regex done")

	
	remover2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	print("Stopwords done")

	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	
	#stage_3=CountVectorizer(inputCol="filtered_words", outputCol="vector", vocabSize=10000, minDF=5)
	
	
	print("Word2vec done")

	
	indexer = StringIndexer(inputCol="feature2", outputCol="categoryIndex")

	print("Target column Done")
	
	#nsamples, nx, ny = x.shape
	#df_new = df_new.withColumn('vector').reshape((nsamples,nx*ny))
	
	pipeline=Pipeline(stages=[regex, remover2, stage_3, indexer])
	pipelineFit=pipeline.fit(df)
	dataset=pipelineFit.transform(df)
	new_df=dataset.select(['vector'])
	new_df_target=dataset.select(['categoryIndex'])
	new_df.show(5)


	x=np.array(new_df.select('vector').collect())
	y=np.array(new_df_target.select('categoryIndex').collect())
	
	x = [np.concatenate(i) for i in x]
	
	
lines = ssc.socketTextStream("localhost",6100).map(convert_jsn).foreachRDD(convert_df)



ssc.start() 
ssc.awaitTermination(200)
ssc.stop()

loaded_model=pickle.load(open('model.sav', 'rb'))
result=loaded_model.score(x, y)
print(result)

