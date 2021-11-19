# importing required libraries
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
	df.show()

lines = ssc.socketTextStream("localhost",6100).map(convert_jsn).foreachRDD(convert_df)

ssc.start() 
ssc.awaitTermination(100)
ssc.stop()
