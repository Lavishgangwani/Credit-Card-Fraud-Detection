from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
import pickle 

# Initialize Spark session
spark = SparkSession.builder.appName("RealTimeFraudDetection").getOrCreate()

# Set Spark configuration for local directory (to store temporary files like shuffle data)
spark.conf.set("spark.local.dir", "/Users/lavish/Desktop/ML_Projects/Credit-Card-Default-Prediction/spark-temp")

# Load the pre-trained model using pickle
with open("/Users/lavish./Desktop/ML_Projects/Credit-Card-Default-Prediction/artifacts/model.pkl", "rb") as model_file:
    fraud_model = pickle.load(model_file)

# Initialize Streaming Context
ssc = StreamingContext(spark.sparkContext, 10)  # Batch interval of 10 seconds

# Function to process each batch of streaming data
def process_stream(rdd):
    if not rdd.isEmpty():
        # Read the CSV file from the streaming data (replace with a stream source like Kafka or Flume in production)
        df = spark.read.csv("notebooks/data/UCI_Credit_Card.csv", header=True, inferSchema=True)

        # Define feature columns for fraud detection
        feature_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                        'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

        # VectorAssembler to combine feature columns into a single vector for model input
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)

        # Predict fraudulent or legitimate transactions
        predictions = fraud_model.transform(df)
        
        # Show predictions (transaction ID and prediction result)
        predictions.select("transaction_id", "prediction").show()

# Simulate real-time data input from a socket (Replace with Kafka or Flume in production)
transactions = ssc.socketTextStream("localhost", 9999)
transactions.foreachRDD(process_stream)

# Start Spark Streaming
ssc.start()
ssc.awaitTermination()
