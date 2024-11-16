from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
import json

# Initialize Spark session
spark = SparkSession.builder.appName("RealTimeFraudDetection").getOrCreate()

# Set Spark configuration for local directory (to store temporary files like shuffle data)
spark.conf.set("spark.local.dir", "/Users/lavish./Desktop/ML_Projects/Credit-Card-Default-Prediction/streaming")  # Update this to your actual path

# Load the pre-trained Random Forest model
fraud_model = RandomForestClassificationModel.load("streaming/spark_model")

# Initialize Streaming Context with batch interval of 10 seconds
ssc = StreamingContext(spark.sparkContext, 10)

# Define feature columns
feature_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Function to process each RDD of streaming data
def process_stream(rdd):
    if not rdd.isEmpty():
        # Parse the incoming JSON data into a DataFrame
        raw_data = rdd.collect()  # Collect data from RDD
        json_data = [json.loads(row) for row in raw_data]  # Parse JSON strings
        df = spark.createDataFrame(json_data)

        # Transform data into feature vectors
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)

        # Predict fraudulent or legitimate transactions
        predictions = fraud_model.transform(df)

        # Display results
        predictions.select("transaction_id", "prediction").show()

# Create a DStream to listen for incoming data on localhost:9999
transactions = ssc.socketTextStream("localhost", 1234)
parsed_data = transactions.map(lambda x: json.loads(x))
parsed_data.pprint()  # Print parsed data for debugging


# Apply the processing function to each RDD in the DStream
transactions.foreachRDD(process_stream)

# Start the streaming context
ssc.start()
ssc.awaitTermination()
