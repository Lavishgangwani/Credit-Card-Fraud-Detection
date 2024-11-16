from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionModelTraining").getOrCreate()

# Load your dataset
data = spark.read.csv("notebooks/data/UCI_Credit_Card.csv", header=True, inferSchema=True)

# Define feature columns and label column
feature_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Split into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train the Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="default_payment_next_month", numTrees=10)
model = rf.fit(train_data)

# Save the model to disk
model.write().overwrite().save("streaming/spark_model")
