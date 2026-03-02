from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import happybase
import json
from pyspark.sql.functions import split, col

# Spark session with Hive
spark = SparkSession.builder.appName("SleepQuality Prediction").enableHiveSupport().getOrCreate()

# Load Hive table
data = spark.sql("SELECT * FROM sleep").na.drop()

# Blood pressure feature split
data = data.withColumn("bp_systolic", split(col("blood_pressure"), "/").getItem(0).cast("float"))
data = data.withColumn("bp_diastolic", split(col("blood_pressure"), "/").getItem(1).cast("float"))

# Categorical columns encoding 
categorical_cols = ["gender", "occupation", "bmi_category", "sleep_disorder"]
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
    data = indexer.fit(data).transform(data)

# Model features
feature_cols = [
    "age",
    "sleep_duration",
    "physical_activity_minutes",
    "stress_level",
    "heart_rate",
    "daily_steps",
    "bp_systolic",
    "bp_diastolic",
    "gender_index",
    "occupation_index",
    "bmi_category_index",
    "sleep_disorder_index"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

assembled = assembler.transform(data).select("features", "quality_of_sleep")
assembled = assembled.withColumnRenamed("quality_of_sleep", "label")

# Test/training split
train_data, test_data = assembled.randomSplit([0.8, 0.2], seed=42)

# Random forest regressor model
rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100)
model = rf.fit(train_data)

# Evaluate model
predictions = model.transform(test_data)
rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
print(f"RMSE: {rmse}, R2: {r2}")

# Feature importances
feature_importances = {feature_cols[i]: float(model.featureImportances[i]) for i in range(len(feature_cols))}

# Write to HBase
data = [
    ('sleep1', 'metrics:rmse', str(rmse)),
    ('sleep1', 'metrics:r2', str(r2)),
    ('sleep1', 'importance:feature_importances', json.dumps(feature_importances))
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('localhost', 9090)
    connection.open()
    table = connection.table('sleep_metrics')
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Stop Spark
spark.stop()