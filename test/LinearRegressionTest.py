import os
from pyspark.sql import SparkSession
import sys
sys.path.append("..")
from regression.AutoLinearRegression import AutoLinearRegression
if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = "python3"
    spark = SparkSession\
        .builder\
        .appName("AutoRandomForestRegressor")\
        .getOrCreate()
    sc = spark.sparkContext
    training = spark.read \
        .format("libsvm") \
        .load("../data/sample_linear_regression_data.txt")
    autoLr = AutoLinearRegression(
        trainData=training, spark=spark, cvParallelism=3)
    model = autoLr.getBestModel(ta_run_limit=50)
    print(autoLr.evaluate(model.transform(training)))
    # model2 = autoLr.getBestModelWithSubSampling(
    #     lowRatio=0.3, midRatio=0.8, taeLimit=30, runcount_limit=50)
    # print(evaluator.evaluate(model2.transform(training)))
