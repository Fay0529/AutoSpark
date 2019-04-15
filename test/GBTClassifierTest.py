import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import sys
import time
sys.path.append("..")
from classification.AutoGBTClassifier import AutoGBTClassifier
if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = "python3"

    spark = SparkSession\
        .builder\
        .appName("AutoGBTClassifier")\
        .getOrCreate()
    sc = spark.sparkContext
    df = spark.read \
        .format("parquet") \
        .load("../data/data.par")
    trainDF, testDF = df.randomSplit([0.7, 0.3], 1)
    evaluator = BinaryClassificationEvaluator()
    autoLr = AutoGBTClassifier(
        trainData=trainDF, evaluator=evaluator, spark=spark)
    # smac
    # f = open("getBestModel.txt", "a")
    # start = time.perf_counter()
    # model = autoLr.getBestModel(runcount_limit=200)
    # print(
    #     evaluator.evaluate(model.transform(testDF)),
    #     ",time used: %s sec" % (time.perf_counter() - start),
    #     # autoLr.evaluate(model.transform(testDF)),
    #     file=f)
    # f.close()
    # smac_sample
    f = open("sample.txt", "a")

    start = time.perf_counter()
    model = autoLr.getBestModelWithSubSampling(
        lowRatio=0.03, midRatio=0.15, taeLimit=50, runcount_limit=50, basePredictorsDataSizeRatio=60)
    print(autoLr.evaluate(
        model.transform(testDF)),
        "time used: %s seconds" % (time.perf_counter() - start),
        file=f)
    f.close()
    # grid search
    # start = time.perf_counter()
    # gbt = GBTClassifier()
    # grid = ParamGridBuilder()
    # grid = grid.addGrid(gbt.maxDepth, list(range(3, 13, 6)))\
    #     .addGrid(gbt.maxIter, list(range(1, 51, 30)))\
    #     .addGrid(gbt.minInstancesPerNode, list(range(1, 100, 50)))\
    #     .addGrid(gbt.minInfoGain, [1e-06, 1e-01])\
    #     .addGrid(gbt.stepSize, [0.001, 1])\
    #     .build()
    # cv = CrossValidator(estimator=gbt,
    #                     estimatorParamMaps=grid,
    #                     evaluator=evaluator)
    # cvModel = cv.fit(trainDF)
    # print(evaluator.evaluate(
    #     cvModel.transform(testDF)),
    #     "time used: %s seconds" % (time.perf_counter() - start))
