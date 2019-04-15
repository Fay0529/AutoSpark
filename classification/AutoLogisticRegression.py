
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os
sys.path.append("..")
from AutoEstimator import AutoEstimator


'''
线性回归模型自己实现
支持basePredictor的保存与直接使用
'''


class AutoLogisticRegression(AutoEstimator):

    def __init__(self, evaluator=None,
                 trainData=None,
                 predictorPath="basePredictors",
                 spark=None,
                 cvParallelism=1,
                 numFolds=3):
        AutoEstimator.__init__(self,
                               evaluator=evaluator,
                               trainData=trainData,
                               predictorPath=predictorPath,
                               spark=spark,
                               estimator=LogisticRegression,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getPCS(self):
        '''
        elasticNetParam：[0,0.5], 弹性网络混合参数,默认0.1
        maxIter: [1,100]最大迭代次数,默认50
        regParam :[0,0.4] 正则化参数，默认0
        tol:[1e-6,1e-1] 迭代算法收敛性，默认 1e-6

        '''
        # Build Configuration Space which defines all parameters and their
        # ranges
        cs = ConfigurationSpace()
        elasticNetParam = UniformFloatHyperparameter(
            "elasticNetParam", 0.0, 1, default_value=0.1)
        maxIter = UniformIntegerHyperparameter(
            "maxIter", 1, 100, default_value=50)
        regParam = UniformFloatHyperparameter(
            "regParam", 0, 0.4, default_value=1e-04)
        tol = UniformFloatHyperparameter(
            "tol", 1e-06, 1e-01, default_value=1e-06)
        cs.add_hyperparameters([elasticNetParam, maxIter, regParam, tol])
        return cs

    def getDimension(self):
        return 5


if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = "python3"
    spark = SparkSession.builder.appName(
        "AutoLogisticRegression").getOrCreate()
    sc = spark.sparkContext
    training = spark.read \
        .format("libsvm") \
        .load("../data/sample_multiclass_classification_data.txt")
    evaluator = MulticlassClassificationEvaluator()
    autoLr = AutoLogisticRegression(
        trainData=training, evaluator=evaluator, spark=spark)
    model = autoLr.getBestModel(ta_run_limit=50)
    print(evaluator.evaluate(model.transform(training)))
    # model2 = autoLr.getBestModelWithSubSampling(
    #     lowRatio=0.3, midRatio=0.8, taeLimit=30, runcount_limit=50)
    # print(evaluator.evaluate(model2.transform(training)))
