from pyspark.ml.regression import LinearRegression
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter
from pyspark.ml.evaluation import RegressionEvaluator
import sys
sys.path.append("..")
from AutoEstimator import AutoEstimator


class AutoLinearRegression(AutoEstimator):
    def __init__(self,
                 trainData=None,
                 predictorPath="basePredictors",
                 spark=None,
                 cvParallelism=1,
                 numFolds=3):
        AutoEstimator.__init__(self,
                               evaluator=RegressionEvaluator(),
                               trainData=trainData,
                               predictorPath=predictorPath,
                               spark=spark,
                               estimator=LinearRegression,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getPCS(self):
        '''
        elasticNetParam：[0,1], 弹性网络混合参数,默认0.1
        maxIter: [1,100]最大迭代次数,默认50
        regParam :[0,0.2] 正则化参数，默认0
        tol:[1e-6,1e-1] 迭代算法收敛性，默认 1e-6


        '''
        # Build Configuration Space which defines all parameters and their
        # ranges
        cs = ConfigurationSpace()
        maxIter = UniformIntegerHyperparameter(
            "maxIter", 1, 100, default_value=50)
        regParam = UniformFloatHyperparameter(
            "regParam", 0, 0.4, default_value=1e-04)
        tol = UniformFloatHyperparameter(
            "tol", 1e-06, 1e-01, default_value=1e-06)
        elasticNetParam = UniformFloatHyperparameter(
            "elasticNetParam", 0.0, 1, default_value=0.1)
        cs.add_hyperparameters(
            [maxIter, regParam, tol, elasticNetParam])
        return cs

    def getDimension(self):
        return 4
