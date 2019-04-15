from pyspark.ml.regression import GBTRegressor
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter
from pyspark.ml.evaluation import RegressionEvaluator
import sys
sys.path.append("..")
from AutoEstimator import AutoEstimator


class AutoGBTRegressor(AutoEstimator):
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
                               estimator=GBTRegressor,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getPCS(self):
        '''
        maxDepth  [3,12] 默认5
        minInstancesPerNode：[1,100] 默认10
        minInfoGain=:[0,0.1] 默认0
        maxIter=[ 1,50] ,默认20
        stepSize [0.001,1]默认0.1,

        '''
        # Build Configuration Space which defines all parameters and their
        # ranges
        cs = ConfigurationSpace()
        maxDepth = UniformIntegerHyperparameter(
            "maxDepth", 3, 12, default_value=5)
        maxIter = UniformIntegerHyperparameter(
            "maxIter", 1, 50, default_value=20)
        minInstancesPerNode = UniformIntegerHyperparameter(
            "minInstancesPerNode", 1, 100, default_value=10)
        minInfoGain = UniformFloatHyperparameter(
            "minInfoGain", 1e-06, 1e-01, default_value=1e-06)
        stepSize = UniformFloatHyperparameter(
            "stepSize", 0.001, 1, default_value=0.1)
        cs.add_hyperparameters(
            [maxDepth, maxIter, minInstancesPerNode, minInfoGain,
                stepSize])
        return cs

    def getDimension(self):
        return 5
