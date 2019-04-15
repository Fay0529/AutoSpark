from pyspark.ml.regression import RandomForestRegressor
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter
from pyspark.ml.evaluation import RegressionEvaluator
import sys
sys.path.append("..")
from AutoEstimator import AutoEstimator


class AutoRandomForestRegressor(AutoEstimator):
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
                               estimator=RandomForestRegressor,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getPCS(self):
        '''
        maxDepth  [3,12] 默认5
        minInstancesPerNode：[1,100] 默认10
        minInfoGain=:[0,0.1] 默认0
        subsamplingRate=[0.01,1], 默认1
        numTrees：[1,100] 默认 50

        '''
        # Build Configuration Space which defines all parameters and their
        # ranges
        cs = ConfigurationSpace()
        maxDepth = UniformIntegerHyperparameter(
            "maxDepth", 3, 12, default_value=5)
        minInstancesPerNode = UniformIntegerHyperparameter(
            "minInstancesPerNode", 1, 100, default_value=10)
        minInfoGain = UniformFloatHyperparameter(
            "minInfoGain", 1e-06, 1e-01, default_value=1e-06)
        numTrees = UniformIntegerHyperparameter(
            "numTrees", 1, 100, default_value=50)
        cs.add_hyperparameters(
            [maxDepth, minInstancesPerNode, minInfoGain, numTrees])
        return cs

    def getDimension(self):
        return 4
