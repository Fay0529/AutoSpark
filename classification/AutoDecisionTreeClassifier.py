from pyspark.ml.classification import DecisionTreeClassifier
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter, CategoricalHyperparameter
import sys
sys.path.append("..")
from AutoEstimator import AutoEstimator


class AutoDecisionTreeClassifier(AutoEstimator):
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
                               estimator=DecisionTreeClassifier,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getPCS(self):
        '''
        maxDepth :[3,12] 树深度 默认5
        minInstancesPerNode：[1,100] 默认10
        minInfoGain:[0,0.1] 默认0
        impurity :["gini", "entropy"],默认”gini”
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
        impurity = CategoricalHyperparameter(
            "impurity", ["gini", "entropy"], default_value="gini")
        cs.add_hyperparameters(
            [maxDepth, minInstancesPerNode, minInfoGain, impurity])
        return cs

    def getDimension(self):
        return 4
