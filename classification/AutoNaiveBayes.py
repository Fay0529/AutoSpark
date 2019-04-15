from pyspark.ml.classification import NaiveBayes
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
import sys
sys.path.append("..")
from AutoEstimator import AutoEstimator


class AutoNaiveBayes(AutoEstimator):
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
                               estimator=NaiveBayes,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getPCS(self):
        '''
        smoothing: [0.01,100] 默认1

        '''
        # Build Configuration Space which defines all parameters and their
        # ranges
        cs = ConfigurationSpace()
        smoothing = UniformFloatHyperparameter(
            "smoothing", 0.01, 100, default_value=1)

        cs.add_hyperparameters(
            [smoothing])
        return cs

    def getDimension(self):
        return 1
