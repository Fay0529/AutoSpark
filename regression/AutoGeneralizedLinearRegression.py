
from pyspark.ml.regression import GeneralizedLinearRegression
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter, CategoricalHyperparameter
from pyspark.ml.evaluation import RegressionEvaluator
from ConfigSpace.conditions import InCondition
import sys
sys.path.append("..")
from AutoEstimator import AutoEstimator


class AutoGeneralizedLinearRegression(AutoEstimator):
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
                               estimator=GeneralizedLinearRegression,
                               cvParallelism=cvParallelism,
                               numFolds=numFolds)

    def getConfigurationDict(self, cfg):
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        family = cfg["family"]
        if family == "gaussian":
            cfg["link"] = cfg["gaussianLink"]
            cfg.pop("gaussianLink", None)
        elif family == "poisson":
            cfg["link"] = cfg["poissonLink"]
            cfg.pop("poissonLink", None)
        return cfg

    def getPCS(self):
        '''
        maxIter: [1,100]最大迭代次数,默认50
        regParam :[0,0.2] 正则化参数，默认0
        tol:[1e-6,1e-1] 迭代算法收敛性，默认 1e-6
        family ,link, variancePower 对应关系
        •   “gaussian” -> “identity”, “log”, “inverse”
        •   “binomial” -> “logit”, “probit”, “cloglog”
        •   “poisson” -> “log”, “identity”, “sqrt”
        •   “gamma” -> “inverse”, “identity”, “log”
        •   “tweedie” -> power link function specified through “linkPower”.
        The default link power in the tweedie family is 1 - variancePower.


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
        family = CategoricalHyperparameter(
            "family",
            ["gaussian", "poisson"],
            default_value="gaussian")
        gaussianLink = CategoricalHyperparameter(
            "gaussianLink",
            ["identity", "log", "inverse"],
            default_value="identity")
        poissonLink = CategoricalHyperparameter(
            "poissonLink",
            ["log", "identity", "sqrt"],
            default_value="log")
        cs.add_hyperparameters(
            [maxIter, regParam, tol, family, gaussianLink,
                poissonLink])
        cs.add_condition(InCondition(
            child=gaussianLink, parent=family, values=["gaussian"]))
        cs.add_condition(InCondition(
            child=poissonLink, parent=family, values=["poisson"]))
        return cs

    def getDimension(self):
        return 6
