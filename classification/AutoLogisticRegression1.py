
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import numpy as np
from numpy.linalg import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import logging
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor,\
    RandomForestRegressionModel
'''
线性回归模型自己实现
支持basePredictor的保存与直接使用
'''


class AutoLogisticRegression(object):

    def __init__(self, evaluator=None, trainData=None,
                 predictorPath="basePredictors", spark=None):
        self._evaluator = evaluator
        self._trainData = trainData
        self._spark = spark
        self._estimator = LogisticRegression
        self._baseDataPath = "basePredictorData.txt"
        self._predictorPath = predictorPath
        self._predictorModelPath = predictorPath + "/basePredictor{0}.model"

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

    def __eval(self, cfg):
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        estimator = self._estimator(**cfg)
        paramMap = estimator.extractParamMap()
        paramGrid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=estimator, evaluator=self._evaluator,
                            estimatorParamMaps=paramGrid, numFolds=3)
        cvModel = cv.fit(self._trainData, params=paramMap)
        return 1 - cvModel.avgMetrics[0]

    def getBestModel(self, **kw):
        cs = self.getPCS()
        if "run_obj" not in kw:
            kw["run_obj"] = "quality"
        kw["cs"] = cs
        kw["deterministic"] = "true"
        scenario = Scenario(kw)
        # Optimize, using a SMAC-object
        print("Optimizing! Depending on your machine, \
            this might take a few minutes.")
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=self.__eval)
        cfg = smac.optimize()
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        print("best config by calling getBestModel: ", cfg)
        estimator = self._estimator(**cfg)
        model = estimator.fit(self._trainData)
        return model

    def getBestModelWithSubSampling(self, lowRatio=0.05, midRatio=0.2,
                                    taeLimit=50, basePredictorsNum=5, **kw):
        self._lowRatio = lowRatio
        self._midRatio = midRatio
        self._taeLimit = taeLimit
        self._basePredictors = self.__getBasePredictors(num=basePredictorsNum)
        self._finalLowTrainData = self._trainData.sample(
            fraction=self._lowRatio, seed=42)
        self._finalPredictor = None
        bestEvaluation = 0
        bestModel = None
        z = np.array([[]])
        y = np.array([[]])
        cs = self.getPCS()
        self._omiga = np.zeros((basePredictorsNum + 1, 1))
        if "run_obj" not in kw:
            kw["run_obj"] = "quality"
        kw["cs"] = cs
        kw["deterministic"] = "true"
        scenario = Scenario(kw)
        for i in range(taeLimit):
            # Optimize, using a SMAC-object
            smac = SMAC(scenario=scenario, rng=np.random.RandomState(i + 1),
                        tae_runner=self.__finalEval)
            cfg = smac.optimize()
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            cfgList = [cfg[k] for k in sorted(cfg.keys())]
            estimator = self._estimator(**cfg)
            paramMap = estimator.extractParamMap()
            paramGrid = ParamGridBuilder().build()
            cv = CrossValidator(estimator=estimator, evaluator=self._evaluator,
                                estimatorParamMaps=paramGrid, numFolds=3)
            lowModel = cv.fit(self._finalLowTrainData, params=paramMap)
            hModel = cv.fit(self._trainData, params=paramMap)
            if hModel.avgMetrics[0] > bestEvaluation:
                bestModel = hModel.bestModel
                bestEvaluation = hModel.avgMetrics[0]
            transformData = self.__getBaseTransformedData(cfgList)
            if not z.any():
                z = np.array(transformData).reshape(
                    (-1, basePredictorsNum + 1))
            else:
                z = np.row_stack((z, transformData))
            if not y.any():
                y = np.array(hModel.avgMetrics[
                             0] - lowModel.avgMetrics[0]).reshape((-1, 1))
            else:
                y = np.row_stack(
                    (y, hModel.avgMetrics[0] - lowModel.avgMetrics[0]))
            self._omiga = pinv(z.T.dot(z)).dot(z.T).dot(y)
        return bestModel

    def __baseEval(self, cfg):
        '''
        objective function is middle fidelity function
        '''

        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        cfgList = [cfg[k] for k in sorted(cfg.keys())]
        estimator = self._estimator(**cfg)
        paramMap = estimator.extractParamMap()
        paramGrid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=estimator, evaluator=self._evaluator,
                            estimatorParamMaps=paramGrid, numFolds=3)
        midModel = cv.fit(self._midTrainData, params=paramMap)
        lowModel = cv.fit(self._lowTrainData, params=paramMap)
        f = open(self._baseDataPath, "a")
        self.__printLabelFeatures(
            midModel.avgMetrics[0] - lowModel.avgMetrics[0], cfgList, f)
        f.close()
        return 1 - midModel.avgMetrics[0]

    def __finalEval(self, cfg):

        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        cfgList = [cfg[k] for k in sorted(cfg.keys())]
        estimator = self._estimator(**cfg)
        paramMap = estimator.extractParamMap()
        paramGrid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=estimator, evaluator=self._evaluator,
                            estimatorParamMaps=paramGrid, numFolds=3)
        lowModel = cv.fit(self._finalLowTrainData, paramMap)
        return 1 - (lowModel.avgMetrics[0] +
                    self.__getFinalTransformedData(cfgList))

    def __getBaseTransformedData(self, cfgList):
        res = []
        df = self._spark.createDataFrame([(Vectors.dense(*cfgList),)]
                                         ).toDF("features")
        for x in self._basePredictors:
            res.append(x.transform(df).head().prediction)
        res.append(1)
        return res

    def __getFinalTransformedData(self, cfgList):
        baseTransformedData = self.__getBaseTransformedData(cfgList)
        z = np.array(baseTransformedData).reshape((1, -1))
        return z.dot(self._omiga)[0, 0]

    def __getBasePredictor(self, randomSeed):
        f = open(self._baseDataPath, "w")
        f.truncate()
        f.close()
        self._lowTrainData = self._trainData.sample(
            fraction=self._lowRatio, seed=randomSeed)
        self._midTrainData = self._trainData.sample(
            fraction=self._midRatio, seed=randomSeed)
        cs = self.getPCS()
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": 80,
                             "cs": cs,
                             "deterministic": "true",
                             })
        # Optimize, using a SMAC-object
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=self.__baseEval)
        smac.optimize()
        df = self._spark.read.format("libsvm").load(self._baseDataPath)
        rf = RandomForestRegressor()
        rfModel = rf.fit(df)
        return rfModel

    def __getBasePredictors(self, num=5):
        res = []
        if not os.path.exists(self._predictorPath):
            os.mkdir(self._predictorPath)
        if os.listdir(self._predictorPath):
            for i in range(num):
                res.append(RandomForestRegressionModel.load(
                    self._predictorModelPath.format(i)))
        else:
            for i in range(num):
                model = self.__getBasePredictor(i)
                res.append(model)
                model\
                    .write()\
                    .overwrite().\
                    save(self._predictorModelPath.format(i))
        return res

    def __printLabelFeatures(self, label, features, file):
        buff = "%s" % label
        for i in range(len(features)):
            buff += " {0}:{1}".format(i + 1, features[i])
        print(buff, file=file)

if __name__ == "__main__":
    os.environ["PYSPARK_PYTHON"] = "python3"
    logging.basicConfig(level=logging.INFO)
    spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()
    sc = spark.sparkContext
    datapath = "/home/ubuntu/Documents/auto-spark\
        /data/sample_multiclass_classification_data.txt"
    training = spark.read \
        .format("libsvm") \
        .load("../data/sample_multiclass_classification_data.txt")
    evaluator = MulticlassClassificationEvaluator()
    autoLr = AutoLogisticRegression(
        trainData=training, evaluator=evaluator, spark=spark)
    model = autoLr.getBestModel(ta_run_limit=100)
    print(evaluator.evaluate(model.transform(training)))
    # model2 = autoLr.getBestModelWithSubSampling(
    #     lowRatio=0.3, midRatio=0.8, taeLimit=30, runcount_limit=50)
    # print(evaluator.evaluate(model2.transform(training)))
