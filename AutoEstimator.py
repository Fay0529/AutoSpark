
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import numpy as np
from numpy.linalg import *
import logging
import time
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor,\
    RandomForestRegressionModel
'''
线性回归模型自己实现
支持basePredictor的保存与直接使用
'''


class AutoEstimator(object):

    def __init__(self,
                 evaluator=None,
                 trainData=None,
                 predictorPath="basePredictors",
                 spark=None,
                 estimator=None,
                 cvParallelism=1,
                 numFolds=3):
        self._evaluator = evaluator
        self._trainData = trainData
        self._spark = spark
        self._estimator = estimator
        self._baseDataPath = "basePredictorData.txt"
        self._predictorPath = predictorPath
        self._predictorModelPath = predictorPath + "/basePredictor{0}.model"
        self._cvParallelism = cvParallelism
        self._numFolds = numFolds
        logging.basicConfig(level=logging.ERROR)

    def evaluate(self, df):
        return self._evaluator.evaluate(df)

    def getPCS(self):
        pass

    def getDimension(self):
        pass

    def getConfigurationDict(self, cfg):
        res = {k: cfg[k] for k in cfg if cfg[k]}
        return res

    def _eval(self, cfg):
        cfg = self.getConfigurationDict(cfg)
        estimator = self._estimator(**cfg)
        paramMap = estimator.extractParamMap()
        paramGrid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=estimator,
                            evaluator=self._evaluator,
                            estimatorParamMaps=paramGrid,
                            numFolds=self._numFolds,
                            parallelism=self._cvParallelism)
        cvModel = cv.fit(self._trainData, params=paramMap)
        if self._evaluator.isLargerBetter():
            return 1 - cvModel.avgMetrics[0]
        else:
            return cvModel.avgMetrics[0]

    def getBestModel(self,
                     cutoff_time=None,
                     wallclock_limit=None,
                     runcount_limit=None,
                     tuner_timeout=None):
        cs = self.getPCS()
        dic = {"run_obj": "quality", "cs": cs, "deterministic": "true"}
        if wallclock_limit:
            dic["wallclock_limit"] = wallclock_limit
        if runcount_limit:
            dic["runcount_limit"] = runcount_limit
        if tuner_timeout:
            dic["tuner_timeout"] = tuner_timeout
        scenario = Scenario(dic)
        # Optimize, using a SMAC-object
        print("Optimizing! Depending on your machine, \
            this might take a few minutes.")
        self._trainData.cache()
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=self._eval)
        cfg = smac.optimize()
        cfg = self.getConfigurationDict(cfg)
        estimator = self._estimator(**cfg)
        model = estimator.fit(self._trainData)
        self._trainData.unpersist()
        return model

    def getBestModelWithSubSampling(self,
                                    lowRatio=0.05,
                                    midRatio=0.2,
                                    taeLimit=50,
                                    basePredictorsNum=5,
                                    basePredictorsDataSizeRatio=100,
                                    cutoff_time=None,
                                    wallclock_limit=None,
                                    runcount_limit=None,
                                    tuner_timeout=None):
        self._lowRatio = lowRatio
        self._midRatio = midRatio
        self._taeLimit = taeLimit
        self._BPDS = basePredictorsDataSizeRatio * self.getDimension()
        self._basePredictors = self._getBasePredictors(num=basePredictorsNum)
        self._finalLowTrainData = self._trainData.sample(
            fraction=self._lowRatio, seed=42).cache()
        self._trainData.cache()
        self._finalPredictor = None
        if self._evaluator.isLargerBetter():
            bestEvaluation = 0
        else:
            bestEvaluation = float('inf')
        bestModel = None
        z = np.array([[]])
        y = np.array([[]])
        cs = self.getPCS()
        self._omiga = np.zeros((basePredictorsNum + 1, 1))
        dic = {"run_obj": "quality", "cs": cs, "deterministic": "true"}
        if wallclock_limit:
            dic["wallclock_limit"] = wallclock_limit
        if runcount_limit:
            dic["runcount_limit"] = runcount_limit
        if tuner_timeout:
            dic["tuner_timeout"] = tuner_timeout
        scenario = Scenario(dic)
        for i in range(taeLimit):
            # Optimize, using a SMAC-object
            smac = SMAC(scenario=scenario, rng=np.random.RandomState(i + 1),
                        tae_runner=self._finalEval)
            cfg = smac.optimize()
            cfg = self.getConfigurationDict(cfg)
            cfgList = [cfg[k] for k in sorted(cfg.keys())]
            estimator = self._estimator(**cfg)
            paramMap = estimator.extractParamMap()
            paramGrid = ParamGridBuilder().build()
            cv = CrossValidator(estimator=estimator,
                                evaluator=self._evaluator,
                                estimatorParamMaps=paramGrid,
                                numFolds=self._numFolds,
                                parallelism=self._cvParallelism)
            lowModel = cv.fit(self._finalLowTrainData, params=paramMap)
            hModel = cv.fit(self._trainData, params=paramMap)
            if self._evaluator.isLargerBetter():
                if hModel.avgMetrics[0] > bestEvaluation:
                    bestModel = hModel.bestModel
                    bestEvaluation = hModel.avgMetrics[0]
            elif hModel.avgMetrics[0] < bestEvaluation:
                bestModel = hModel.bestModel
                bestEvaluation = hModel.avgMetrics[0]
            transformData = self._getBaseTransformedData(cfgList)
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
        self._finalLowTrainData.unpersist()
        self._trainData.unpersist()
        return bestModel

    def _baseEval(self, cfg):
        '''
        objective function is middle fidelity function
        '''

        cfg = self.getConfigurationDict(cfg)
        cfgList = [cfg[k] for k in sorted(cfg.keys())]
        estimator = self._estimator(**cfg)
        midM = estimator.fit(self._midTrainData)
        lowM = estimator.fit(self._lowTrainData)
        midEva = self.evaluate(midM.transform(self._midTrainData))
        lowEva = self.evaluate(lowM.transform(self._lowTrainData))
        f = open(self._baseDataPath, "a")
        self._printLabelFeatures(
            midEva - lowEva,
            cfgList,
            f)
        f.close()
        if self._evaluator.isLargerBetter():
            return 1 - midEva
        else:
            return midEva

    def _finalEval(self, cfg):

        cfg = self.getConfigurationDict(cfg)
        cfgList = [cfg[k] for k in sorted(cfg.keys())]
        estimator = self._estimator(**cfg)
        lowModel = estimator.fit(self._finalLowTrainData)
        lowEva = self.evaluate(lowModel.transform(self._finalLowTrainData))
        if self._evaluator.isLargerBetter():
            return 1 - (lowEva + self._getFinalTransformedData(cfgList))
        else:
            return lowEva + self._getFinalTransformedData(cfgList)

    def _getBaseTransformedData(self, cfgList):
        res = []
        df = self._spark.createDataFrame([(Vectors.dense(*cfgList),)]
                                         ).toDF("features")
        for x in self._basePredictors:
            res.append(x.transform(df).head().prediction)
        res.append(1)
        return res

    def _getFinalTransformedData(self, cfgList):
        baseTransformedData = self._getBaseTransformedData(cfgList)
        z = np.array(baseTransformedData).reshape((1, -1))
        return z.dot(self._omiga)[0, 0]

    def _getBasePredictor(self, randomSeed):
        f = open(self._baseDataPath, "w")
        f.truncate()
        f.close()
        self._lowTrainData = self._trainData.sample(
            fraction=self._lowRatio, seed=randomSeed).cache()
        self._midTrainData = self._trainData.sample(
            fraction=self._midRatio, seed=randomSeed).cache()
        cs = self.getPCS()
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": self._BPDS,
                             "cs": cs,
                             "deterministic": "true"
                             })
        # Optimize, using a SMAC-object
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=self._baseEval)
        smac.optimize()
        df = self._spark.read.format("libsvm").load(self._baseDataPath)
        rf = RandomForestRegressor()
        rfModel = rf.fit(df)
        self._lowTrainData.unpersist()
        self._midTrainData.unpersist()
        return rfModel

    def _getBasePredictors(self, num=5):
        res = []
        if not os.path.exists(self._predictorPath):
            os.mkdir(self._predictorPath)
        if os.listdir(self._predictorPath):
            for i in range(num):
                res.append(RandomForestRegressionModel.load(
                    self._predictorModelPath.format(i)))
        else:
            for i in range(num):
                model = self._getBasePredictor(i)
                res.append(model)
                model\
                    .write()\
                    .overwrite().\
                    save(self._predictorModelPath.format(i))
        return res

    def _printLabelFeatures(self, label, features, file):
        buff = "%s" % label
        for i in range(len(features)):
            buff += " {0}:{1}".format(i + 1, features[i])
        print(buff, file=file)
