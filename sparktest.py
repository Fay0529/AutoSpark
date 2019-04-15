from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import os
os.environ["PYSPARK_PYTHON"] = "python3"
spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()
sc = spark.sparkContext

# Load training data
dataset = spark.read.format("libsvm")\
    .load("data/sample_linear_regression_data.txt")

glr = GeneralizedLinearRegression(
    family="gaussian", link="identity", maxIter=10, regParam=0.3)
paramMap = glr.extractParamMap()
print("333333333333333333", paramMap)
paramGrid = ParamGridBuilder().build()
cv = CrossValidator(estimator=glr, evaluator=RegressionEvaluator(),
                    estimatorParamMaps=paramGrid, numFolds=3)
cvModel = cv.fit(self._trainData, params=paramMap)
# Fit the model
model = glr.fit(dataset)

# Print the coefficients and intercept for generalized linear regression model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " +
      str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()
