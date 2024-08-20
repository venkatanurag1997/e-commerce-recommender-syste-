import os
import sys
import logging
from typing import List, Tuple, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import col, expr, when, rand, monotonically_increasing_id
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.pipeline import Pipeline
import yaml
import mlflow
import mlflow.spark
from delta import DeltaTable
from pyspark.sql.window import Window
import pyspark.sql.functions as F

class AdvancedRecommenderSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()
        self.logger = self._setup_logging()
        self._setup_mlflow()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            sys.exit(f"Failed to load configuration: {str(e)}")

    def _create_spark_session(self) -> SparkSession:
        return (SparkSession.builder
                .appName(self.config['app_name'])
                .config("spark.executor.memory", self.config['spark_executor_memory'])
                .config("spark.driver.memory", self.config['spark_driver_memory'])
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .getOrCreate())

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.config['mlflow_tracking_uri'])
        mlflow.set_experiment(self.config['mlflow_experiment_name'])

    def load_data(self) -> DataFrame:
        schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("item_id", StringType(), True),
            StructField("rating", FloatType(), True),
            StructField("timestamp", IntegerType(), True)
        ])
        try:
            data = self.spark.read.schema(schema).parquet(self.config['data_path'])
            self.logger.info(f"Loaded {data.count()} records from {self.config['data_path']}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def preprocess_data(self, data: DataFrame) -> Tuple[DataFrame, DataFrame, Dict[str, StringIndexer]]:
        try:
            # Convert string columns to numeric
            indexers = {
                col: StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
                for col in ["user_id", "item_id"]
            }
            
            pipeline = Pipeline(stages=list(indexers.values()))
            indexed_data = pipeline.fit(data).transform(data)

            # Add additional features
            indexed_data = indexed_data.withColumn("day_of_week", F.dayofweek(F.from_unixtime("timestamp")))
            indexed_data = indexed_data.withColumn("hour", F.hour(F.from_unixtime("timestamp")))

            # Normalize ratings
            w = Window.partitionBy("user_id")
            indexed_data = indexed_data.withColumn(
                "normalized_rating",
                (col("rating") - F.avg("rating").over(w)) / F.stddev("rating").over(w)
            )

            # Split data
            train, test = indexed_data.randomSplit(self.config['train_test_split'], seed=42)
            self.logger.info(f"Data split into {train.count()} training and {test.count()} testing records")
            return train, test, indexers
        except Exception as e:
            self.logger.error(f"Failed to preprocess data: {str(e)}")
            raise

    def train_model(self, train_data: DataFrame) -> ALSModel:
        try:
            als = ALS(
                maxIter=self.config['als_max_iter'],
                regParam=self.config['als_reg_param'],
                userCol=f"{self.config['user_col']}_idx",
                itemCol=f"{self.config['item_col']}_idx",
                ratingCol="normalized_rating",
                coldStartStrategy="drop"
            )

            param_grid = ParamGridBuilder() \
                .addGrid(als.rank, self.config['als_rank_values']) \
                .addGrid(als.regParam, self.config['als_reg_param_values']) \
                .build()

            evaluator = RegressionEvaluator(
                metricName="rmse", 
                labelCol="normalized_rating", 
                predictionCol="prediction"
            )

            cv = CrossValidator(
                estimator=als,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3
            )

            with mlflow.start_run():
                model = cv.fit(train_data)
                mlflow.log_params({
                    "best_rank": model.bestModel._java_obj.parent().getRank(),
                    "best_regParam": model.bestModel._java_obj.parent().getRegParam()
                })
                mlflow.spark.log_model(model.bestModel, "als_model")

            self.logger.info("Model training completed")
            return model.bestModel
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            raise

    def evaluate_model(self, model: ALSModel, test_data: DataFrame) -> Dict[str, float]:
        try:
            predictions = model.transform(test_data)
            reg_evaluator = RegressionEvaluator(
                metricName="rmse", 
                labelCol="normalized_rating", 
                predictionCol="prediction"
            )
            rmse = reg_evaluator.evaluate(predictions)

            # Convert predictions to binary classification problem
            binary_predictions = predictions.withColumn(
                "liked",
                when(col("normalized_rating") > 0, 1).otherwise(0)
            ).withColumn(
                "predicted_liked",
                when(col("prediction") > 0, 1).otherwise(0)
            )

            binary_evaluator = BinaryClassificationEvaluator(
                rawPredictionCol="prediction",
                labelCol="liked"
            )

            auc = binary_evaluator.evaluate(binary_predictions)

            metrics = {"RMSE": rmse, "AUC": auc}
            self.logger.info(f"Model Metrics: {metrics}")
            
            with mlflow.start_run(nested=True):
                mlflow.log_metrics(metrics)

            return metrics
        except Exception as e:
            self.logger.error(f"Failed to evaluate model: {str(e)}")
            raise

    def generate_recommendations(self, model: ALSModel, users: List[int], n: int) -> DataFrame:
        try:
            user_df = self.spark.createDataFrame([(user,) for user in users], ["user_id_idx"])
            recommendations = model.recommendForUserSubset(user_df, n)
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
            raise

    def train_click_model(self, recommendations: DataFrame, actual_clicks: DataFrame) -> RandomForestClassifier:
        try:
            # Join recommendations with actual clicks
            joined_data = recommendations.join(actual_clicks, ["user_id", "item_id"], "left_outer")
            joined_data = joined_data.withColumn("clicked", when(col("click") == 1, 1).otherwise(0))

            # Prepare features
            assembler = VectorAssembler(
                inputCols=["prediction", "rank", "day_of_week", "hour"],
                outputCol="features"
            )
            prepared_data = assembler.transform(joined_data)

            # Train Random Forest model
            rf = RandomForestClassifier(labelCol="clicked", featuresCol="features", numTrees=100)
            rf_model = rf.fit(prepared_data)

            self.logger.info("Click model training completed")
            return rf_model
        except Exception as e:
            self.logger.error(f"Failed to train click model: {str(e)}")
            raise

    def simulate_ab_test(self, recommendations: DataFrame, click_model: RandomForestClassifier, 
                         control_group: List[int], test_group: List[int]) -> float:
        try:
            # Add random assignment to control/test groups
            recommendations = recommendations.withColumn(
                "group",
                when(col("user_id").isin(control_group), "control")
                .when(col("user_id").isin(test_group), "test")
                .otherwise("other")
            )

            # Prepare features for click prediction
            assembler = VectorAssembler(
                inputCols=["prediction", "rank", "day_of_week", "hour"],
                outputCol="features"
            )
            prepared_data = assembler.transform(recommendations)

            # Predict clicks
            predictions = click_model.transform(prepared_data)

            # Calculate CTR for control and test groups
            group_ctrs = predictions.groupBy("group").agg(F.avg("prediction").alias("ctr"))
            control_ctr = group_ctrs.filter(col("group") == "control").select("ctr").first()[0]
            test_ctr = group_ctrs.filter(col("group") == "test").select("ctr").first()[0]

            lift = (test_ctr - control_ctr) / control_ctr
            self.logger.info(f"Estimated lift in CTR: {lift * 100:.2f}%")
            return lift
        except Exception as e:
            self.logger.error(f"Failed to simulate A/B test: {str(e)}")
            raise

    def save_model(self, model: ALSModel, path: str):
        try:
            model.save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    def update_recommendations(self, recommendations: DataFrame):
        try:
            # Assuming recommendations are stored in a Delta table
            delta_table = DeltaTable.forPath(self.spark, self.config['recommendations_table_path'])

            # Perform a merge operation
            delta_table.alias("old").merge(
                recommendations.alias("new"),
                "old.user_id = new.user_id AND old.item_id = new.item_id"
            ).whenMatchedUpdate(set={
                "prediction": "new.prediction",
                "rank": "new.rank",
                "timestamp": F.current_timestamp()
            }).whenNotMatchedInsert(values={
                "user_id": "new.user_id",
                "item_id": "new.item_id",
                "prediction": "new.prediction",
                "rank": "new.rank",
                "timestamp": F.current_timestamp()
            }).execute()

            self.logger.info("Recommendations updated in Delta table")
        except Exception as e:
            self.logger.error(f"Failed to update recommendations: {str(e)}")
            raise

    def run(self):
        try:
            with mlflow.start_run():
                data = self.load_data()
                train_data, test_data, indexers = self.preprocess_data(data)
                model = self.train_model(train_data)
                metrics = self.evaluate_model(model, test_data)
                
                # Generate recommendations for all users
                all_users = train_data.select(f"{self.config['user_col']}_idx").distinct()
                recommendations = self.generate_recommendations(model, all_users.collect(), self.config['n_recommendations'])
                
                # Simulate clicks (in a real scenario, you would use actual click data)
                simulated_clicks = recommendations.withColumn("click", (rand() > 0.9).cast("integer"))
                
                click_model = self.train_click_model(recommendations, simulated_clicks)
                lift = self.simulate_ab_test(recommendations, click_model, self.config['control_group'], self.config['test_group'])
                
                self.save_model(model, self.config['model_save_path'])
                self.update_recommendations(recommendations)
                
                mlflow.log_metric("lift", lift)
        finally:
            self.spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    recommender = AdvancedRecommenderSystem(config_path)
    recommender.run()
