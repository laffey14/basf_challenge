from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, kurtosis, lit, mean, skewness, std, when
from pyspark.sql import Window
from pyspark.sql.types import DoubleType, FloatType, IntegerType
import pandas as pd
import numpy as np
import os
from scipy import stats
from Constants import STATISTICS_WRITING_PATH


'''
def normalize_inefficient_columns(df: DataFrame) -> DataFrame:
    window = Window.partitionBy("dummy")
    column_lists = [c for c in df.columns if c != "country_names"]
    print("Number of columns: " + str(len(column_lists)))
    df = df.withColumn("dummy", lit("a"))
    i = 1
    for column_name in column_lists:
        print("Iteration: " + str(i))
        df = (df
              .withColumn(column_name + "_mean", mean(col(column_name)).over(window))
              .withColumn(column_name + "_std", std(col(column_name)).over(window))
              .withColumn(column_name, (col(column_name) - col(column_name + "_mean")) / col(column_name + "_std"))
              .drop(column_name + "_mean", column_name + "_std")
              )
        i = i + 1
    df = df.drop("dummy")
    return df
'''

def normalize_columns(spark: SparkSession, df: DataFrame, statistics_path: str, fields: list[str] = []) -> DataFrame:
    """We normalize all the numeric fields or the ones specified

    Parameters
    ----------
    spark : SparkSession
        object to be able to read
    df: DataFrame
        data with the fields
    statistics_path: str
        path to the statistics calculated previously


    """
    if fields:
        numeric_cols = fields
    else:
        numeric_cols = [field.name for field in df.schema.fields if
                        field.dataType in [IntegerType(), DoubleType(), FloatType()]]

    calculation_df = spark.read.parquet(statistics_path)
    mean_std_columns = [f.name for f in calculation_df.schema.fields if "mean" in f.name or "stddev" in f.name]

    mean_std_df = calculation_df.select(*mean_std_columns)

    df_extra = df.crossJoin(mean_std_df)

    select_expr = ["country_names"] + [((col(c) - col(c + "_mean")) / col(c + "_stddev")).alias(c) for c in numeric_cols]

    return df_extra.select(*select_expr)


def normal_skew_kur_analysis(spark: SparkSession, df: DataFrame, statistics_path: str, fields: list[str] = []) -> DataFrame:
    """We create a report table with each field with its kurtosis and skewness and whether is gaussian

    Parameters
    ----------
    spark : SparkSession
        object to be able to read
    df: DataFrame
        data with the fields
    statistics_path: str
        path to the statistics calculated previously


    """

    calculation_df = spark.read.parquet(statistics_path)

    first_boring_column = [f.name for f in df.schema.fields if f.name.startswith("boring")][0]
    skewness_kurtosis_df = calculation_df.select(lit(first_boring_column).alias("column_name"),
                                                 col(first_boring_column + "_skewness").alias("skewness"),
                                                 col(first_boring_column + "_kurtosis").alias("kurtosis"))
    if len(df.schema.fields) > 1:
        for c in [f.name for f in df.schema.fields if f.name.startswith("boring")][1:]:
            skewness_kurtosis_df = skewness_kurtosis_df.union(
                calculation_df.select(lit(c).alias("column_name"), col(c + "_skewness").alias("skewness"),
                                  col(c + "_kurtosis").alias("kurtosis"))
            )

    normal_df = skewness_kurtosis_df.withColumn("normal", when(
        col("skewness").between(-0.5,0.5) & col("kurtosis").between(-1,1), lit(True)
    ).otherwise(lit(False)))

    return normal_df

def calculate_statistics(data: DataFrame, fields: list[str] = []) -> ():
    """We create a one row table with the following metrics for each column: mean, std, kurtosis and skewness

    Parameters
    ----------
    data : DataFrame
        table with the data
    fields: list of strings, optional parameter
        columns to process


    """
    if fields:
        numeric_cols = fields
    else:
        numeric_cols = [field.name for field in data.schema.fields if
                        field.dataType in [IntegerType(), DoubleType(), FloatType()]]

    print("Columns to process: ", str(numeric_cols))
    agg_exprs = []
    for c in numeric_cols:
        agg_exprs += [
            mean(c).alias(f"{c}_mean"),
            std(c).alias(f"{c}_stddev"),
            skewness(c).alias(f"{c}_skewness"),
            kurtosis(c).alias(f"{c}_kurtosis")
        ]

    calculation_df = data.agg(*agg_exprs)
    calculation_df.write.mode("overwrite").parquet(os.getcwd() + STATISTICS_WRITING_PATH)
    return os.getcwd() + STATISTICS_WRITING_PATH