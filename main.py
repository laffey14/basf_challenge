from os import truncate

from Constants import DATA_READING_PATH
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, schema_of_json
import os
import json

from IO import read_population
from math_calc import normalize_columns, normal_skew_kur_analysis, calculate_statistics


def get_spark_session():
    return (SparkSession.builder.appName("API Fetch Example")
            .config("spark.executor.memory", "6g")
            .config("spark.driver.memory", "6g")
            .getOrCreate())

def main():
    #Initilizing spark session
    spark = get_spark_session()

    #Retrieving population data
    population_df = read_population(spark)
    population_df.show(truncate=False)

    #Retrieving csv with data
    data_df = spark.read.csv(os.getcwd() + DATA_READING_PATH, header=True, inferSchema=True)
    data_df.show(truncate=False)

    #Calculate statistics from data
    path_statistics = calculate_statistics(data_df.drop(data_df.columns[0]))

    #We normalize all the numeric fields in the data. We remove the first column
    data_normalized_df = normalize_columns(spark, data_df.drop(data_df.columns[0]), path_statistics)
    data_normalized_df.show(truncate=False)

    #We make a kurtosis and skewness analysis to determine if it's a gaussian distribution
    normal_analysis = normal_skew_kur_analysis(spark, data_normalized_df, path_statistics)
    normal_analysis.show(500, truncate=False)

    #We join data with the population data
    countries_data_df = (data_df.join(population_df, "country_names")
                         .select(col("country_names"), col("boring_population")))
    countries_data_df.show(truncate=False)
    print("countries_data_df count: ", countries_data_df.count())

    # Calculate statistics for population
    path_statistics = calculate_statistics(countries_data_df, ["boring_population"])
    spark.read.parquet(path_statistics).show( truncate=False)

    #We normalize all the numeric fields in the data
    population_normalized_df = normalize_columns(spark, countries_data_df, path_statistics)
    population_normalized_df.show(truncate=False)

    #We make a kurtosis and skewness analysis to determine if it's a gaussian distribution
    normal_analysis = normal_skew_kur_analysis(spark, population_normalized_df, path_statistics)
    normal_analysis.show(500, truncate=False)




if __name__ == "__main__":
    main()
