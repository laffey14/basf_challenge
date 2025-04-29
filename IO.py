import json
import os
import requests
from pyspark.sql.types import DoubleType, FloatType
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, retry_if_exception_type
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, explode, lit, replace

from Constants import HEADERS, COUNTRIES_URL, COUNTRIES_WRITING_PATH

@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(2)

)
def read_population(spark: SparkSession) -> DataFrame:
    """We petition through REST API the data from the countries website. we save all the data as a json to afterward just extract the data that is needed.

    Parameters
    ----------
    spark : SparkSession
        Object to be able to read data

    Raises
    ------
    RequestException
        If any exception with the request is raised, it will be retried.
    """

    try:
        response = requests.get(COUNTRIES_URL, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed after retries: {e}")

    current_path = os.getcwd()
    with open(current_path + COUNTRIES_WRITING_PATH, 'w') as f:
        json.dump(data, f, indent=4)

    population_df = spark.read.json(current_path + COUNTRIES_WRITING_PATH, multiLine=True)
    df = (population_df
            .select(explode(col("data")).alias("data"))
            .select(col("data.name").alias("country_names"), replace(col("data.population"), lit(",")).cast("double").alias("boring_population")))
    df.printSchema()
    return df