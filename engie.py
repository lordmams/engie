import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, dayofyear, year, month
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_spark_session():
    try:
        spark = SparkSession.builder \
            .appName("Analyse de consommation énergétique") \
            .config("spark.mongodb.input.uri", "mongodb+srv://mamadn21:Mahamadou21!@cluster0.9mfyw6g.mongodb.net/engie.energy_data?ssl=true&replicaSet=atlas-lcptar-shard-0&authSource=admin&retryWrites=true")\
            .config("spark.mongodb.output.uri", "mongodb+srv://mamadn21:Mahamadou21!@cluster0.9mfyw6g.mongodb.net/engie.energy_data?ssl=true&replicaSet=atlas-lcptar-shard-0&authSource=admin&retryWrites=true")\
            .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
            .getOrCreate()
        return spark
    except Exception as e:
        print("Erreur lors de la création de la session Spark:", e)
        raise e
    
def load_data(spark,fallback_csv_path):
    try:
        pipeline = "{'$sample': {'size': 10000}}"
        df = spark.read.format("mongo") \
               .option("pipeline", pipeline) \
               .option("spark.mongodb.input.partitioner", "MongoSamplePartitioner") \
               .option("spark.mongodb.input.partitionerOptions.partitionSizeMB", "64") \
               .load()
    except Exception as e:
        print(f"Erreur lors de la connexion à MongoDB: {e}")
        print("Chargement des données à partir du fichier CSV de secours.")
        df = spark.read.csv(fallback_csv_path, header=True, inferSchema=True)

    df = df.withColumnRenamed("Date - Heure", "Date_Heure")
    df = df.withColumn("Date_Heure", df["Date_Heure"].cast("timestamp"))
    df = df.withColumn("Date", df["Date_Heure"].cast("date"))
    df = df.na.drop(subset=["Consommation brute totale (MW)"])
    mean_value = df.select("Consommation brute gaz (MW PCS 0°C) - GRTgaz").na.drop().agg({'Consommation brute gaz (MW PCS 0°C) - GRTgaz': 'mean'}).collect()[0][0]
    df = df.na.fill({"Consommation brute gaz (MW PCS 0°C) - GRTgaz": mean_value})
    df = df.withColumn("mouvement_social_num", when(df["mouvement_social"], 1).otherwise(0))
    df = df.withColumn("Year", year(df["Date"]))
    df = df.withColumn("Month", month(df["Date"]))
    df = df.withColumn("DayOfYear", dayofyear(df["Date"]))

    return df


def statistical_analysis(df):
    consommation_pd = df.toPandas()
    group1 = consommation_pd[consommation_pd['mouvement_social_num'] == 1]['Consommation brute totale (MW)']
    group2 = consommation_pd[consommation_pd['mouvement_social_num'] == 0]['Consommation brute totale (MW)']
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value, consommation_pd

def plot_average_consumption_per_year(df):
    avg_consumption_year = df.groupBy("Year").avg("Consommation brute totale (MW)").orderBy("Year").toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_consumption_year, x='Year', y='avg(Consommation brute totale (MW))')
    plt.title('Moyenne de la Consommation par Année')
    plt.xlabel('Année')
    plt.ylabel('Moyenne de Consommation (MW)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()

def plot_monthly_average_consumption(df):
    avg_consumption_month = df.groupBy("Month").avg("Consommation brute totale (MW)").orderBy("Month").toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_consumption_month, x='Month', y='avg(Consommation brute totale (MW))')
    plt.title('Moyenne de la Consommation par Mois')
    plt.xlabel('Mois')
    plt.ylabel('Moyenne de Consommation (MW)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()

def plot_gas_vs_electricity_consumption(df):
    consommation_pd = df.toPandas()
    consommation_pd['Date'] = pd.to_datetime(consommation_pd['Date'])
    consommation_pd.set_index('Date', inplace=True)
    consommation_pd = consommation_pd.resample('W').mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=consommation_pd, x=consommation_pd.index, y='Consommation brute gaz (MW PCS 0°C) - GRTgaz', label='Consommation de Gaz')
    sns.lineplot(data=consommation_pd, x=consommation_pd.index, y='Consommation brute électricité (MW) - RTE', label='Consommation d\'Électricité')
    plt.title('Consommation de Gaz vs Consommation d\'Électricité')
    plt.xlabel('Date')
    plt.ylabel('Consommation (MW)')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    plt.close()

def plot_heatmap_daily_hourly_consumption(df):
    consommation_pd = df.toPandas()
    consommation_pd['Hour'] = pd.to_datetime(consommation_pd['Heure']).dt.hour
    consommation_pd['DayOfWeek'] = pd.to_datetime(consommation_pd['Date']).dt.dayofweek
    pivot_table = consommation_pd.pivot_table(values='Consommation brute totale (MW)', index='Hour', columns='DayOfWeek', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='coolwarm')
    plt.title('Heatmap of Energy Consumption by Hour and Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day')
    st.pyplot(plt)
    plt.close()

def plot_smoothed_time_series(df):
    consommation_pd = df.toPandas()
    consommation_pd['Date'] = pd.to_datetime(consommation_pd['Date'])
    if consommation_pd['Date'].duplicated().any():
        consommation_pd = consommation_pd.drop_duplicates('Date')
    consommation_pd.set_index('Date', inplace=True)
    consommation_pd['Consommation_smoothed'] = consommation_pd['Consommation brute totale (MW)'].rolling(window=7).mean()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=consommation_pd, x=consommation_pd.index, y='Consommation_smoothed')
    plt.title('Smoothed Time Series of Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('Smoothed Consumption (MW)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()

def plot_correlation(df):
    consommation_pd = df.toPandas()
    conditions = [
        consommation_pd['Consommation brute gaz (MW PCS 0°C) - GRTgaz'] > consommation_pd['Consommation brute électricité (MW) - RTE'],
        consommation_pd['Consommation brute gaz (MW PCS 0°C) - GRTgaz'] <= consommation_pd['Consommation brute électricité (MW) - RTE']
    ]
    colors = ['red', 'blue']  
    consommation_pd['colors'] = np.select(conditions, colors)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Consommation brute gaz (MW PCS 0°C) - GRTgaz',
                    y='Consommation brute électricité (MW) - RTE',
                    data=consommation_pd,
                    palette=colors,
                    hue='colors',
                    legend=None)  

    plt.title('Correlation between Gas and Electricity Consumption')
    plt.xlabel('Gas Consumption (MW)')
    plt.ylabel('Electricity Consumption (MW)', color='black')
    st.pyplot(plt)
    plt.close()

def plot_monthly_boxplot(df):
    consommation_pd = df.toPandas()
    consommation_pd['Month'] = pd.to_datetime(consommation_pd['Date']).dt.month
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Month', y='Consommation brute totale (MW)', data=consommation_pd)
    plt.title('Monthly Boxplot of Energy Consumption')
    plt.xlabel('Month')
    plt.ylabel('Energy Consumption (MW)')
    st.pyplot(plt)
    plt.close()

def main():
    st.title('Analyse de Consommation Énergétique')

    tabs = st.sidebar.radio("Navigation", ["Visualisation", "Analyse"])
    fallback_csv_path = './Consomation&Mouvement.csv'
    spark = create_spark_session()
    
    df = load_data(spark, fallback_csv_path)

    if tabs == "Visualisation":
        st.subheader("Visualisation des Données")

        visualization_options = {
            "Moyenne de la Consommation par Année": plot_average_consumption_per_year,
            "Moyenne de la Consommation par Mois": plot_monthly_average_consumption,
            "Consommation de Gaz vs Consommation d'Électricité": plot_gas_vs_electricity_consumption,
            "Heatmap de la Consommation Énergétique par Heure et Jour de la Semaine": plot_heatmap_daily_hourly_consumption,
            "Consommation énergétique au fil du temps": plot_smoothed_time_series,
            "Corrélation entre la Consommation de Gaz et d'Électricité": plot_correlation,
            "Distribution Mensuelle de la Consommation Énergétique": plot_monthly_boxplot


        }

        selected_visualization = st.selectbox("Sélectionnez une visualisation prédéfinie", list(visualization_options.keys()))

        visualization_options[selected_visualization](df)

    elif tabs == "Analyse":
        st.subheader("Analyse Statistique")

        t_stat, p_value, consommation_pd = statistical_analysis(df)
        st.write("T-statistic:", t_stat)
        st.write("P-value:", p_value)
        st.write("p est inférieure au seuil prédéfini ( 0.05), alors on rejette l'hypothèse nulle, ce qui suggère que les différences entre les moyennes des groupes sont statistiquement significatives.")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='mouvement_social_num', y='Consommation brute totale (MW)', data=consommation_pd)
        plt.title('Distribution de la Consommation Énergétique par Statut de Mouvement Social')
        st.pyplot(plt)

    spark.stop()

if __name__ == "__main__":
    main()