import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_data(data_cleaned):
    # Aggregation: Average delay by time of the day
    avg_delay_by_station = data_cleaned.groupby('time_of_day')['delay_minutes'].mean().reset_index()
    print("Average delay by time of the day\n",avg_delay_by_station.sort_values(by='time_of_day', ascending=False))
    # plt.bar(avg_delay_by_station.sort_values(by='time_of_day', ascending=False))
    avg_delay_by_station.sort_values(by='time_of_day', ascending=True).plot(kind='bar')
    plt.grid(axis='y')
    plt.xticks(ticks=[0,1,2,3],labels=["Morning","Afternoon","Evening","Night"], rotation=0)
    plt.xlabel("Time of Day")
    plt.ylabel("Average Delay")
    plt.show()

    # Aggregation: Average delay by date
    avg_delay_by_date = data_cleaned.groupby('date')['delay_minutes'].mean().reset_index()
    print("\nAverage delay by date\n",avg_delay_by_date)
    ticks = list(range(0, 31))
    ticks_labels=list(range(1, 32))
    avg_delay_by_date.sort_values(by='date', ascending=True).plot(kind='bar')
    plt.grid(axis='y')
    plt.xticks(ticks=ticks, labels=ticks_labels)
    plt.xlabel("Date (Mar-2018)")
    plt.ylabel("Average Delay")
    plt.show()

def do_eda_graphs(cleaned_dataset):
    cleaned_dataset['long_delay'] = cleaned_dataset['delay_minutes'] > 5
    cleaned_dataset.groupby('line')['long_delay'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xticks(rotation=45, fontsize='xx-small')
    plt.grid(axis='y')
    plt.show()

    ax = cleaned_dataset.groupby('stop_sequence')["delay_minutes"].mean().plot()
    ax.set_ylabel("average delay_minutes")
    cleaned_dataset.date = pd.to_datetime(cleaned_dataset.date)
    x = cleaned_dataset.groupby('date')['delay_minutes'].mean()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.autofmt_xdate()
    plt.grid()
    ax.plot(x)
    ax.set_ylabel('average delay_minutes')
    plt.show()

    cleaned_dataset['date'] = pd.to_datetime(cleaned_dataset['date'])
    cleaned_dataset['day_of_weeks'] = cleaned_dataset['date'].dt.day_name()
    cleaned_dataset['date'] = pd.to_datetime(cleaned_dataset['date'])
    cleaned_dataset['day_of_weeks'] = cleaned_dataset['date'].dt.day_name()

    week_av_delay = cleaned_dataset.groupby(cleaned_dataset['day_of_weeks'], as_index=False)['delay_minutes'].mean()
    week_av_delay['num'] = [5, 1, 6, 7, 4, 2, 3]
    week_av_delay = cleaned_dataset.groupby(cleaned_dataset['day_of_weeks'], as_index=False)['delay_minutes'].mean()
    week_av_delay['num'] = [5, 1, 6, 7, 4, 2, 3]
    print("\nAverage delay on each day of the week\n", week_av_delay.sort_values(by='num',ascending=True))