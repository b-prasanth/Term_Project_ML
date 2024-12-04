import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import env_config
from pandas_profiling import ProfileReport

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

def do_eda_profiling(data_cleaned):
    profile = ProfileReport(
        data_cleaned,
        title="EDA Report for Arrival Delay Analysis",
        explorative=True
    )

    # Save the profiling report
    profile_file = "eda_report.html"
    profile.to_file(profile_file)

    # Create a covariance heatmap using Seaborn
    numeric_data = data_cleaned.select_dtypes(include=[np.number])
    cov_matrix = numeric_data.cov()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Covariance Heatmap")

    # Save the heatmap to a base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded_heatmap = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    # Inject the heatmap into the HTML report
    heatmap_html = f"""
    <h2 style="text-align: center;">Covariance Heatmap</h2>
    <div style="text-align: center;">
        <img src="data:image/png;base64,{encoded_heatmap}" alt="Covariance Heatmap" />
    </div>
    """

    # Add the heatmap HTML content to the report
    with open(profile_file, "r+") as eda_file:
        content = eda_file.read()
        # Insert the heatmap HTML content before the end of the report
        updated_content = content.replace("</body>", heatmap_html + "</body>")
        eda_file.seek(0)
        eda_file.write(updated_content)
        eda_file.truncate()