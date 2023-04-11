import logging
import os
from typing import Optional, Union, Any
import pandas as pd
import matplotlib.pyplot as plt
import glob
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Generate folder
folders_count = len(glob.glob("autogluon_output/*"))
folders_count = 0
output_folder = os.path.join("autogluon_output", f"results_{folders_count}")

try:
    os.mkdir(output_folder)
except IOError:
    pass


def get_dataframe_from_csv(source: str) -> pd.DataFrame:
    df = pd.read_csv(source)
    df['group'] = ['m1'] * len(df)
    return df


def save_raw_input_to_png(df: pd.DataFrame, file_name: str):
    plt.figure(figsize=(20, 10))
    plt.plot(df['timestamp'], df['value'])
    plt.savefig(file_name)
    plt.clf()


def get_training_df(source_df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    test_set_size = int(len(source_df) * ratio)
    frame = source_df.iloc[:test_set_size]
    return frame


def plot_results(idf: pd.DataFrame, tdf: pd.DataFrame, y_pred: Any, show_full_original: bool):
    ver_dates = idf['timestamp'][len(tdf):]

    if show_full_original:
        plt.plot(idf['timestamp'], idf['value'], label=f"Original data")
    else:
        plt.plot(ver_dates, idf['value'][len(tdf):], label=f"Original data")

    plt.plot(ver_dates, y_pred['mean'], color='orange', alpha=0.5, label=f"Expected values")

    plt.fill_between(
        ver_dates, y_pred["0.1"], y_pred["0.9"], color="red", alpha=0.1, label=f"10%-90% confidence interval"
    )

    plt.fill_between(
        ver_dates, y_pred["0.2"], y_pred["0.7"], color="orange", alpha=0.2, label=f"20%-70% confidence interval"
    )
    plt.fill_between(
        ver_dates, y_pred["0.4"], y_pred["0.6"], color="green", alpha=0.5, label=f"40%-60% confidence interval"
    )
    plt.legend()


def fit_and_print(
        hp: Optional[dict[Union[str, type], Any]] = None,
        idf: pd.DataFrame = None,
        tdf: pd.DataFrame = None,
        output_file_name: str = None,
        presets = "medium_quality"
):
    train_data = TimeSeriesDataFrame.from_data_frame(
        training_df,
        id_column="group",
        timestamp_column="timestamp"
    )

    predictor = TimeSeriesPredictor(
        prediction_length=len(input_df) - len(training_df),
        path="synthetic_test",
        target="value",
        eval_metric="sMAPE",
    )

    if hp is None:
        predictor.fit(
            train_data,
            presets=presets,
            time_limit=6000
        )
    else:
        predictor.fit(
            train_data,
            presets=presets,
            time_limit=600,
            hyperparameters=hp
        )

    predictions = predictor.predict(train_data)
    plt.figure(figsize=(20, 10))

    y_pred = predictions.loc['m1']

    plot_results(idf, tdf, y_pred, True)
    plt.savefig(os.path.join(output_folder, "full_" + output_file_name))
    plt.clf()

    plot_results(idf, tdf, y_pred, False)
    plt.savefig(os.path.join(output_folder, output_file_name))
    plt.clf()


input_df = get_dataframe_from_csv(os.path.join('data', 'synthetic_daily_sin_1_hours.csv'))
save_raw_input_to_png(input_df, os.path.join(output_folder, "data_raw.png"))
training_df = get_training_df(input_df, 0.8)

models = [
    # ('Naive', None),
    # ('SeasonalNaive', None),
    # ('ARIMA', None),
    # ('AutoARIMA', None),
    # ('ETS', None),
    # ('AutoETS', {'AutoETS': {'model': 'AAA'}}),
    # ('Theta', None),
    # ('DynamicOptimizedTheta', {'DynamicOptimizedTheta': {'decomposition_type': 'additive'}}),

    # ('DeepAR', None),
    # ('AutoGluonTabular', None),
    # ('TemporalFusionTransformer', None),

]

fit_and_print(
            hp=None,
            idf=input_df,
            presets="high_quality",
            tdf=training_df,
            output_file_name=f"out_best.png"
        )

for model, hp in models:

    try:
        fit_and_print(
            hp=hp if hp is not None else {model: {}},
            idf=input_df,
            tdf=training_df,
            output_file_name=f"out_{model.lower()}.png"
        )
    except ValueError as e:
        logging.exception(e)
