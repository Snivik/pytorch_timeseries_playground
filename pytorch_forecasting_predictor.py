import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, GroupNormalizer
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ ==  '__main__':

    prediction_length = 60 * 24
    batch_size = 64
    num_workers = 0
    optimal_learning_rate = 0.008


    input_df = pd.read_csv('data/synthetic_daily_sin_1.csv')


    # Prep data
    input_df['minutes_since_start'] = range(0, len(input_df))
    input_df['group'] = ['m1'] * len(input_df)
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    print(input_df.head())

    # Let's predict 15th day
    cutoff_time_index = max(input_df['minutes_since_start']) - prediction_length
    training_ds = TimeSeriesDataSet(
        data=input_df[lambda x: x.minutes_since_start <= cutoff_time_index],
        time_idx='minutes_since_start',
        target='value',
        group_ids=['group'],
        min_encoder_length=1,
        max_encoder_length=60,
        min_prediction_length=1,
        max_prediction_length=prediction_length,

        static_categoricals=["group"],
        static_reals=[],
        time_varying_known_categoricals=[],
        variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["minutes_since_start"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "value",
        ],
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_ds = TimeSeriesDataSet.from_dataset(training_ds, input_df, predict=True, stop_randomization=True)
    train_dataloader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = validation_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")  # log to tensorboard

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=[0],
        enable_model_summary=True,
        gradient_clip_val=0,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    optimal_learning_rate = 0.008
    # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=optimal_learning_rate if optimal_learning_rate is not None else 0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4,
        optimizer='adam'
    )

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    if optimal_learning_rate is None:
        res = trainer.tuner.lr_find(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        print(f"suggested learning rate, update optimal_learning_rate with it to avoid reruns: {res.suggestion()}")
        print(f"Update it in the variables up top and start again")
    else:
        trainer.fit(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        )

        # https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html#Evaluate-performance

        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        raw_predictions, x = best_tft.predict(val_dataloader, mode='raw', return_x=True)
        best_tft.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)