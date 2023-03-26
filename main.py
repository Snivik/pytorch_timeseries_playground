import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

prediction_length = 24 # Should be 2hrs really, but my first synthetic is hourly scoped
input_df = pd.read_csv('data/synthetic1.csv')
input_df['group'] = ['m1'] * len(input_df) # Assign groups

# Convert timestamps
input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])

#
input_df['hours_since_start'] = range(0, len(input_df))
cutoff_time_index = len(input_df) - 24

input_df.tail(prediction_length)

training_ds = TimeSeriesDataSet(

    data = input_df[lambda x: x.hours_since_start <= cutoff_time_index],
    time_idx='hours_since_start',
    target='value',
    group_ids=['group'],
    max_encoder_length=2,
    max_prediction_length=prediction_length,
)

validation_ds = TimeSeriesDataSet.from_dataset(training_ds, input_df, predict=True, stop_randomization=True)


batch_size=128
num_workers=2

train_dataloader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
val_dataloader = validation_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=4, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
)

# create the model
tft = TemporalFusionTransformer.from_dataset(
    training_ds,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
res = trainer.tuner.lr_find(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model
trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)



