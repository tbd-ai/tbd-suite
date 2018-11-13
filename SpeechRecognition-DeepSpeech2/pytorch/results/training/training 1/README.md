This run produced our first model.

Training dataset is:
	LibriSpeech train_clean-100, train_clean-360, train_other-500
Validation dataset is:
	LibriSpeech val_clean, val_other
Test dataset is:
	LibriSpeech test_clean
	
Epoch time: took 15-20 hours to train.
Batch size, learning rate, annealing rate were varied and experimented upon.
In "training 2" we fix the parameters.

WER

|Epoch		|Test		|Validation			|Hyperparameter Notes|
|-----------|-----------|-------------------|--------------------|
|Start		|--			|82.4				|MLPerf Default|
|3			|--			|41.9				|MLPerf Default|
|6			|--			|37.5				|MLPerf Default|
|9			|--			|35.4				|Original Paper|
|13			|--			|34.1				|"Aggressive LR"|
|20			|21.1		|33.7				|"Aggressive LR"|

See exact parameters in the params.py found inside each epoch folder.