This code is for MVCINN.

The dataset can be download from [here](https://github.com/mfiddr/MFIDDR)
# TODO
The pre-train weight is in [here]


First, you should set dataset folds as follows (Attention: move the csv files to dataset fold):
```
|--
    |--MFIDDR
        |--test
            |--3392_19491104_left_05.jpg
            |--3392_19491104_left_06.jpg
            |--...
        |--train
            |--1_19970611_left_05.jpg
            |--1_19970611_left_06.jpg
            |--...
        |--test_fourpic_label.csv
        |--train_fourpic_label.csv

```
# TODO     
Second, you can download pre-train weight to weights as:
```
    |--weights
        |--final_0.8010.pth
```

Then, you can run test.py to test a pre-trained model or run train.py to train the MVCINN.