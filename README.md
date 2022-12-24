# Real world Anomaly Detection in Surveillance Videos with CNN+LSTM : Pytorch 
# This is the End of 4th year project at INSAT
## Goal  : Design and development of a crime forecasting system based on the videos of the surveillance cameras.



## Datasets

Download following data [link](https://drive.google.com/file/d/18nlV4YjPM93o-SdnPQrvauMN_v-oizmZ/view?usp=sharing) and unzip under your $DATA_ROOT_DIR.
/workspace/DATA/UCF-Crime/all_rgbs
* Directory tree
 ```
    DATA/
        UCF-Crime/ 
            ../all_rgbs
                ../~.npy
            ../all_flows
                ../~.npy
        train_anomaly.txt
        train_normal.txt
        test_anomaly.txt
        test_normal.txt
        
```

## Architecture : 
#### Prediction model  : 

![image](https://user-images.githubusercontent.com/71349228/209438013-a5a0581d-410b-4d8d-8817-6721390453db.png)

#### IOT idea :

![image](https://user-images.githubusercontent.com/71349228/209438048-d42a6887-43d3-40ef-9b58-c1db6f8bbdc9.png)

## Results:

![image](https://user-images.githubusercontent.com/71349228/209438068-3cab3407-38c9-44bf-9d36-187c7101e03d.png)

## train-test script
```
python main.py
```
```
python mainlstm.py
```

