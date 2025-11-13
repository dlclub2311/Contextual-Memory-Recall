## Contextual Memory Recall: A Novel Metric for Class Incremental Learning
<img width="2151" height="1391" alt="CMR-Visualisation-Complete" src="https://github.com/user-attachments/assets/0c36fba9-89da-44d9-a204-763833844c11" />

#### 1. Code Dependencies
To install the required packages and to switch to the new environment: 
```bash
conda env create --file environment.yaml && conda activate cpcmr
```
Add the appropriate ANACONDA path in ```prefix``` inside ```environment.yaml```

#### 2. Data Setup

1. Cifar-100 dataset is automatically downloaded by the code and the required data setup is done by the code.
2. ImageNet-100 and Imagenet-1K datasets have to be downloaded and organized according to the paths given in the respective files in *imagenet_split* folder.

#### 3. Experiments
To reproduce CPCMR value in Table 1(a) for the **INC10** setting on **CIFAR100** with three different class orders:

```bash
python3 -minclearn --options options/<model-name>/<model>.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> \
    --data-path <PATH/TO/DATA> --log-file CPCMR_cifar100_INC10_p0.1.txt \
    --hint-replace-prob 0.1 --calc-hint --save-model task
```
For results on other methods and other datasets, change the ```options``` value accordingly.


#### 4. Results
The results are saved in the specified log files (see the ```--log-file``` option).


## Acknowledgements

This repository is developed based on [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch).
