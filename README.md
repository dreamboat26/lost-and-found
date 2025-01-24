# NextX: Forecasting Wind Speed for the Next X Hours

This repository contains various state-of-the-art models and custom models for predicting the next 12 hours of wind speed from a weather station. The aim of this project is to evaluate the performance of different models on the same dataset, as well as to test some novel models that have not been explored in existing research. At this point, all the models are designed in a multivariate-to-univariate format.

##  Dataset
The dataset used in this project is collected from a weather station, containing historical wind speed (target variable) and 11 other climatological data measurements with a 1-hour resolution, for 4 years from 2010 to 2013. The dataset is preprocessed and split into training , validation, and test sets for model training and evaluation. Data was collected from [here](https://www.ncei.noaa.gov/data/local-climatological-data/).

##  Hardware
All the models are trained/tested on a local machine with the following config:
- Model: Alienware Alienware m15 R4
- CPU: Intel Core i7-10870H
- GPU: NVIDIA GeForce RTX 3060 Laptop - 6 GB
- Memory: 16 GB - DDR4 SDRAM
- OS: Microsoft Windows 11 Professional (x64)

## Dependencies
All the models are implemented in python (v3.9.16) with the following key dependencies:

- pytorch: 2.0.0
- numpy: 1.24.1
- pandas: 1.5.3
- scikit-learn: 1.2.2
- tqdm: 4.65.0

All the dependecies are provided in the requirement.txt file.

### Features
[dataprep](www.TODO.com) config file can be manually adjusted for different settings. The deafult setting used for this project is:
- TARGET =  windspeed
- WINDOWSIZE = 168
- HORIZON = 12
- SKIP = 0 (in case you desire to skip some hours between lookback and the forecasting horizon)
- TEST_SIZE = 0.2
- VALID_SIZE = 0.2

## Models:
I will implement some custom models as well as some well-known published models and evaluate their performance.

### Benchmarks and SOTA models that will be implemented:
- [x] [VanillaLSTM](https://github.com/Farzad-R/NextX/blob/main/src/models/univariate/LSTMBased.py): Custom design
- [x] [LSTMDENSE](https://github.com/Farzad-R/NextX/blob/main/src/models/univariate/LSTMBased.py): Custom design
- [x] [LSTMAutoEncoder](https://github.com/Farzad-R/NextX/blob/main/src/models/univariate/LSTMBased.py): Custom design
- [x] [Informer](https://arxiv.org/abs/2012.07436) (AAAI 2021 Best paper)
- [x] [FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022)
- [x] [Autoformer](https://arxiv.org/abs/2106.13008) (NeuIPS 2021)
- TODO: [Vanilla-Transformer](https://arxiv.org/abs/1706.03762) (NeuIPS 2017)
- TODO:CNNLSTM: Custom design

## Comparison of different models
I will arrange the models in descending order of performance, with the top model having the lowest mean squared error (MSE) and the subsequent models having progressively higher MSEs.

| Model                 | TEST MSE              | Number of parameters      | Number of epochs          | AVG epoch time (s)     |
| ---------------       | --------------------  | -----------------------   | ------------------------  | -----------------------|
| VanillaLSTM           | 0.0117                |128,364                    | 52                        | TBD                    |
| LSTMDENSE             | 0.0119                |137,932                    | 34                        | TBD                    |
| LSTMAutoEncoder       | 0.0119                |305,662                    | 44                        | TBD                    |
| Informer              | 0.0143                |673,420                    | 47                        | 15.5                   |
| Autoformer            | 0.0144                |674,344                    | 38                        | 26.0                   |
| VanillaTransformer    | 0.0158                |45,484                     | 55                        | 13.8                   |
| FEDformer(Wavelet)    | TBD                   |100,785,984                | TBD                       | TBD                    |

## Getting Started

### Project structure

```
.
├── data
│   ├── training                    # (auto-generate)
│   ├── clean                       # (auto-generate)
│   └── raw                         # The raw data
│       └── WTH.csv                 # The weather forecast data
│
├── xplr_notebooks                  # Data exploration jupyter notebooks
│
├── config                          # config files for the pipeline
│
├── src                             # Contains the codes of the main pipeline
│   ├── utils                       # Includes utils.py module
│   │
│   ├── clean
│   │
│   ├── dataprep
│   └── models
│       ├── Linear
│       ├── LSTMBased
│       └── Transformers
│
├── logs                    
│   ├── debug.log               
│   └── info.log
│
├── requirements.txt                # Required dependencies of the project
├── README.md
├── .gitignore
│
├── main.py                         # Entry point for the preprocesing pipeline
├── FEDformerTrain.py
├── InformerTrain.py
└── AutoformerTrain.py
```
### Environment Requirements

First, please make sure you have installed Conda. Then, the environment can be installed using:
```
conda create -n nextX python=3.9.16
conda activate nextX
pip install -r requirements.txt
```

### Data Preparation

The required raw weather file (WTH.csv) is already placed in `./data/raw`. The dataset required to train transformer models is different from other benchamrks. Therefore two differnet sets of datasets can be generated. Please follwo the steps below:

In order to preprocess the data:
1. Open the terminal
2. Activate the environment

For `transformer based` models run:
```
python main.py clean_wth                
python main.py prepare_transformer_dataset
```

For `other benchmarks` run:
```
python main.py clean_wth                
python main.py prepare_target           
python main.py prepare_features
python main.py prepare_time_features
```

These commands will automatially clean and process the weather data. And due to the purpose of experimenting different form of datasets and models, the following datasets will be created in `data/training/wth/{windowsize}_{horizon}_{skip}` folder and for each {train/test/validation} set.
- **x_feat_{}.npy**: Contains the  selected climatological data measurements in sequential format (Number of samples, windosize, number of features)
- **x_{}.npy**: Contains the target (wind speed) data measurements in sequential format (Number of samples, windosize, 1)
- **x_{}_fut_time.npy**: Contains the future cyclic time features (Number of samples, horizon, 6)
- **x_{}_hist_time.npy**: Contains the historical cyclic time features (Number of samples, windowsize, 6)
- **y_{}.npy**: Contains the labels (Number of samples, horizon)

### Training Example
As this repository is intended for experimentation purposes, allowing for exploration of various model architectures and dataset combinations, the training process is not fully automated. Therefore, each training module can be executed following these steps:

1. Set the desired config in `config/training.cfg`
2. Import the desired model, in the corresponding module and set the model
3. Open the terminal and run the module. 

Example:
```
python trainLSTMBased.py
```

# Acknowledgment
I would like to express my gratitude to the authors of the models used in this project. With the exception of the custom design models, all other models were sourced from their corresponding repositories and only minor changes were made to adjust the models to this project. I acknowledge the authors for their significant contributions. The list of repositories used in this project is provided below:

- [FEDformer](https://github.com/MAZiqing/FEDformer)
- [Informer2020](https://github.com/zhouhaoyi/Informer2020)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)







