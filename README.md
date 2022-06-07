## Multiwave COVID-19 Prediction from Social Awareness using Web Search and Mobility Data

A Social Awareness-Based Graph Neural Network (SAB-GNN) architecture predicting daily Covid-19 infection cases based on inner-city mobility and web search data from mobile phone. 

## Data Description 
To replicate the implmentation of SAB-GNN, we need 
  - feature 1: mobility data
  - feature 2: web search data
  - feature 3: historical infection data  
and
  - code 1: training and testing code
  - code 2: sab_gnn model code

Features 1 and 2 can be found as "YJ Covid-19 Prediction Data" from https://randd.yahoo.co.jp/en/softwaredata.
Feature 3 is under "/data-collection/" in this Github repository.
Codes 1 and 2 are under "/SAB-GNN/" in this Github repository.

To see implementation results, please refer to
"MultiwaveCovidPrediction/SAB-GNN/sab_gnn/implementation_results_html_files/"
or
"MultiwaveCovidPrediction/SAB-GNN/sab_gnn_wsa/implementation_results_html_files/".

## Background
* Recurring outbreaks of COVID-19 call for a predictor of pandemic waves with early availability.
* Existing prediction models that forecast the first outbreak wave using mobility data may not be applicable to the multiwave prediction, because mobility patterns across different waves exhibit varying relationships with fluctuations in infection cases.
* We propose a SAB-GNN that considers the decay of symptom-related web search frequency to predict multiple waves.

## Introduction
* Problem Setting: the urban study area is divided into N different urban districts.
* Prediction Features: 
  - feature 1: the daily population flow between N urban districts 
  - feature 2: the number of Covid-19 symptom-related web search records for residents living in each district
  - feature 3: historical daily new infection cases for each district. 
* Prediction Labels: 
  - the daily new infection cases for each district for the next 7/14/21 days.
* Model Architecture: 
  - spatial module: graph neural network 
  - the social awareness recovery module
  - temporal module: LSTM. 
* This GitHub repository presents codes of 
  - training and testing the SAB-GNN model.
  - the SAB-GNN model.

## Publication

**Multiwave COVID-19 Prediction from Social Awareness using Web Search and Mobility Data**
Jiawei Xue, Takahiro Yabe, Kota Tsubouchi, Jianzhu Ma\*, Satish V. Ukkusuri\*, Accepted by the 28TH ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD-22), Washington DC Convention Center, August 14-18, 2022. 

## Requirements
* Ubuntu 16.04
* Python 3.8.5
* PyTorch 1.9.0 

## Data Collection and Preprocessing

<p align="center">
  <img src="https://github.com/JiaweiXue/MultiwaveCovidPrediction/blob/main/figures/figure_flow.png" width="350">
</p>

## SAB-GNN Architecture

<p align="center">
  <img src="https://github.com/JiaweiXue/MultiwaveCovidPrediction/blob/main/figures/figure_model.png" width="550">
</p>

## License
MIT license
