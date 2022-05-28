## Multiwave COVID-19 Prediction via Social Awareness-Based Graph Neural Networks using Mobility and Web Search Data

A Social Awareness-Based Graph Neural Network (SAB-GNN) architecture predicting daily Covid-19 infection cases based on inner-city mobility and web search data from mobile phone 

## Background
* Recurring outbreaks of COVID-19 call for a predictor of pandemic waves with early availability.
* Existing prediction models that forecast the first outbreak wave using mobility data may not be applicable to the multiwave prediction, because mobility      patterns across different waves exhibit varying relationships with fluctuations in infection cases.
* We propose a SAB-GNN that considers the decay of symptom-related web search frequency to predict multiple waves.

## Introduction

* Problem Setting: the urban study area is divided into N different urban districts.
* Prediction Features: 
  - the daily population flow between N urban districts 
  - the number of Covid-19 symptom-related web search records for residents living in each district
  - historical daily new infection cases for each district. 
* Prediction Labels: 
  - the daily new infection cases for each district for the next 7/14/21 days.
* Model Architecture: 
  - spatial module: graph neural network 
  - the social awareness recovery module
  - temporal module: LSTM. 
* This GitHub repository presents codes of 
  - extracting and preprocessing the raw mobility and web search data
  - training and testing the SAB-GNN model.

## Publication

**Multiwave COVID-19 Prediction via Social Awareness-Based Graph Neural Networks using Mobility and Web Search Data**
Jiawei Xue, Takahiro Yabe, Kota Tsubouchi, Jianzhu Ma\*, Satish V. Ukkusuri\*, The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD-22). 

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
