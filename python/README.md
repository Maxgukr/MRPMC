# MRPMC: Mortality Risk Prediction Model for COVID-19

MRPMC: Mortality Risk Prediction Model for COVID-19


## Pre-requirements
* Python 3.7
* numpy 1.16
* pandas 0.25
* lightgbm 2.3.1
* scikit-learn
* shap 0.35
* tensorflow 2.2.0
* mlxtend 0.17.2


## Installation

### Installation from Github
To clone the repository and install manually, run the following from a terminal:
```Bash
git clone https://github.com/paprikachan/CIRPMC.git
cd CIRPMC
```

## Usage

### Help page

In command line:
```shell
Usage: predict_CIRPMC.R [options]

Options:
        -i CHARACTER, --infile=CHARACTER
                Path of X input file

        -o CHARACTER, --outfile=CHARACTER
                Path of Y output file

        -h, --help
                Show this help message and exit
```

### Quick start
The following code runs an example of CIRPMC.

```shell
predict_CIRPMC.R -i test_X.csv -o pred_Y.csv
```

## File format

### Input file


Input file is a csv file, stores the measurements of 23 markers for each patient:
* BUN	
* SpO2	
* RR	
* No. comorbidities	
* D-Dimer	
* Age	
* Hypertention	
* LDH	
* APTT	
* WBC	
* LBC	
* PT	
* UA	
* Consciousness	
* Hb	
* ALB	
* Diabetes	
* Coronary heart disease	
* PLT	
* CRP	
* Sputum	
* TB	
* PCT
 

### Output file
Out file is a csv file, stores the predicted results from MRPMC:
* LR: The predicted critical illness probablity from logistic regression
* SVM: The predicted critical illness probablity from supported vector machine
* RF: The predicted critical illness probablity from random forest
* GBDT: The predicted critical illness probablity from gradient boosted decision tree
* NN: The predicted critical illness probablity from neural network
* Probability: The predicted critical illness probablity from our ensemble model CIRPMC
* Cluster: The predicted critical illness status, 0 or 1.
* Risk group: The stratified risk group, Non-critical or Critical.


## Cite us

## Help
If you have any questions or require assistance using CIRPMC, please open an issue.

