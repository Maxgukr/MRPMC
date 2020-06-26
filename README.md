# CIRPMC: Critical Illness Risk Prediction Model for COVID-19

CIRPMC: Critical Illness Risk Prediction Model for COVID-19


## Pre-requirements
* R3.6
* caret
* e1071
* gbm
* randomForest


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


Input file is a csv file, stores the measurements of 7 inflammatory markers for each patient:
* IL-1β   (pg/mL, < 5.0)
* TNF-α (pg/mL, < 8.1)
* IL-6     (pg/mL, < 7.0)
* IL-10   (pg/mL, < 9.1)
* IL-8     (pg/mL, < 62)
* PCT     (ng/mL, > 0)
* CRP     (mg/L,   > 0)

Note: CRP, C reactive protein. PCT, procalcitonin. TNF-α, tumor necrosis factor α. IL-1 β, interleukin 1β. IL-2R, IL-6, interleukin 6. IL-8, interleukin 8. IL-10, interleukin 10. 

### Output file
Out file is a csv file, stores the predicted results from CIRPMC:
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

