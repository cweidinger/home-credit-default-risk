# Documentation

* proposal.pdf contains my proposal for this Udacity Capstone Project
** The review for the proposal is at https://review.udacity.com/#!/reviews/1313808
* project-report.pdf contains the final documentation for this project

# Steps to recreate

## Download my code

git clone https://github.com/cweidinger/home-credit-default-risk.git

## Download datasets

Navigate to https://www.kaggle.com/c/home-credit-default-risk/data

Press the "Download All" button

When finished downloading unzip the all.zip into the git repo's input/ directory.

## Install libraries

pip install lightgbm matplotlib seaborn scipy numpy pandas scikit-learn ipython jupyter

If you get "Library Not Loaded" for lightgbm then you may need to follow these instructions: https://stackoverflow.com/questions/44937698/lightgbm-oserror-library-not-loaded.

## Run the Code

### Exploratory Analysis

python exploration.py

### Benchmark model

python benchmark.py

### Implementation refinement history

python implementation-refinement-history.py

### Free-form Visualization

python free-form-visualization.py

## Helper Libraries

* library/datasets.py discovers the hierarhical relationships between the target variable and the other variables within many datasets
* transformations.py takes a datasets tree and returns a totally preprocessed pandas dataframe and feature list
* roc.py this file implements the LGBMClassifier for giving an roc given a set of features. It also contains the tuned model
* feature_selection.py is a collection of feature selection tools
* plot_helpers.py can be used to visualize results in a general way

## Executable Helpers

* populate_col_effect.py was used generate a score for each column individual to aid column selection
* hyperparameter-tuning.py was in part used to tune the model
* Udacity Capstone Project.ipynb is the notebook I used to develop all the above scripts. I then copied them to their respective files when they were fleshed out.


