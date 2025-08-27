# Project-Code

This project is for the Msc. Thesis in Machine Learning and Data Science at Imperial in 2025, written by Robert N de Witt.

There is an accompanying paper here: (link to be provided)

To reproduce the results, the parquet version of the data is provided so one can run the models and play them back. 

Here are the steps to proceed:

1. Clone the repo from github
2. tar xvfz mana_data.tgz - this will unpack the parquet formed market data and analytics for the models to run against.
3. Inside the Gym directory, the primary model driver can be found: research_train_test_vec.ipynb
4. Run the cells sequentially and you should be able to reproduce the results as reported. All models are seeded so they should be deterministic. 
