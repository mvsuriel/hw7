# Exercise 7: Python: APIS

- María Victoria Suriel
- Mariajosé Argote
- Elvis Casco

In this exercise we created our own API that serves a Machine Learning model.

Steps of the exercise:

1. Create a repository in the account of one of the components of the group.

2. Train a Machine Learning model and generate a pickle file that stores it:

- we used the file `generate_pickle_file.py to generate two pickle files: `model_logistic_regression.pkl`and `model_random_forest.pkl`.

3. Create a Python file that will run the API in the same directory where the pickle file is stored.

- File: `api_fastapi.py`

4. Create an endpoint that receives an input file, ideally as a .json or a dictionary, reads the model (you can use joblib.load()) and outputs the prediction.

- File: `run_api.py`

5. Create a .json file with an example of the type of input the model needs, i.e. the features used to train the model.

- File: `example_batch_patients.json`: many entries
- File: `example_patient_data.json`: single entry

6. Create a file that contains the code to do a post request to the API sending the datapoint as input and prints the prediction. This code should handle potential API related issues as the ones seen in class.

- File: `api_client.py`

7. Try out the whole system by running in a terminal 

`uvicorn api_fastapi:app --reload`

and in a second terminal executing the second file with the request. 

Remember that usually the API service will be run in localhost:8000.

# Helpful files  and links to use API:

- [documentation FastAPI](http://localhost:8000/docs)
- `api_usage_examples.py` : contains examples of how to use the API created in this exercise.
- `api_predictions_notebook.ipynb`: demonstrates how to make predictions using the API within a Jupyter notebook environment.
