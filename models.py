from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info('Starting training script')

# Read in the training dataset
def read_training_data() -> pd.DataFrame:
    logging.info('Reading training data')
    try:
        return pd.read_csv('Training_Data2.csv')
    except:
        logging.debug('Failed to read training data')
        logging.exception('')
    pass


models = {
    #"Logreg": LogisticRegression(solver='lbfgs'),
    #"NN": KNeighborsClassifier(),
    #"LinearSVM": SVC(probability=True, kernel='linear'), #class_weight='balanced'
    #"GBC": GradientBoostingClassifier(),
    #"DT": tree.DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB()
}

metrics = {
    "Accuracy": accuracy_score,
    #"Recall": recall_score,
    "MSE": mean_squared_error
}

def split_training_data(data_X: pd.DataFrame, data_Y: pd.DataFrame, test_size: float, rand_state:int):
   return train_test_split(data_X, data_Y, test_size=test_size, random_state=rand_state)
   

def train_model(models: dict, metrics: dict, data_X: pd.DataFrame, data_Y: pd.DataFrame):
    df = pd.DataFrame(index=models.keys(), columns=metrics)
    trained_models = []
    X_train, X_test, y_train, y_test = split_training_data(data_X, data_Y, test_size= 0.3, rand_state= 1)
    for model_name, model in models.items():
        logging.info(F"Starting to train the {model_name} model")
        try:
            trained_model = model.fit(X_train, y_train)
            trained_model_y = trained_model.predict(X_test)
            trained_models.append([model_name, trained_model])
            for metric_name, metric in metrics.items():
                df.at[model_name, metric_name] = metric(y_test, trained_model_y)
        except:
            logging.exception("")

    return df, pd.DataFrame(trained_models)

def select_optimal_model(trained_models:pd.DataFrame, scores: pd.DataFrame, metric:str, metric_dir: int):
    logging.info(F"Beginning to find optimal model")
    try:
        if metric_dir == 1:
            Best_model = scores[[metric]].astype(float).squeeze().argmin()
        else: 
            Best_model = scores[[metric]].astype(float).squeeze().argmax()
    except:
        logging.exception('')

    return trained_models[Best_model]

df_train = read_training_data()
df_scores, trained_models = train_model(models=models, metrics= metrics, data_X= df_train.iloc[:,:-1], data_Y= df_train.iloc[:,-1:])
optimal_model = select_optimal_model(scores= df_scores, trained_models= trained_models, metric= 'MSE', metric_dir=0)


