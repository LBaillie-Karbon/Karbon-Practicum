from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    mean_squared_error,
    confusion_matrix,
)
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("Starting training script")

# Read in the training dataset
def read_training_data() -> pd.DataFrame:
    logging.info("Reading training data")
    try:
        return pd.read_csv("Training_Data3.csv")
    except:
        logging.debug("Failed to read training data")
        logging.exception("")
    pass


models = {
    # "Logreg": LogisticRegression(solver='lbfgs'),
    # "NN": KNeighborsClassifier(),
    # "LinearSVM": SVC(probability=True, kernel='linear'), #class_weight='balanced'
    # "GBC": GradientBoostingClassifier(),
    # "DT": tree.DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB(),
}

metrics = {
    "Accuracy": accuracy_score,
    # "Recall": recall_score,
    "MSE": mean_squared_error,
}


def split_training_data(
    data_X: pd.DataFrame, data_Y: pd.DataFrame, test_size: float, rand_state: int
):
    return train_test_split(
        data_X, data_Y, test_size=test_size, random_state=rand_state
    )


def train_model(
    models: dict, metrics: dict, data_X: pd.DataFrame, data_Y: pd.DataFrame
):
    df = pd.DataFrame(index=models.keys(), columns=metrics)
    trained_models = []
    X_train, X_test, y_train, y_test = split_training_data(
        data_X, data_Y, test_size=0.3, rand_state=1
    )
    
    for model_name, model in models.items():
        logging.info(f"Starting to train the {model_name} model")
        try:
            trained_model = model.fit(X_train, y_train)
            trained_model_y = trained_model.predict(X_test)
            trained_models.append([model_name, trained_model, trained_model_y])
            for metric_name, metric in metrics.items():
                df.at[model_name, metric_name] = metric(y_test, trained_model_y)
            # log_confusion_matrix(model_name=model_name, y_true=y_test, y_pred=trained_model_y)
        except Exception as e:
            logging.exception(f"Exception occurred with traceback: {e}")

    return df, pd.DataFrame(
        trained_models, columns=["Model Name", "Trained Model", "Predictions"]
    )


def select_optimal_model(
    trained_models: pd.DataFrame, scores: pd.DataFrame, metric: str, metric_dir: int
):
    ################################################
    # Optimal Model = DF
    #                   Model Name
    #                   Trained Model
    #                   Predictions on Test
    ################################################
    logging.info(f"Beginning to find optimal model")
    try:
        if metric_dir == 1:
            Best_model = scores[[metric]].astype(float).squeeze().argmin()
        else:
            Best_model = scores[[metric]].astype(float).squeeze().argmax()
    except:
        logging.exception("")

    logging.info(trained_models)
    logging.info(
        f"The optimal model was {trained_models['Model Name'].loc[Best_model]}"
    )
    return trained_models.loc[Best_model]


def log_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: pd.Series):
    #      LOG THE CONFUSION MATRIX
    logging.info(f"Below is the confusion matrix for {model_name}")
    logging.info(confusion_matrix(y_true=y_true, y_pred=y_pred))
    pass

def run_model():
    df_train = read_training_data()
    df_ids = df_train[['TenantPermaKey', 'EmailMessagePermaKey']]
    df_train = df_train.drop(['TenantPermaKey','EmailMessagePermaKey'] , axis=1)
    df_scores, trained_models = train_model(
        models=models,
        metrics=metrics,
        data_X=df_train.iloc[:, :-1],
        data_Y=df_train.iloc[:, -1:],
    )
    optimal_model = select_optimal_model(
    scores=df_scores, trained_models=trained_models, metric="MSE", metric_dir=0
    )
    df_train = df_train.drop('Label', axis=1)
    df_train['Prediction'] = optimal_model['Trained Model'].predict(df_train)
    df_train = pd.merge(df_ids, df_train, left_index=True, right_index=True)
    return df_train

run_model()