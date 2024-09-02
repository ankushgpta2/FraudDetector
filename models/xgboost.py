import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os


class XGBoost():
    def __init__(self):
        # Define XGBoost parameters
        self.params = {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'eta': 0.1,
            'eval_metric': 'auc'
        }

        # Define the parameter grid for hyperparameter tuning
        self.param_grid = {
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        # define the base path for certain outputs
        self.base_path_to_use = os.path.dirname(os.path.abspath(__file__))
    
    def tune_and_train(self, training, evaluation, y_train, y_val):
        """
        """
        # check if the best tuned params already exists
        if not os.path.isfile(os.path.join(self.base_path_to_use, 'xgboost_best_tuned.json')):
            # initialize the XGBoost classifier
            xgb_clf = xgb.XGBClassifier(**self.params)
            
            # calculate the base value for scale_pos_weight
            num_pos = sum(y_train == 1) 
            num_neg = sum(y_train == 0)  
            base_scale_pos_weight = num_neg / num_pos
            self.param_grid['scale_pos_weight'] = [base_scale_pos_weight * 0.5, base_scale_pos_weight, base_scale_pos_weight * 2]

            # perform Grid Search or Randomized Search
            random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=self.param_grid, 
                                    scoring='roc_auc', cv=5, n_iter=100, verbose=1, n_jobs=-1)
            random_search.fit(training, y_train, eval_set=[(evaluation, y_val)], verbose=False)

            # get the best estimator and save under self.best_model
            self.best_model = random_search.best_estimator_
            print("\nBest Parameters found: ", random_search.best_params_)

            # save this best tuned model 
            self.best_model.save_model(os.path.join(self.base_path_to_use, 'xgboost_best_tuned.json'))
        else:
            # load in model params 
            self.load_model()

    def predict(self, test, y_test):
        """
        """
        # predict and evaluate
        y_pred = self.best_model.predict(test)
        y_pred_proba = self.best_model.predict_proba(test)[:, 1]

        # calculate multiple evaluation metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"AUC: {auc}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        return y_pred, y_pred_proba
    
    def load_model(self):
        """
        """
        # load the model from a file
        self.best_model = xgb.XGBClassifier()
        self.best_model.load_model(os.path.join(self.base_path_to_use, 'xgboost_best_tuned.json'))