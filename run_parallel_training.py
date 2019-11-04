import h2o
import pandas as pd
import pickle

from concurrent import futures
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


def train(estimator, h2o_f_train, features, label, h2o_server_port=54321):
    """
    Train the GLM model, return the H2O trained estimator.
    """

    h2o.init(ip="localhost", port=h2o_server_port, max_mem_size='4G')

    if estimator == 'glm':
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator

        glm = H2OGeneralizedLinearEstimator(family = 'gaussian')
        glm.train(x=features, y=label, training_frame=h2o_f_train)

        return glm
    elif estimator == 'drf':
        from h2o.estimators.random_forest import H2ORandomForestEstimator

        drf = H2ORandomForestEstimator()
        drf.train(x=features, y=label, training_frame=h2o_f_train)

        return drf
    elif estimator == 'xgb':
        from h2o.estimators.xgboost import H2OXGBoostEstimator

        xgb = H2OXGBoostEstimator()
        xgb.train(x=features, y=label, training_frame=h2o_f_train)

        return xgb


def ensemble(estimators, h2o_f_train, features, label, h2o_server_port=54321):
    """
    Get the list of trained models, create an ensemble and save it as
    a pickle object.
    """

    h2o.init(ip="localhost", port=h2o_server_port, max_mem_size='4G')

    ensemble = H2OStackedEnsembleEstimator(
        seed=1234567,
        keep_levelone_frame=True,
        base_models=estimators)
    ensemble.train(x=features, y=label, training_frame=h2o_f_train)

    pickle.dump(
        h2o.save_model(
            model=ensemble,
            path='h2o_ensemble',
            force=True),
        open('model/h2o_ensemble.sav', 'wb'))


if __name__ == "__main__":
    # load the dataset into a H2O Frame - run a local H2O server
    try:
        h2o.init(ip="localhost", port=54321, max_mem_size='4G')
        h2o.remove_all()

        h2o_f = h2o.H2OFrame(
            pd.read_csv("data/winequality-white.csv", delimiter=';'))
        h2o_f_train, h2o_f_test = h2o_f.split_frame(ratios = [0.8],
                                                    seed = 1234567)

        # the column in the H2O frame that hosts
        # the labels
        label = 'quality'

        # the list of features - exclude the label
        features = list(h2o_f.columns)
        features.remove(label)

        estimators = []
        list_of_models = [
            (i, model) for i, model in enumerate(['glm', 'drf', 'xgb'])]
        with futures.ThreadPoolExecutor(
                max_workers=len(list_of_models)) as executor:
            future_list = {
                executor.submit(
                    train,
                    x[1],
                    h2o_f_train,
                    features,
                    label,
                    54322 + x[0]): x for x in list_of_models}

            for future in futures.as_completed(future_list):
                estimators.append(future.result())

        ensemble(estimators, h2o_f_train, features, label, 54321)
    except Exception as e:
        print(e)
    finally:
        h2o.remove_all()
        h2o.cluster().shutdown()
