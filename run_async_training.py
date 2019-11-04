import h2o
import pandas as pd
import pickle

import asyncio
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


async def train(estimator, h2o_f_train, features, label, h2o_server_port=54321):
    """
    Train the GLM model, return the H2O trained estimator.
    """

    h2o.init(ip="localhost", port=h2o_server_port, max_mem_size='4G')

    if estimator == 'glm':
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator

        glm = H2OGeneralizedLinearEstimator(
            family = 'gaussian', keep_cross_validation_predictions=True)
        glm.train(x=features, y=label, training_frame=h2o_f_train)

        return glm
    elif estimator == 'drf':
        from h2o.estimators.random_forest import H2ORandomForestEstimator

        drf = H2ORandomForestEstimator(keep_cross_validation_predictions=True)
        drf.train(x=features, y=label, training_frame=h2o_f_train)

        return drf
    elif estimator == 'xgb':
        from h2o.estimators.xgboost import H2OXGBoostEstimator

        xgb = H2OXGBoostEstimator(keep_cross_validation_predictions=True)
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


async def submitter(loop, h2o_f_train, features, label):
    """
    Async submitter of tasks
    """

    i = 0
    tasks = []
    for model in ['glm', 'drf', 'xgb']:
        tasks.append(
            loop.create_task(
                train(model, h2o_f_train, features, label, 54322 + i)))

    tasks = await asyncio.gather(*tasks)
    estimators = []
    for task in tasks:
        estimators.append(task)

    return estimators


if __name__ == "__main__":
    try:
        # load the dataset into a H2O Frame - run a local H2O server
        h2o.init(ip="localhost", port=54321, max_mem_size='4G')
        h2o.remove_all()

        h2o_f = h2o.H2OFrame(
            pd.read_csv("data/winequality-white.csv", delimiter=';'))
        h2o_f_train, h2o_f_test = h2o_f.split_frame(
            ratios = [0.8], seed = 1234567)

        # the column in the H2O frame that hosts
        # the labels
        label = 'quality'

        # the list of features - exclude the label
        features = list(h2o_f.columns)
        features.remove(label)

        loop = asyncio.get_event_loop()
        estimators = loop.run_until_complete(
            submitter(loop, h2o_f_train, features, label))
        loop.close()

        ensemble(estimators, h2o_f_train, features, label, 54321)
    except Exception as e:
        print(e)
    finally:
        h2o.remove_all()
        h2o.cluster().shutdown()
