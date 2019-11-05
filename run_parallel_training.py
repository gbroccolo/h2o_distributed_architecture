import sys
import h2o
import pandas as pd

from concurrent import futures
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.persist import set_s3_credentials


S3BUCKET = None


def train(estimator, h2o_f_train, features, label, h2o_server_port=54321):
    """
    Train the GLM model, return the H2O trained estimator.
    """

    h2o.init(ip="localhost", port=h2o_server_port, max_mem_size='4G')

    if estimator == 'glm':
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator
        model = H2OGeneralizedLinearEstimator(
            nfolds=2,
            family='gaussian',
            seed=1234,
            keep_cross_validation_predictions=True)
    elif estimator == 'drf':
        from h2o.estimators.random_forest import H2ORandomForestEstimator
        model = H2ORandomForestEstimator(
            nfolds=2,
            seed=1234,
            keep_cross_validation_predictions=True)
    elif estimator == 'xgb':
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        model = H2OXGBoostEstimator(
            nfolds = 2,
            seed=1234,
            keep_cross_validation_predictions=True)

    model.train(x=features, y=label, training_frame=h2o_f_train)

    # NOTE that files are saved in each node of H2O cluster
    pathstring = h2o.save_model(
        model=model, path=S3BUCKET, force=True)
    return pathstring


def ensemble(
        trained_models, h2o_f_train, features, label, h2o_server_port=54321):
    """
    Get the list of trained models, create an ensemble and save it as
    a pickle object.
    """

    h2o.init(ip="localhost", port=h2o_server_port, max_mem_size='4G')

    # NOTE model pickles are loaded remotely
    trained_models = [h2o.load_model(x) for x in trained_models]

    ensemble = H2OStackedEnsembleEstimator(
        seed=1234567,
        keep_levelone_frame=True,
        base_models=trained_models)

    ensemble.train(x=features, y=label, training_frame=h2o_f_train)

    pathstring = h2o.save_model(
        model=ensemble,
        path=S3BUCKET,
        force=True)
    return pathstring


if __name__ == "__main__":
    # load the dataset into a H2O Frame - run a local H2O server
    try:
        if len(sys.argv) < 2:
            print("Specify the S3 object-based overlay to R/W binary files")
            print("\n\n    e.g. s3a://bucketname/folder \n")

        S3BUCKET = sys.argv[1]

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

        list_of_models = [
            (i, model) for i, model in enumerate(['glm', 'drf', 'xgb'])]
        trained_models = []
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
                obj = future.result()
                trained_models.append(obj)

        pathstring = ensemble(
            trained_models, h2o_f_train, features, label, 54321)
        print("\n\n    Model binary file saved in %s \n\n" % pathstring)
    except Exception as e:
        print(e)
    finally:
        h2o.remove_all()
