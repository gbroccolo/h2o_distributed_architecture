# h2o_distributed_architecture

An example of distributed architecture to run H2O's models training in parallel
using futures in Python.

The main Python script is able to use the H2O client in order to init different
server processes in parallel and run training of different models (i.e.
estimators), that are finally ensembled.

The hard part is to share objects (H2O frames and H2O binary models) among the
H2O cluster. If objects are initialised locally (like opening the H2O frame
passed to train the estimators) the client is able to share it among the
cluster, but the problem is still present for trained model saved as binary
pickles, since they are saved locally in the node that executed the training.

To solve the issue, we used HDFS support in order to use a distributed storage
where to share all produced binaries.


## The script

`run_parallel_training.py` is the script that init parallel connections in the
H2O cluster, aggregate the obtained model binaries in order to perform a final
ensemble based on stacking technique. All models are saved in a shared S3
storage.

To install the needed Python's dependencies just needed to run the H2O client,
run

```
$ pip install -r requirements.txt && pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
```

To run the script, passing the storage as argument:

```
$ python run_parallel_training.py s3a://<bucket>/<folder>
```


## The H2O cluster

The cluster is recreated using a dockerised cluster, using the `Dockerfile`
officially released by `h2o.ai`. H2O servers are launched integrating HDFS
support. For this purpose, remember to edit the file `hadoop/core-site.xml`
with your credentials.

To launch the cluster just run

```
$ docker-compose up -d
```

# Shutdown the cluster

```
$ docker-compose down && docker rmi -f h2oai_server:latest
```


## The sample used in this example

The CSV it's open and retrievable from
https://www.kaggle.com/danielpanizzo/wine-quality
