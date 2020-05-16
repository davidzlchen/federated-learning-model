# Federated Learning Demo

Demo of federated learning.

## Setting up requirements

**These instructions assume you are on Ubuntu 16.04, they may be different for other operating systems.**

Clone this repository and cd to it.

Install [mosquitto](https://mosquitto.org/), a MQTT broker.

Install virtualenv if you don't have it already.

```bash
pip install virtualenv
```

Then create a new virtualenv and switch to it.

```bash
virtualenv ENV
source ENV/bin/activate
```

Then install the requirements

```bash
pip install -r requirements.txt
```

## Running

### Starting the server
```bash
source ENV/bin/activate
python server.py
```

### Starting clients (add as many clients as you want)
```bash
source ENV/bin/activate
python client.py
```

### Starting a task
Navigate to http://localhost:5000 and configure your task.

## Contributing

### Server/Client code

**Changing the learning task**

The model used is the [PersonBinaryClassifier](https://github.com/davidzchen-ut/federated-learning-model/blob/a70294382a9e95a57c76d6810d287f57084f83df/common/models.py#L9) object defined in [common/models.py](https://github.com/davidzchen-ut/federated-learning-model/blob/master/common/models.py).

As long as you are using a PyTorch Module object, there should not be any other changes that need to be made. 

The [train](https://github.com/davidzchen-ut/federated-learning-model/blob/a70294382a9e95a57c76d6810d287f57084f83df/common/models.py#L48) and [test](https://github.com/davidzchen-ut/federated-learning-model/blob/a70294382a9e95a57c76d6810d287f57084f83df/common/models.py#L107) function are defined in the same file, [common/models.py](https://github.com/davidzchen-ut/federated-learning-model/blob/master/common/models.py).

In general, the ModelRunner object is used throughout the project to run the learning task. The project uses the [get_model_runner function](https://github.com/davidzchen-ut/federated-learning-model/blob/a70294382a9e95a57c76d6810d287f57084f83df/common/person_classifier.py#L101) in the common/person_classifier.py file to get this ModelRunner object. To find where we use this function, you can use any IDE/GitHub's find references feature to see all the locations that it is being used.

**Changing the training data**

The client and server code to read in training / test data expect the data to be in a certain format. Any dataset can be used as long as the images are preprocessed into the following format. The data is expected to be in a .pkl (Python pickle) format, and contain an array of tuples. The first argument in each tuple is the image itself (ex. 2D array of image pixels) and the second argument is an array of labels. The labels are specific to the learning task, but for our case either "person" or "no-person" must exist (only one). In addition, this array of labels also contains all clusters that this image can be a part of (ex. "sky", "water", "ground").

The [Jupyter Notebook](https://github.com/davidzchen-ut/federated-learning-model/blob/data_preprocessing/data/data_setup/Convert%20Coco%20Dataset%20to%20Federated%20Learning%20Demo%20Dataset.ipynb) used by us to preprocess the Coco dataset into our format is provided for reference in data/data_setup/.

In server.py, there is a global variable CLUSTER_NAMES. This variable is an array of cluster names that the server is allowed to assign to clients. It is set by the developer and is required to be changed when the dataset / cluster labels change.

**Miscellaneous**

After making changes, make sure to format your code.

```bash
python3 -m autopep8 --in-place --aggressive --aggressive --recursive *.py common utils
```

### React
Install dependencies. Make sure you have Node and NPM. Our configuration uses Node v10.16.3 and npm v6.14.4.
```bash
cd website
npm install
```

To make changes, it is recommended to use the live server.
```bash
npm start
```

After making changes, create a minified compiled version that Flask will serve.
```bash
npm run build
```
