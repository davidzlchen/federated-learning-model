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
