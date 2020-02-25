# Federated Learning Demo

Demo of federated learning.

## Setting up requirements

Clone this repository and cd to it.

Install [mosquitto](https://mosquitto.org/), a MQTT broker.

Install virtualenv if you don't have it already.

```bash
  pip install virtualenv
```

Then create a new virtualenv and switch to it.

```bash
  virtualenv ENV
  source ENV/in/activate
```

Then install the requirements

```bash
  pip install -r requirements.txt
```

## Running

```bash
  source ENV/bin/activate
  python server.py
  python client.py
```

## Contributing

After making changes, make sure to format your code.

```bash
autopep8 --in-place --aggressive --aggressive *.py
```
