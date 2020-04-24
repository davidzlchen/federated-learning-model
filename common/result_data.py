import json


class ResultData(object):
    def __init__(self, test_loss=-1, model_accuracy=-1, size=-1, specs=-1, iteration=-1, epochs=-1):
        self.test_loss = test_loss
        self.model_accuracy = model_accuracy
        self.specs = specs
        self.size = size
        self.iteration = iteration
        self.epochs = epochs


class ResultDataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ResultData):
            model_acc_str = str(obj.model_accuracy)[7:13]
            return {
                "__resultdata__": "true",
                "test_loss": str(
                    obj.test_loss),
                "model_accuracy": model_acc_str,
                "size": str(obj.size),
                "specs": str(obj.specs),
                "iteration": str(obj.iteration),
                "epochs": str(obj.epochs)}

        return json.JSONEncoder.default(self, obj)


def as_result_data(dct):
    if "__resultdata__" in dct:
        test_loss = dct["test_loss"]
        model_accuracy = dct["model_accuracy"]
        size = dct["size"]
        specs = dct["specs"]
        iteration = dct["iteration"]
        epochs = dct["epochs"]
        return ResultData(test_loss, model_accuracy, size, specs, iteration, epochs)
