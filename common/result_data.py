import json


class ResultData(object):
    def __init__(self, test_loss=-1, model_accuracy=-1, size=-1):
        self.test_loss = test_loss
        self.model_accuracy = model_accuracy
        self.size = size


class ResultDataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ResultData):
            model_acc_str = str(obj.model_accuracy)[7:13]
            return {
                "__resultdata__": "true",
                "test_loss": str(
                    obj.test_loss),
                "model_accuracy": model_acc_str,
                "size": str(obj.size)}

        return json.JSONEncoder.default(self, obj)


def as_result_data(dct):
    if "__resultdata__" in dct:
        test_loss = dct["test_loss"]
        model_accuracy = dct["model_accuracy"]
        size = dct["size"]
        return ResultData(test_loss, model_accuracy, size)
