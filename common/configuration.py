from utils.enums import LearningType
import json

PUBLIC_ENUMS = {
    'LearningType': LearningType,
}


class Configuration(object):
    def __init__(self, learning_type=LearningType.CENTRALIZED):
        self.learning_type = learning_type

    def __str__(self):
        return "CONFIGURATION OBJECT -- LearningType: " + self.learning_type


class ConfigurationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Configuration):
            return {"__configuration__": "true", "learning_type": str(obj.learning_type)}
        return json.JSONEncoder.default(self, obj)


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_configuration(dct):
    if "__configuration__" in dct:
        name, member = dct["learning_type"].split(".")
        return Configuration(getattr(PUBLIC_ENUMS[name], member))


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(LearningType[name], member)
    else:
        return d
