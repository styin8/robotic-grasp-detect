import importlib
import torch.nn as nn
import os
from .base_model import BaseModel


def find_model_using_name(model_name):
    """
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls
    if model is None:
        print(
            f"In {model_name}.py, there are no subclasses of BaseModel that match the {model_name}!")
    return model


def create_model(opt):
    """
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model {type(instance).__name__} was created!")
    return instance
