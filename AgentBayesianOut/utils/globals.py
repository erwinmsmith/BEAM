import sys
import random
from typing import Union, Literal, List
from transformers import AutoTokenizer
import transformers

class Singleton:
    _instance = None

    @classmethod
    def instance(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    def reset(self):
        self.value = 0.0

class Cost(Singleton):
    def __init__(self):
        self.value = 0.0

class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class Time(Singleton):
    def __init__(self):
        self.value = ""

class Mode(Singleton):
    def __init__(self):
        self.value = ""

class Tokenizer(Singleton):
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.value = ""


class Deepseek_Tokenizer:
    _instance = None

    def __new__(cls, model_path):
        if cls._instance is None:
            cls._instance = super(Deepseek_Tokenizer, cls).__new__(cls)
            cls._instance.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True
            )
            cls._instance.value = ""
        return cls._instance

    @classmethod
    def instance(cls, model_path):
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance