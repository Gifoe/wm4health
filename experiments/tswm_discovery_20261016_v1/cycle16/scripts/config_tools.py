"""Attribute-access facade for unchanged safe-loaded YAML values."""
import yaml
class Config(dict):
 def __getattr__(self,key):
  try:return self[key]
  except KeyError as error:raise AttributeError(key) from error
def convert(value):
 if isinstance(value,dict):return Config({key:convert(item) for key,item in value.items()})
 if isinstance(value,list):return [convert(item) for item in value]
 return value
def load(path):return convert(yaml.safe_load(path.read_text(encoding='utf-8')))
