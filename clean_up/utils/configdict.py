

















from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _convert_sub_configs(value):
    if isinstance(value, dict):
        return ConfigDict(value)

    if isinstance(value, list):
        return [_convert_sub_configs(subvalue) for subvalue in value]

    return value


class ConfigDict(dict):


    def __init__(self, initial_dictionary=None):

        if initial_dictionary:
            for field, value in initial_dictionary.items():
                initial_dictionary[field] = _convert_sub_configs(value)
            super().__init__(initial_dictionary)
        else:
            super().__init__()

    def __setattr__(self, attribute, value):
        self[attribute] = _convert_sub_configs(value)

    def __getattr__(self, attribute):
        try:
            return self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, attribute):
        try:
            del self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __setitem__(self, key, value):
        super().__setitem__(key, _convert_sub_configs(value))
