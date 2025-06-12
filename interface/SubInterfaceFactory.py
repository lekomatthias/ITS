from interface.ButtonInterface import ButtonInterface
from interface.PromptInterface import PromptInterface
from interface.SelectInterface import SelectInterface
from interface.OpenWindowInterface import OpenWindowInterface
from interface.BatchInterface import BatchInterface

from interface.InvalidInteface import InvalidInterface

class SubInterfaceFactory:
    @staticmethod
    def create(mode, function, name, master, options=None):
        if mode == 'button':
            return ButtonInterface(function, name, master)
        elif mode == 'prompt':
            return PromptInterface(function, name, master)
        elif mode == 'select':
            return SelectInterface(function, name, master, options)
        elif mode == 'open':
            return OpenWindowInterface(function, name, master)
        elif mode == 'batch':
            return BatchInterface(function, name, master, options)
        else:
            return InvalidInterface(function, name, master)
