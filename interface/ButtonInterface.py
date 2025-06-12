import sys

from interface.SubInterface import SubInterface
from interface.TextRedirector import TextRedirector

class ButtonInterface(SubInterface):
    def _run_function(self):
        sys.stdout = TextRedirector(self.text_widget)
        try:
            self.function()
            print(f"\n{self.name} executado com sucesso.")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            sys.stdout = sys.__stdout__
