import sys
from tkinter import Toplevel

from interface.SubInterface import SubInterface
from interface.TextRedirector import TextRedirector

class OpenWindowInterface(SubInterface):
    def _start_action(self):
        self._open_output_popup()
        self._run_tkinter_loop()

    def _run_tkinter_loop(self):
        sys.stdout = TextRedirector(self.text_widget)
        try:
            self.function(root=Toplevel(self.master))
            print(f"\n{self.name} executado com sucesso.")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            sys.stdout = sys.__stdout__
