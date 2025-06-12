import sys
from tkinter import Label

from interface.SubInterface import SubInterface
from interface.TextRedirector import TextRedirector

class BatchInterface(SubInterface):
    def __init__(self, function, name, master, entries):
        self.entries = entries
        super().__init__(function, name, master)

    def _build_interface(self):
        Label(self.master, text=f"Executar {self.name}:").pack(anchor="w")
        super()._build_interface()

    def _run_function(self):
        sys.stdout = TextRedirector(self.text_widget)
        try:
            self.function(mode=self.entries)
            print(f"\n{self.name} executada com sucesso.")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            sys.stdout = sys.__stdout__
