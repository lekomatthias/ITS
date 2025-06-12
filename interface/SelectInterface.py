import sys
from tkinter import ttk, Label

from interface.SubInterface import SubInterface
from interface.TextRedirector import TextRedirector

class SelectInterface(SubInterface):
    def __init__(self, function, name, master, options):
        self.options = options
        super().__init__(function, name, master)

    def _build_interface(self):
        Label(self.master, text=f"Selecione o modo para {self.name}:").pack(anchor="w")
        self.combobox = ttk.Combobox(
            self.master,
            values=self.options,
            state="readonly",
            width=37
        )
        if self.options:
            self.combobox.current(0)
        self.combobox.pack(pady=3)
        super()._build_interface()

    def _run_function(self):
        sys.stdout = TextRedirector(self.text_widget)
        try:
            selection = self.combobox.get()
            self.function(mode=selection)
            print(f"\n{self.name} executado com sucesso.")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            sys.stdout = sys.__stdout__
