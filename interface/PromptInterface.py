import sys
from tkinter import Label, Entry

from interface.SubInterface import SubInterface
from interface.TextRedirector import TextRedirector

class PromptInterface(SubInterface):
    def _build_interface(self):
        Label(self.master, text=f"Entrada para {self.name}:").pack(anchor="w")
        self.entry = Entry(self.master, width=40)
        self.entry.pack(pady=3)
        super()._build_interface()

    def _run_function(self):
        sys.stdout = TextRedirector(self.text_widget)
        try:
            user_input = self.entry.get()
            self.function(mode=user_input)
            print(f"\n{self.name} executado com sucesso.")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            sys.stdout = sys.__stdout__
