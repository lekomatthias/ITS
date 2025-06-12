import tkinter as tk
from tkinter import messagebox
from interface.SubInterface import SubInterface


class InvalidInterface(SubInterface):
    def _start_action(self):
        self._show_invalid_mode_popup()

    def _show_invalid_mode_popup(self):
        messagebox.showerror(
            title="Modo Inválido",
            message=f"O modo selecionado para '{self.name}' é inválido.\nVerifique o valor de 'mode'."
        )
