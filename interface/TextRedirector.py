
from tkinter import END

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        if string:
            self.text_widget.after(0, self._write_to_widget, string)

    def _write_to_widget(self, string):
        self.text_widget.config(state='normal')
        self.text_widget.insert(END, string)
        self.text_widget.see(END)
        self.text_widget.config(state='disabled')

    def flush(self):
        pass