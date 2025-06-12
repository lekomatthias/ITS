
from tkinter import END

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        if string:
            self.text_widget.after(0, self._write_to_widget, string)

    def _write_to_widget(self, string):
        try:
            if self.text_widget.winfo_exists():
                self.text_widget.config(state='normal')
                
                if string.startswith('\r'):
                    # Apaga a última linha (simulando sobrescrita como em terminal)
                    self._delete_last_line()
                    string = string.lstrip('\r')

                self.text_widget.insert(END, string)
                self.text_widget.see(END)
                self.text_widget.config(state='disabled')
        except Exception:
            pass

    def _delete_last_line(self):
        last_line_index = self.text_widget.index("end-2l")
        self.text_widget.delete(last_line_index, "end-1c")

    def flush(self):
        pass
    