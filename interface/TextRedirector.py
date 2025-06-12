from tkinter import END

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self._last_line_index = None

    def write(self, string):
        if string:
            self.text_widget.after(0, self._write_to_widget, string)

    def _write_to_widget(self, string):
        try:
            if not self.text_widget.winfo_exists():
                return

            self.text_widget.config(state='normal')

            if '\r' in string:
                self._delete_last_line()
                string = string.replace('\r', '')

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
