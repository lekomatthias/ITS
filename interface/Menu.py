import tkinter as tk

from interface.SubInterfaceFactory import SubInterfaceFactory

class Menu:
    def __init__(self, functions, title="Menu"):
        self.functions = functions
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("300x500")
        self.canvas = None
        self.scrollable_frame = None
        self._scroll_pending = False
        self.window_id = None

    def build_interface(self):
        self._add_title()
        self._create_scrollable_area()
        self._populate_functions()
        self._add_exit_button()
        self.root.mainloop()

    def _add_title(self):
        tk.Label(self.root, text="Menu de funções", font=('Arial', 16)).pack(pady=10)

    def _bind_mousewheel(self):
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.root.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

    def _unbind_mousewheel(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _create_scrollable_area(self):
        container = tk.Frame(self.root)
        container.pack(fill='both', expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(container, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')

        scrollbar = tk.Scrollbar(container, orient='vertical', command=self.canvas.yview, width=15)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.bind("<Enter>", lambda e: self._bind_mousewheel())
        self.canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.window_id, width=event.width)

    def _reset_scroll_flag(self):
        self._scroll_pending = False

    def _on_mousewheel(self, event):
        if self._scroll_pending:
            return
        self._scroll_pending = True
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.root.after_idle(self._reset_scroll_flag)

    def _populate_functions(self):
        self.function_index = 0
        self._load_next_function()

    def _load_next_function(self):
        if self.function_index < len(self.functions):
            item = self.functions[self.function_index]
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(pady=10)

            SubInterfaceFactory.create(
                mode=item['mode'],
                function=item['function'],
                name=item['name'],
                master=frame,
                options=item.get('list', [])
            )

            self.function_index += 1
            self.root.after(50, self._load_next_function)


    def _add_exit_button(self):
        tk.Button(self.root, text="Sair", width=30, height=2, command=self.root.quit).pack(pady=5)
