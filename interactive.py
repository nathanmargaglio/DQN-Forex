import os
import ipywidgets as widgets
from IPython.display import display, clear_output

load_button = widgets.Button(description="Load Previous Session?",
                        layout=widgets.Layout(width='200px'))

valid = widgets.Valid(
    value=False,
    readout='',
    layout=widgets.Layout(display='inline-block')
)

box = widgets.Box([load_button, valid])
display(box)

sessions = sorted(os.listdir('sessions'), key=lambda x: os.path.getctime('sessions/' + x))
session = widgets.Dropdown(
    options=sessions,
    value=sessions[-1],
    disabled=True,
    layout=widgets.Layout(width='200px')
)
display(session)

def on_load_button_clicked(b):
    clear_output()
    display(box)
    display(session)

    session.disabled = not session.disabled
    if not session.disabled:
        valid.value = True
    else:
        valid.value = False

load_button.on_click(on_load_button_clicked)
