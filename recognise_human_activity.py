from tkinter import *
from PIL import ImageTk, Image
import subprocess
import tkinter as tk

win = Tk()
win.geometry("600x350")
win.title("Human Activity Recognition")

img = Image.open("ok.jpg")
bg = ImageTk.PhotoImage(img)
label = Label(win, image=bg)
label.place(x=0, y=0)

tk.Label(win, text="HUMAN ACTIVITY RECOGNITION", fg="red", font=("Times", 25)).pack(padx=35, pady=20)

def browse():
        # Run the browser.py script as a separate process
        subprocess.Popen(['python', 'browser.py'])

def camera():
    try:
        # Run the camera.py script as a separate process
        subprocess.Popen(['python', 'camera.py'])
    except Exception as e:
        print("Error:", e)

at = Button(win, text="Browse", font=("Calibri", 14, "bold"), command=browse).pack(side=LEFT, padx=80, pady=50)
B = Button(win, text="Camera", font=("Calibri", 14, "bold"), command=camera).pack(side=LEFT, padx=30, pady=20)
bt = Button(win, text="Quit", font=("Calibri", 14, "bold"), command=win.quit).pack(side=LEFT, padx=70, pady=20)

win.mainloop()
