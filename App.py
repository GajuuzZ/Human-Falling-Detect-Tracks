import os
import cv2
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt


class main:
    def __init__(self, master):
        self.master = master
        self.master.title('Human Falling Detection')


root = tk.Tk()
app = main(root)
root.mainloop()
