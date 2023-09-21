import tkinter as tk
from tkinter import filedialog
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_files():
    file_paths = []
    for _ in range(4):
        file_path = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat")])
        if file_path:
            file_paths.append(file_path)
    return file_paths

def plot_data():
    file_paths = load_files()
    
    plt.figure(figsize=(8, 6))
    
    for file_path in file_paths:
        mat_data = loadmat(file_path)
        # Assuming your data is stored under a specific key, adjust this accordingly
        data_to_plot = mat_data['data_key'][0][0]
        plt.plot(data_to_plot, label=file_path)
    
    plt.xlabel("X Label")
    plt.ylabel("Y Label")
    plt.title("Line Plot of Data from MAT Files")
    plt.legend()
    plt.show()

root = tk.Tk()
root.title("MAT File Data Plotter")

load_button = tk.Button(root, text="Load Files", command=load_files)
load_button.pack()

plot_button = tk.Button(root, text="Plot Data", command=plot_data)
plot_button.pack()

root.mainloop()
