import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[
        ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((500,500), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

#Initalize the window
root = tk.Tk()
root.title("Image Viewer")
root.geometry("600x600")

#Create a button to open image
btn = tk.Button(root, text="Open Image", command =open_image)
btn.pack(pady=20)

#Create a label to display the image
label = tk.Label(root)
label.pack()

root.mainloop()