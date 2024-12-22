import tkinter as tk
from tkinter import simpledialog
import pystray
from PIL import Image, ImageDraw
import screeninfo

# Function to create an icon
def create_image():
    width = 64
    height = 64
    image = Image.new('RGB', (width, height), (255, 255, 255))
    dc = ImageDraw.Draw(image)
    dc.rectangle(
        (width // 2, 0, width, height // 2),
        fill='black')
    dc.rectangle(
        (0, height // 2, width // 2, height),
        fill='black')
    return image

# Function to show the chat window
def show_chat():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    chat_window = tk.Toplevel(root)
    chat_window.title("Chat")
    chat_window.geometry("300x400")

    # Get screen width and height
    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    # Adjust the window size and position it at the bottom right corner above the taskbar
    window_width = 300
    window_height = 500
    taskbar_height = 80  # Approximate height of the Windows taskbar

    x = screen_width - window_width
    y = screen_height - window_height - taskbar_height
    chat_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    chat_log = tk.Text(chat_window, state='normal')
    chat_log.pack(pady=10)

    messages = []

    def send_message(event=None):
        message = message_entry.get()
        if message:
            messages.append(message)
            chat_log.config(state='normal')
            chat_log.insert(tk.END, f"You: {message}\n")
            chat_log.config(state='disabled')
            message_entry.delete(0, tk.END)
            return messages

    print(messages)

    message_entry = tk.Entry(chat_window)
    message_entry.pack(pady=10)
    message_entry.bind("<Return>", send_message)

    send_button = tk.Button(chat_window, text="Send", command=send_message)
    send_button.pack(pady=10)

    chat_window.mainloop()

# Function to quit the application
def quit_app(icon, item):
    icon.stop()

# Create the system tray icon
icon = pystray.Icon("chat")
icon.icon = create_image()
icon.menu = pystray.Menu(
    pystray.MenuItem("Open Chat", show_chat),
    pystray.MenuItem("Quit", quit_app)
)

# Run the system tray application
icon.run()
