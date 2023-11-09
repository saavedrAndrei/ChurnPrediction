import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f'File {event.src_path} has been modified. Reloading Streamlit app...')
        subprocess.run(['streamlit', 'run', 'main.py'])

observer = Observer()
observer.schedule(MyHandler(), path='.', recursive=True)
observer.start()

try:
    while True:
        pass
except KeyboardInterrupt:
    observer.stop()
observer.join()