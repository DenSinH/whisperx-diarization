from tkinter import Tk
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import subprocess
import threading
import time
from pathlib import Path
import os.path
import sys

root: Tk = None
PROC: subprocess.Popen = None
OUTPUT_LOCK = threading.Lock()
OUTPUT_BUFFER = ""


def output_string(log, string: str):
    global OUTPUT_BUFFER
    log.write(string)
    log.flush()
    sys.stdout.write(string)
    sys.stdout.flush()
    with OUTPUT_LOCK:
        OUTPUT_BUFFER += string


def stream_transcription(filepath: Path, diarize):
    """ Main function, calls the diarize script with the appropriate parameters """
    global PROC, OUTPUT_BUFFER
    if PROC is not None:
        PROC.kill()

    # clear output buffer
    with OUTPUT_LOCK:
        OUTPUT_BUFFER = ""

    cmd = [
        sys.executable,
        "./diarize.py",
        "-a", os.path.abspath(filepath),
        "--whisper-model", "large-v3",
        # "--device", "cpu",  # commented out for auto-selection based on cuda availability
        "--batch-size", "2",  # 16 is too large, run out of GPU memory
        "--language", "nl",
    ]

    if not diarize:
        cmd.append("--no-diarize")

    PROC = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )

    yield f"Started process {PROC.pid} with python interpreter {sys.executable}\n\n"

    # stream word by word
    word = ""
    for c in iter(lambda: PROC.stdout.read(1), b""):
        k: str = c.decode("utf-8", errors="ignore")
        if k.isspace():
            yield word + k
            word = ""
        else:
            word += k
    if word:
        yield word

    # cleanup
    PROC.wait()
    PROC = None
    yield "Cleanup complete\n"


def start_process():
    """ Start the transcription if a file is selected """
    filepath = filedialog.askopenfilename()
    if filepath:
        # clear the text box
        output_text.delete("1.0", tk.END)
        threading.Thread(target=poll_subprocess, args=(Path(filepath), diarize_var.get()), daemon=True).start()


def poll_subprocess(filepath: Path, diarize: bool):
    """ Poll the transcription process to stream output to the window """
    try:
        with open(filepath.with_suffix(".log"), "w+", errors="ignore") as log:
            for output in stream_transcription(filepath, diarize):
                output_string(log, output)
    except RuntimeError:
        # it is possible we tried to append to the output text
        # after the window was closed
        pass


def on_close():
    """ Extra safety, asking the user if they want to cancel
        an active transcription, if applicable """
    # ask if the user really wants to cancel the current transcription
    if PROC is not None:
        cancel_transcription = messagebox.askokcancel(
            "Transcriptie bezig...",
            "Wilt u de huidige transcriptie annuleren?"
        )
        if cancel_transcription:
            PROC.kill()
            root.destroy()
    else:
        # no active process, just destroy the frontend
        root.destroy()


if __name__ == '__main__':
    root = Tk()
    root.title("AutoNotulist")
    root.geometry("1200x600")

    # style the window
    root.configure(bg="#2b2b2b")

    # add a frame for better organization
    frame = tk.Frame(root, bg="#2b2b2b")
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # top frame for controls
    top_frame = tk.Frame(frame, bg="#2b2b2b")
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

    # widgets
    select_file_btn = tk.Button(
        top_frame,
        text="Kies Bestand",
        command=start_process,
        bg="#2b2b2b",
        fg="#FFFFFF",
        padx=10,
        pady=5
    )
    select_file_btn.pack(side=tk.LEFT, padx=(0, 10))

    # Add a checkbox for 'Sprekerherkenning'
    diarize_var = tk.BooleanVar()
    diarize_cb = tk.Checkbutton(
        top_frame,
        text="Sprekerherkenning",
        variable=diarize_var,
        bg="#2b2b2b",
        fg="#FFFFFF",
        selectcolor="#2b2b2b",
        activebackground="#2b2b2b",
        activeforeground="#FFFFFF"
    )
    diarize_cb.pack(side=tk.LEFT)

    # scrolled output to view progress
    output_text = tk.scrolledtext.ScrolledText(
        frame,
        wrap=tk.WORD,
        bg="#1e1e1e",
        fg="#FFFFFF",
        insertbackground="#FFFFFF",
    )
    output_text.config(state=tk.DISABLED, cursor="arrow")
    output_text.pack(side=tk.BOTTOM, pady=10, fill=tk.BOTH, expand=True)
    frame.pack_propagate(True)

    # register on-close handler
    root.protocol("WM_DELETE_WINDOW", on_close)

    def _update_output():
        prev_output_length = 0
        while True:
            with OUTPUT_LOCK:
                if len(OUTPUT_BUFFER) == prev_output_length:
                    continue

                output_text.config(state=tk.NORMAL)
                # clear output if we removed text
                if len(OUTPUT_BUFFER) < prev_output_length:
                    output_text.delete('1.0', tk.END)
                    prev_output_length = 0

                # insert new output
                end_visible = output_text.yview()[1] == 1.0
                output_text.insert(tk.END, OUTPUT_BUFFER[prev_output_length:])
                output_text.config(state=tk.DISABLED)
                prev_output_length = len(OUTPUT_BUFFER)

            if end_visible:
                # scroll to the end if the end was already visible
                # this prevents autoscrolling if the user is scrolling
                # manually
                output_text.see(tk.END)

            time.sleep(0.016)

    threading.Thread(target=_update_output, daemon=True).start()

    # run the ui
    root.mainloop()
