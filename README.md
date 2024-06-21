# WhisperX Diarization

This repository contains a simple program to diarize (long)
Dutch audio files. I wrote this to help my mom, so there
are some extra batch files for easier setup and running.

Basically, the `diarize.py` script is the main script,
that does all of the diarization. It can be called
from the command line as well, just look at `run.bat`.

Then there is `interface.py`, which simply starts 
`diarize.py` in a subprocess, with specific parameters.

`interface.bat` runs `interface.py` in a virtual environment
if it was found, or in the global interpreter if no virtual 
environment is setup in the current directory.

## Setup

Simply install the requirements from `requirements.txt`.
If you have a CUDA-enabled GPU (with the appropriate drivers),
you can then run `install-torch-cuda.bat` to install a CUDA-compiled
version of `torch`, which significantly
speeds up the transcription process.

## Patches

There are some patches applied to `whisperx` in `diarize.py`,
which are used to print the progress of the current step.
Otherwise, it is kind of a black box, and there user has
no idea how far along the script is.

## Output

The script produces two files:
- `<audio file>.log` containing the log of the diarizion process.
  This includes a full transcript WITHOUT speaker recognition, regardless
  of whether you selected speaker recognition or not.
- `<audio file>.txt` containing the full transcript, either with or 
  without speaker recognition, depending on your selection.
