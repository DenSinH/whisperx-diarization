import typing
from dotenv import load_dotenv
load_dotenv()

import os
import argparse
from pathlib import Path
import hashlib
from functools import wraps
import numpy as np
import pickle
import logging
from contextlib import contextmanager


# get a huggingface (read) token from
# https://huggingface.co/settings/tokens
# and fill in the form at
# https://huggingface.co/pyannote/segmentation-3.0
# to access the diarization model
HF_TOKEN = os.environ["HF_TOKEN"]
MTYPES = {"cpu": "int8", "cuda": "float16"}
MODEL_DIR = "./.models/"
CACHE_DIR = Path("./.cache/")
AUDIO_HASH = None


class LogIter:
    """ Basic wrapper around an iterator to log progress
    This is needed to log progress in whisperx alignment
    which does not support a progress printing hook
    (at least, not one that works...) """

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        for i, segment in enumerate(self.iterable):
            logging.info(f"Processing segment {i} / {len(self.iterable)}")
            yield segment

    def __len__(self):
        return len(self.iterable)

    def __getitem__(self, item):
        return self.iterable[item]

    def __setitem__(self, key, value):
        self.iterable[key] = value


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio", help="name of the target audio file", required=True
    )

    """ 
    from https://github.com/openai/whisper:
    Size     Parameters  English-only model    Multilingual model    Required VRAM    Relative speed
    tiny     39 M        tiny.en               tiny                  ~1 GB            ~32x
    base     74 M        base.en               base                  ~1 GB            ~16x
    small    244 M      small.en              small                  ~2 GB            ~6x
    medium   769 M     medium.en             medium                  ~5 GB            ~2x
    large    1550 M     N/A                   large                  ~10 GB           1x
    large-v2
    large-v3
    """

    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="large-v3",
        help="name of the Whisper model to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=16,
        help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["nl", "en"],
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )

    parser.add_argument(
        "--no-diarize",
        action="store_false",
        dest="diarize",
        default=True,
        help="Disables speaker recognition. "
             "This significantly speeds up the transcription process.",
    )

    return parser.parse_args()


def cached(file: str, onload=None):
    """ Decorator to cache function result to a file,
    expected to be used on functions for which the first
    argument is the audio file being processed
    We use the global AUDIO_HASH variable to set the correct
    cache path. """

    def decorator(func):
        @wraps(func)
        def wrapped(audio, *args, **kwargs):
            assert isinstance(audio, np.ndarray), "Cache function argument is not audio data"
            assert AUDIO_HASH is not None, "No audio hash"

            # full cache path
            cache = CACHE_DIR / AUDIO_HASH / file

            # make cache folder
            os.makedirs(cache.parent.absolute(), exist_ok=True)

            # check cache file, load result if exists
            if os.path.exists(cache):
                logging.info(f"Loading '{file}' result from file")
                with open(cache, "rb") as f:
                    result = pickle.load(f)
                    if onload is not None:
                        onload(result)
                    return result
            else:
                # no cache, compute and dump
                result = func(audio, *args, **kwargs)
                with open(cache, "wb+") as f:
                    pickle.dump(result, f)
                return result

        return wrapped
    return decorator


@contextmanager
def _transcription_context():
    """ Transcription context for clarifying transcription in stdout """
    print("Starting transcription\n\n")
    yield
    print("\n\nEnd of transcription, start post-processing")


def _print_segment(segment, **kwargs):
    """ Print a single segment, to be followed up by more """
    text = segment["text"]
    if isinstance(text, list):
        text = text[0]
    print(text.strip(), end=" ", flush=True, **kwargs)


def _print_transcription_onload(result):
    """ Print initial transcription to stdout whenever it was
        loaded from the cache as well """
    with _transcription_context():
        for segment in result["segments"]:
            _print_segment(segment)


@cached("transcription", _print_transcription_onload)
def transcribe(audio: np.array):
    logging.info("Loading transcription model")
    model = whisperx.load_model(
        args.model_name,
        args.device,
        compute_type=MTYPES[args.device],
        language=args.language,
        download_root=MODEL_DIR,
        threads=0  # max threads
    )

    """ monkey patch the whisper model call to print intermediate output """
    _whisper_model_call = model.__call__

    def _whisper_model_call_monkey_patch(*args, **kwargs):
        for segment in _whisper_model_call(*args, **kwargs):
            _print_segment(segment)
            yield segment
    model.__call__ = _whisper_model_call_monkey_patch

    logging.info(f"Transcribing audio on {args.device} with batch size {args.batch_size}")
    with _transcription_context():
        result = model.transcribe(
            audio,
            batch_size=args.batch_size,
        )

    return result


@cached("alignment")
def align(audio, transcript):
    logging.info("Loading alignment model")
    model_a, metadata = whisperx.load_align_model(
        language_code=transcript["language"],
        device=args.device,
        model_dir=MODEL_DIR
    )

    logging.info("Aligning segments")
    result = whisperx.align(
        LogIter(transcript["segments"]),
        model_a,
        metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )

    return result


@cached("diarization")
def diarize(audio, aligned, min_speakers=None, max_speakers=None):
    logging.info("Loading diarization model")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN,
        device=args.device
    )

    logging.info("Diarizing speakers")

    """
    See the 'apply' method in 
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization:
    """
    if typing.TYPE_CHECKING:
        from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    def _hook(step, *args, completed=None, total=None, **kwargs):
        message = f"Diarization step {step}"
        if completed is not None and total is not None:
            message += f": {completed} / {total}"
        logging.info(message)

    """ monkey patch DiarizationPipeline.setup_hook """
    def _pipeline_setup_hook_monkey_patch(*args, **kwargs):
        logging.info("Setting up diarization hook")
        return _hook
    diarize_model.model.setup_hook = _pipeline_setup_hook_monkey_patch

    # call patched diarization model
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, aligned)

    return result


def write_transcript(fp, transcript):
    for segment in transcript["segments"]:
        _print_segment(segment, file=fp)


def write_diarized_transcript(fp, diarized):
    if not diarized["segments"]:
        return

    previous_speaker = diarized["segments"][0]["speaker"]
    fp.write(f"{previous_speaker}: ")

    for segment in diarized["segments"]:
        # get previous speaker by default
        speaker = segment.get("speaker", previous_speaker)
        sentence = segment["text"].strip()

        # if this speaker doesn't match the previous one, start a new paragraph
        if speaker != previous_speaker:
            fp.write(f"\n\n{speaker}:\n")
            previous_speaker = speaker

        fp.write(sentence)
        fp.write(" ")


if __name__ == '__main__':
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s] %(message)s",
        level=logging.INFO
    )

    logging.info("Loading whisperx")
    import whisperx
    logging.info("Loading torch")
    import torch

    args = get_args()

    logging.info("Loading audio")
    audio = whisperx.load_audio(args.audio)
    AUDIO_HASH = hashlib.sha256(audio.view(np.int8)).hexdigest()
    logging.info(f"Audio hash {AUDIO_HASH}")

    transcript = transcribe(audio)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    if args.diarize:
        aligned = align(audio, transcript)

        # delete model if low on GPU resources
        # gc.collect(); torch.cuda.empty_cache(); del model_a

        diarized = diarize(audio, aligned)

        with open(Path(args.audio).with_suffix(".txt"), "w+") as f:
            write_diarized_transcript(f, diarized)
    else:
        with open(Path(args.audio).with_suffix(".txt"), "w+") as f:
            write_transcript(f, transcript)
