from pyannote.audio import Pipeline
# from pyannote_whisper.pyannote_whisper.utils import diarize_text
from src.utils import diarize_text
from openai import OpenAI
import torch
import whisper
import glob
import json


KEYS_DICT = json.load(open("config.json"))
OPENAI_KEY = KEYS_DICT["api_key"]
PYANNOTE_KEY = KEYS_DICT["use_auth_token"]

class WhisperDiarization:
    def __init__(self, openai_key, pyannote_key):
        # self.audio_paths = audio_paths
        self.openai_key = openai_key
        self.pyannote_key = pyannote_key
        self.client = OpenAI(api_key=openai_key)
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=pyannote_key)
        # self.audio_files = [open(audio_path, "rb") for audio_path in self.audio_paths]

    def whisper_and_diarization(self, audio_file, audio_path):
        """_summary_
            Given the audio file and the path to the audio file, 
            this function will return the diarized transcription of the audio file.

        Args:
            audio_file (_type_): the audio file to be transcribed
            audio_path (_type_): the path to the audio file

        Returns:
            _type_: lines of diarized transcription
        """
        asr_result = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            timestamp_granularities=["word"],
            response_format="verbose_json"
        )
        diarization_result = self.pipeline(audio_path)
        final_result = diarize_text(asr_result, diarization_result)

        lines = []
        for seg, spk, sent in final_result:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
            lines.append(line)
        return lines

    def run_all(self, audio_paths):
        """_summary_
            For each audio file in the list of audio files,
            this function will run the whisper_and_diarization function.
            And output the diarized transcription of the audio file to a txt file.

        Args:
            audio_paths (_type_): list of audio paths
        """
        audio_files = [open(audio_path, "rb") for audio_path in audio_paths]
        output_names = [audio_path.split("/")[-1].split(".")[0] for audio_path in audio_paths]
        for audio_file, audio_path, output_name in zip(audio_files, audio_paths, output_names):
            lines = self.whisper_and_diarization(audio_file, audio_path)
            lines = self.post_process(lines)
            with open(f"../../out/transcriptions/{output_name}.txt", "w") as f:
                for line in lines:
                    f.write(line)
                    f.write("\n")

    def post_process(self, text_lines):
        """_summary_

        Args:
            text_lines (string): text lines to be post processed

        Returns:
            string: test lines post processed
        """
        return text_lines

# load config.json in a dictionary

# initialize OpenAI client with your API key

# client = OpenAI(api_key=openai_key)
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
#                                     use_auth_token=pyannote_key)
# audio_file = open(audio_path, "rb")

# def whisper_and_diarization(audio_file):
#     asr_result = client.audio.transcriptions.create(
#     model="whisper-1", 
#     file=audio_file,
#     timestamp_granularities=["word"],
#     response_format="verbose_json"
#     )
#     diarization_result = pipeline(audio_path)
#     final_result = diarize_text(asr_result, diarization_result)

#     lines = []
#     for seg, spk, sent in final_result:
#         line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
#         lines.append(line)
#     return lines

# def post_process(text_lines):
#     """_summary_

#     Args:
#         text_lines (string): text lines to be post processed

#     Returns:
#         string: test lines post processed

#     """

#     return text_lines

if __name__ == "__main__":
    audio_paths = glob.glob("../../data/*.mp3")
    whisper_diarization_module = WhisperDiarization(OPENAI_KEY, PYANNOTE_KEY)
    whisper_diarization_module.run_all(audio_paths)