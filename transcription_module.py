from pyannote.audio import Pipeline
# from pyannote_whisper.pyannote_whisper.utils import diarize_text
from utils import diarize_text
from openai import OpenAI
import torch
import whisper
import json


# load ../config.json in a dictionary
audio_path = "pyannote_whisper/data/SIMPSON.mp3"
keys_dict = json.load(open("config.json"))
openai_key = keys_dict["api_key"]
pyannote_key = keys_dict["use_auth_token"]
# initialize OpenAI client with your API key

client = OpenAI(api_key=openai_key)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=pyannote_key)
audio_file = open(audio_path, "rb")

def whisper_and_diarization(audio_file):
    asr_result = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    timestamp_granularities=["word"],
    response_format="verbose_json"
    )
    diarization_result = pipeline(audio_path)
    final_result = diarize_text(asr_result, diarization_result)

    lines = []
    for seg, spk, sent in final_result:
        line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
        lines.append(line)
    return lines

def post_process(text_lines):
    """_summary_

    Args:
        text_lines (string): text lines to be post processed

    Returns:
        string: test lines post processed

    TODO: figure out what needs to be done here, or if we need this in the first place
    """

    return text_lines

if __name__ == "__main__":
    lines = whisper_and_diarization(audio_file)
    lines = post_process(lines) 
    # write each line on seperate lines

    with open("output2.txt", "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")