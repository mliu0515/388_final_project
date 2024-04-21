# import pytest and to pytest
import pytest
# import ../utils/music_gen_small.py and ../utils/transcription_module.py
from src.utils.music_gen_small import MusicGenSmall
# from src.utils.transcription_module import WhisperDiarization


# Define a test function
def test_music_gen_small():
    # Create an instance of MusicGenSmall
    music_gen = MusicGenSmall(llama_key="llama_key", openai_key="openai_key")
    # Define the transcription_file_list and scene_description_file_list
    pass 