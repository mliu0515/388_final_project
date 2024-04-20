from utils.music_gen_small import MusicGenSmall
from utils.transcription_module import WhisperDiarization
import glob
import json

KEYS_DICT = json.load(open("config.json"))
OPENAI_KEY = KEYS_DICT["api_key"]
PYANNOTE_KEY = KEYS_DICT["use_auth_token"]

if __name__ == "__main__":
    audio_paths = glob.glob("../../data/*.mp3")
    whisper_diarization_module = WhisperDiarization(OPENAI_KEY, PYANNOTE_KEY)
    whisper_diarization_module.run_all(audio_paths)
    music_gen = MusicGenSmall(llama_key="llama_key", openai_key="openai_key")
    music_gen.run_all(glob.glob("../../out/transcriptions/*.txt"), glob.glob("../../out/scenes/*.txt"))