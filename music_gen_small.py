from transformers import pipeline
import scipy


synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
music = synthesiser("romantic orchestral piece with swelling strings and a tender piano melody, reflecting both the urgency and the deep emotional resonance.", forward_params={"do_sample": True})
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])

