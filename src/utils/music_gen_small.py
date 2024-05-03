import transformers
import scipy
from llamaapi import LlamaAPI
# also import GPT-4 from OpenAI
from openai import OpenAI
import glob
import torch
import json

TRANSCRIPTION_FILE_LIST = glob.glob("../../out/transcriptions/*.txt")
SCENE_DESCRIPTION_FILE_LIST = ...
KEYS_DICT = json.load(open("config.json"))
OPENAI_KEY = KEYS_DICT["api_key"]
HUGGING_FACE_TOKEN = KEYS_DICT["use_auth_token"]

class MusicGenSmall:
    def __init__(self, llama_key, openai_key):
        self.llama = ... # LlamaAPI(llama_key)
        self.openai = ... # OpenAI(api_key=openai_key)
        self.prompt = "Generate a music description for the following transcription:" # TODO: change this prompt. Rightnow it's shit.
        self.hugging_face_token = HUGGING_FACE_TOKEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_music_description(self, transcription, scene_description, model="Llama"):
        """_summary_

        Args:
            transcription (string): the diariesed transcription plus the video scene description of the video file. which we extracted from a txt file.

        Returns:
            string: the music description generated by GPT-4/Llama (TBD)
        """
        # TODO: Implement this! Currently is just placeholders.
        combined_prompt = "introcude yourself." # self.prompt + transcription + scene_description
        if model == "Llama":
            pipeline = transformers.pipeline(
                "text-generation",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=self.device
                # use_auth_token=self.hugging_face_token
            )
            messages = [{"role": "user", "content": combined_prompt}]
            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            description = outputs[0]["generated_text"][len(prompt):]
            print(description)
            return description
        elif model == "GPT-4":
            description = ...
        return description

    def description_to_audio(self, description, output_name):
        """

        Args:
            description (string): the music description generated by GPT-4/Llama (TBD)
            output_name (string): the name of the audio file to be written
        
        Returns:
            None: writes the audio file to the output_name
        """
        synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
        music = synthesiser(output_name, forward_params={"do_sample": True})
        scipy.io.wavfile.write(f"../../out/mus/{output_name}.wav", rate=music["sampling_rate"], data=music["audio"])

        return None
    
    def run_all(self, transcription_file_list, scene_description_file_list):
        """_summary_
        Given the list of transcription files and the list of scene description files,
        this function will generate the music description for each transcription file and scene description file pair.
        And output the audio file of the music description.

        Args:
            transcription_file_list (_type_): the list of transcription files
            scene_description_file_list (_type_): the list of scene description files

        Returns:
            _type_: None. But writes the audio files to the out folder.
        """
        for transcription_file, scene_description_file in zip(transcription_file_list, scene_description_file_list):
            with open(transcription_file, "r") as f:
                transcription = f.read()
            with open(scene_description_file, "r") as f:
                scene_description = f.read()
            description = self.generate_music_description(transcription, scene_description)
            out_name = transcription_file.split("/")[-1].split(".")[0]
            self.description_to_audio(description, out_name)
        return None
            



if __name__ == "__main__":
    music_gen = MusicGenSmall(llama_key="llama_key", openai_key="openai_key")
    # test 1
    # des = music_gen.generate_music_description("This is a test transcription", "This is a test scene description")

    # At the end, we can run the whole pipeline, which is:
    # music_gen.run_all(TRANSCRIPTION_FILE_LIST, SCENE_DESCRIPTION_FILE_LIST)