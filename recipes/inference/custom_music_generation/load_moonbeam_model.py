import sys
import os

# Get the absolute path to the Moonbeam-MIDI-Foundation_Model directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import torch
from recipes.inference.custom_music_generation.generation import MusicLlama
from src.llama_recipes.datasets.music_tokenizer import MusicTokenizer

# Path to the downloaded model weights
ckpt_dir = "/Users/savi/Documents/PYTHON_PROJECTS/MoonbeamProject/Moonbeam-MIDI-Foundation_Model/models/moonbeam-midi-foundation-model/moonbeam_309M.pt"

# Path to the model configuration file
model_config_path = "/Users/savi/Documents/PYTHON_PROJECTS/MoonbeamProject/Moonbeam-MIDI-Foundation_Model/src/llama_recipes/configs/model_config_small.json"

# Placeholder for tokenizer path (MusicTokenizer doesn't load from a file in the same way as AutoTokenizer)
tokenizer_path = ""

# Define max_seq_len and max_batch_size (adjust as needed)
max_seq_len = 2048
max_batch_size = 1

def load_moonbeam_model():
    """Loads the Moonbeam MIDI Foundation Model and its tokenizer."""
    print(f"Loading model from checkpoint: {ckpt_dir}")
    print(f"Using model configuration: {model_config_path}")
    try:
        # MusicLlama.build handles loading the model and tokenizer
        music_llama_instance = MusicLlama.build(
            ckpt_dir=ckpt_dir,
            model_config_path=model_config_path,
            tokenizer_path=tokenizer_path, # This might be ignored or used differently by MusicTokenizer
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            finetuned_PEFT_weight_path=None, # No PEFT weights for now
        )
        print("Model and tokenizer loaded successfully!")
        return music_llama_instance.model, music_llama_instance.tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_moonbeam_model()
    if model and tokenizer:
        print("Model and tokenizer are ready for use.")
        print("You can now add your conditional chord generation logic here.")
        # Example: You can now use model and tokenizer for generation
        # For instance, if you have a prompt in the format expected by MusicLlama.generate
        # prompt_tokens = [[[...]]]
        # generated_output = music_llama_instance.generate(prompt_tokens, max_gen_len=100)