import os
import glob
import torch
from load_moonbeam_model import load_moonbeam_model
from src.llama_recipes.datasets.music_tokenizer import MusicTokenizer
import mido

# Define paths
INPUT_MIDI_FOLDER = "/Users/savi/Documents/PYTHON_PROJECTS/MoonbeamProject/input_midi"
OUTPUT_MIDI_FOLDER = "/Users/savi/Documents/PYTHON_PROJECTS/MoonbeamProject/output_midi"

NUM_VARIATIONS = 5
CHORDS_TO_ADD = 10 # Approximate number of new chords to generate
MAX_GEN_LEN_PER_CHORD = 6 # Assuming max 6 notes per chord for generation length
MAX_GEN_LEN = CHORDS_TO_ADD * MAX_GEN_LEN_PER_CHORD # Total tokens to generate

def extend_midi_progression():
    print("Loading Moonbeam model...")
    model_instance, tokenizer = load_moonbeam_model()

    if model_instance is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return

    print("Model loaded successfully. Starting MIDI extension.")

    midi_files = glob.glob(os.path.join(INPUT_MIDI_FOLDER, "*.mid"))
    midi_files.extend(glob.glob(os.path.join(INPUT_MIDI_FOLDER, "*.midi")))

    if not midi_files:
        print(f"No MIDI files found in {INPUT_MIDI_FOLDER}. Please place your MIDI files there.")
        return

    for midi_file_path in midi_files:
        file_name = os.path.basename(midi_file_path)
        print(f"\nProcessing: {file_name}")

        try:
            # 1. Load and tokenize input MIDI
            original_compound_tokens = MusicTokenizer.midi_to_compound(midi_file_path)
            if not original_compound_tokens:
                print(f"Warning: No musical events found in {file_name}. Skipping.")
                continue

            # The model expects a list of lists of lists for prompt_tokens
            # And encode_series_labels expects a list of lists
            # So we need to convert original_compound_tokens (list of lists) to the expected format
            # The model's generate method expects a batch, so we wrap it in another list.
            original_compound_tokens_with_sos = tokenizer.encode_series(
                original_compound_tokens, if_add_sos=True, if_add_eos=False
            )
            prompt_tokens_for_model = [original_compound_tokens_with_sos]

            for i in range(NUM_VARIATIONS):
                print(f"  Generating variation {i + 1}/{NUM_VARIATIONS}...")
                # 2. Generate new tokens
                generated_output = model_instance.music_completion(
                    prompt_tokens=prompt_tokens_for_model,
                    max_gen_len=MAX_GEN_LEN,
                    temperature=0.7, # Adjust for more/less randomness
                    top_p=0.9,       # Adjust for more/less diversity
                    logprobs=False,

                )

                # The output is a list of dictionaries, we need the 'tokens' from the 'generation' key
                # generated_tokens will be a list of lists of lists (batch, seq_len, 6)
                generated_tokens_language = generated_output[0]["generation"]["tokens"]

                # Convert language tokens back to compound tokens
                # The model's output includes the prompt, so we need to slice it to get only the generated part
                # The tokenizer.convert_from_language_tokens expects a tensor, so convert it.
                generated_compound_tokens_tensor = tokenizer.convert_from_language_tokens(
                    torch.tensor(generated_tokens_language)
                )
                generated_compound_tokens = generated_compound_tokens_tensor.tolist()

                # Slice to get only the newly generated part (after the original prompt)
                # This assumes the model echoes the prompt. Check the 'echo' parameter in generate.
                # If echo=True, the output contains the prompt + generated. We want only the generated.
                # The length of the original prompt in language tokens:
                original_prompt_len_language = len(prompt_tokens_for_model[0])
                newly_generated_compound_tokens = generated_compound_tokens[original_prompt_len_language:]

                # 3. Combine original and generated tokens
                combined_compound_tokens = original_compound_tokens + newly_generated_compound_tokens

                # 4. Convert back to MIDI and save
                output_midi = MusicTokenizer.compound_to_midi(combined_compound_tokens)
                output_file_path = os.path.join(OUTPUT_MIDI_FOLDER, f"{os.path.splitext(file_name)[0]}_extended_var{i+1}.mid")
                output_midi.save(output_file_path)
                print(f"    Saved variation {i+1} to: {output_file_path}")

        except Exception as e:
            import traceback
            print(f"Error processing {file_name}: {e}")
            traceback.print_exc()

import sys

LOG_FILE = "moonbeam_debug.log"

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Ensure immediate write
    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == "__main__":
    f = open(LOG_FILE, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, f)
    extend_midi_progression()
    sys.stdout = original_stdout # Restore stdout
    f.close()
