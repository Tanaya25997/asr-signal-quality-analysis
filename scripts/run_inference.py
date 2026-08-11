import os, time
import torch 
import torchaudio 
import pandas as pd 
import jiwer 
from datasets import load_dataset 
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC
)

DATA_CACHE = "/data/commonvoice-sample"
HF_CACHE = "/data/hf_cache"
NUM_SAMPLES = 1500
OUTPUT_CSV = "results/predictions.csv"


os.makedirs("results", exist_ok=True)
os.makedirs(DATA_CACHE, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

device = "cpu"

#### load commonvoice sample 

print("Loading CommonVoice samples...")
start = time.time() 

dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "en",
    split="test",
    cache_dir=DATA_CACHE,
    revision="refs/convert/parquet"
    #trust_remote_code=True
)

dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

print(f"Loaded {len(dataset)} samples in {time.time() - start:.2f}s ")


### Load models 

start = time.time() 

print("Loading models....")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", cache_dir="/data/hf_cache")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", cache_dir="/data/hf_cache").to(device)
whisper_model.eval()

w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/data/hf_cache")
w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/data/hf_cache").to(device)
w2v_model.eval()

print(f"Models Loaded in {time.time() - start}s")


### Inference Loop 

results = []

print("Running inference...")

start = time.time()

for i, sample in enumerate(dataset):
    audio_array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    reference = sample["sentence"].upper()

    ### ----------- whisper --------------###
    whisper_inputs = whisper_processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features.to(device) #signal to features or signal amplitude to tokens
    with torch.no_grad():
        whisper_ids = whisper_model.generate(whisper_inputs) ### input tokens to output tokens
    whisper_pred = whisper_processor.batch_decode(whisper_ids, skip_special_tokens=True)[0].upper() ## token to text, strop special tokens like start/end markers etc 

    ### ----------- Wav2Vec2 --------------###
    w2c_inputs = w2v_processor(audio_array, sampling_rate=sr, return_tensors="pt").input_values.to(device) # takes raw normalized input values, not features like spectrograms etc 
    with torch.no_grad():
        logits = w2v_model(w2c_inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    w2v_pred = w2v_processor.batch_decode(pred_ids)[0]

    ### ---- WER per sample ---- ###
    whisper_wer = jiwer.wer(reference, whisper_pred)
    w2v_wer = jiwer.wer(reference, w2v_pred)

    results.append({
        "file_id": sample["path"],
        "reference": reference,
        "whisper_pred": whisper_pred,
        "whisper_wer": whisper_wer,
        "w2v_pred": w2v_pred,
        "w2v_wer": w2v_wer,
    })

    if i % 50 == 0:
        elapsed = time.time() - start
        print(f"[{i}/{len(dataset)}] elapsed={elapsed:.1f}s")

print(f"Inference done in {time.time()-start:.1f}s")


df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_CSV}")


'''

Wav2Vec2 (CTC) does a single forward pass, producing character-score predictions for every timestep in parallel, 
then collapses repeated characters and removes blank tokens to get the final text — requiring an explicit `argmax` 
and manual decoding step. 

so each character in the vocab (a-z, space, | PAD, UNK) gets a prob at eac timestep and the max prob character is 
the ouput char at that timestep. 

Then lets say if its decoded as C C _ _ A A _ _ T T T _
then CTC does -> C _ A _ T _ -> CAT (after dropping special characters like _)


Whisper is autoregressive (encoder-decoder): it generates output one token at a time, 
feeding each predicted token back in as context for predicting the next, repeating until it emits an end-of-sequence 
token -> this entire loop, including the token-selection step, is handled internally by `.generate()`, which is why no
explicit argmax appears in the code. In short: CTC predicts everything at once and cleans it up afterward; Whisper predicts 
sequentially, one token building on the last.

so CTC does P(next| input) autoregrssively 
'''