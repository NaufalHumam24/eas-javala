import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration, VitsModel, AutoTokenizer
from peft import PeftModel, PeftConfig
import io
import soundfile as sf

# Load model translasi
tokenizer_mt5 = T5Tokenizer.from_pretrained("indonlp/cendol-mt5-small-inst")
base_model = MT5ForConditionalGeneration.from_pretrained("indonlp/cendol-mt5-small-inst")
peft_config = PeftConfig.from_pretrained("./best_model_mt5")
model_mt5 = PeftModel.from_pretrained(base_model, "./best_model_mt5")
model_mt5.to("cuda" if torch.cuda.is_available() else "cpu")
model_mt5.eval()

# Load model TTS
tokenizer_tts = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")
model_tts = VitsModel.from_pretrained("facebook/mms-tts-ind")
model_tts.to("cuda" if torch.cuda.is_available() else "cpu")
model_tts.eval()

# Fungsi translasi
def translate_to_javanese(text):
    prompt = "terjemahkan ke bahasa jawa: " + text
    inputs = tokenizer_mt5(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model_mt5.device)
    output_ids = model_mt5.generate(**inputs, max_length=512, num_beams=4)
    return tokenizer_mt5.decode(output_ids[0], skip_special_tokens=True)

# Fungsi TTS
def text_to_speech(text):
    inputs = tokenizer_tts(text, return_tensors="pt").to(model_tts.device)
    with torch.no_grad():
        audio = model_tts(**inputs).waveform.squeeze().cpu().numpy()
    sampling_rate = model_tts.config.sampling_rate
    return sampling_rate, audio
