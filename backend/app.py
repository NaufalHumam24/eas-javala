import gradio as gr
import pandas as pd
import glob
from backend import translate_to_javanese, text_to_speech

# Load cerita
cerita_data = []
for file in glob.glob("cerita/*.csv"):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        cerita_data.append({
            "judul": row["judul"],
            "provinsi": row["provinsi"],
            "id_teks": row["id_teks"],
            "teks": row["teks"]
        })
cerita_df = pd.DataFrame(cerita_data)

# Fungsi untuk Gradio
def process(provinsi, judul):
    rows = cerita_df[(cerita_df["provinsi"] == provinsi) & (cerita_df["judul"] == judul)]
    rows = rows.sort_values(by="id_teks")

    teks_list = rows["teks"].tolist()
    hasil_list = [translate_to_javanese(paragraf) for paragraf in teks_list]

    teks_utuh = "\n\n".join(teks_list)
    hasil_terjemahan_utuh = "\n\n".join(hasil_list)

    sampling_rate, audio = text_to_speech(hasil_terjemahan_utuh)

    return teks_utuh, hasil_terjemahan_utuh, (sampling_rate, audio)

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Aplikasi Penerjemahan Cerita Rakyat")

    provinsi_dropdown = gr.Dropdown(label="Pilih Provinsi", choices=sorted(cerita_df["provinsi"].unique()))
    judul_dropdown = gr.Dropdown(label="Pilih Cerita")
    btn = gr.Button("Terjemahkan")

    input_area = gr.Textbox(label="Cerita Bahasa Indonesia", lines=10)
    translated_area = gr.Textbox(label="Cerita Bahasa Jawa", lines=10)
    audio_out = gr.Audio(label="Audio", type="numpy")

    def update_judul(provinsi):
        filtered = cerita_df[cerita_df["provinsi"] == provinsi]
        return gr.update(choices=sorted(filtered["judul"].unique()))

    provinsi_dropdown.change(fn=update_judul, inputs=provinsi_dropdown, outputs=judul_dropdown)
    btn.click(fn=process, inputs=[provinsi_dropdown, judul_dropdown], outputs=[input_area, translated_area, audio_out])

demo.launch()
