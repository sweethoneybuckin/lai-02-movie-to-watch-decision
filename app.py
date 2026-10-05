import os
import threading
import gradio as gr
import moviedecision

def start_bot():
    moviedecision.main()

# Jalankan bot Discord di background thread
threading.Thread(target=start_bot, daemon=True).start()

# UI sederhana untuk status web Gradio Hugging Face
with gr.Blocks(title="Movie Decision Bot") as demo:
    gr.Markdown("# 🎬 Movie Decision Bot Status")
    gr.Markdown("🟢 Bot Discord aktif 24/7 di background!")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
