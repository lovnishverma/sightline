import gradio as gr
from src.ui.styles import CSS
from src.config import CONFIG

def _status_html(msg: str) -> str:
    """Wrap a status message in the styled status-bar div."""
    return f'<div id="sightline-status" role="status" aria-live="polite">{msg}</div>'

def build_ui(assistant) -> gr.Blocks:
    auto_start_js = """
    function() {
        setInterval(function() {
            let clickTarget = function(btn) {
                let text = (btn.textContent || btn.innerText || '').toLowerCase().trim();
                if (text === 'click to access webcam' || text === 'record') {
                    btn.click();
                }
            };
            
            // Check Shadow DOMs
            document.querySelectorAll('*').forEach(function(el) {
                if (el.shadowRoot) {
                    el.shadowRoot.querySelectorAll('button').forEach(clickTarget);
                }
            });
            
            // Check regular DOM
            document.querySelectorAll('button').forEach(clickTarget);
        }, 1000); // Check every second indefinitely
    }
    """
    with gr.Blocks(title=f"{CONFIG.APP_NAME} | INNOpreneur 2026 @IIT Ropar and Jugaad mela 5.0", js=auto_start_js, css=CSS) as demo:
        gr.HTML('<div class="sr-only" aria-live="assertive" id="aria-live-region" role="status"></div>')
        
        status_bar = gr.HTML(_status_html("Ready — Press D to describe"))

        gr.HTML(f"""
        <div class="sightline-header" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px; margin: 10px 0 20px 0; padding: 14px 20px; background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); border-radius: 12px; border: 1px solid #cbd5e1; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="color: var(--accent, #2563eb);">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                    <circle cx="12" cy="12" r="3"/>
                </svg>
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; display: inline-block; color: #0f172a;">{CONFIG.APP_NAME}</h1>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1e40af; margin-top: 3px;">
                        INNOpreneur 2026 @IIT Ropar and Jugaad mela 5.0
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                <span style="background: #1e40af; color: #ffffff; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.3px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    INNOpreneur 2026 @IIT Ropar
                </span>
                <span style="background: #ea580c; color: #ffffff; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.3px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    Jugaad mela 5.0
                </span>
            </div>
        </div>
        """)
        
        rt_state = gr.State(True)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    describe_btn = gr.Button("Describe (D)", variant="primary", elem_id="btn-describe")
                    realtime_btn = gr.Button("Stop Realtime (R)", variant="secondary", elem_id="btn-realtime")
                
                with gr.Row():
                    task_radio = gr.Radio(
                        choices=["Quick Glance", "Detailed Scene", "Immersive Description", "Read Text", "Ask Question"],
                        value="Detailed Scene",
                        label="Mode"
                    )
            
            with gr.Column(scale=1):
                with gr.Row():
                    repeat_btn = gr.Button("Repeat (P)", elem_id="btn-repeat")
                    stop_btn = gr.Button("Stop (Esc)", elem_id="btn-stop")
                
                voice_dropdown = gr.Dropdown(
                    choices=["English (US) - Aria", "English (US) - Guy", "English (UK) - Sonia", "Hindi - Swara", "Hindi - Madhur"],
                    value="English (US) - Aria",
                    label="Voice"
                )
        
        question_box = gr.Textbox(
            label="Ask a Question (for 'Ask Question' mode)",
            placeholder="e.g. Where are my keys?",
            lines=1
        )

        with gr.Row():
            with gr.Column(scale=1):
                webcam = gr.Image(label="Camera", type="numpy", sources=["webcam"], streaming=True)
                with gr.Accordion("Upload Image", open=False):
                    upload = gr.Image(label="Upload", type="numpy", sources=["upload"])

            with gr.Column(scale=1):
                caption_box = gr.Textbox(label="Description", lines=14, interactive=False)
                audio_player = gr.Audio(label="Audio", type="filepath", autoplay=True)

        components = {
            "webcam": webcam, "upload": upload, "task_radio": task_radio,
            "voice_dropdown": voice_dropdown, "question_box": question_box,
            "describe_btn": describe_btn,
            "realtime_btn": realtime_btn, "caption_box": caption_box,
            "audio_player": audio_player, "status_bar": status_bar,
            "rt_state": rt_state, "repeat_btn": repeat_btn, "stop_btn": stop_btn
        }
        
        from src.ui.events import wire_events
        wire_events(demo, components, assistant)
        
        return demo
