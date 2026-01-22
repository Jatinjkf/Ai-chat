import json
import os
import threading
import queue
import tempfile
import time
import wave
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from llama_cpp import Llama
import onnxruntime as ort

APP_TITLE = "AnimeChat AI"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CHARACTERS_PATH = os.path.join(os.path.dirname(__file__), "characters.json")

DEFAULT_CONFIG = {
    "llm_gguf_path": "",
    "voice_onnx_path": "",
    "compute_mode": "cpu",
    "threads": 4,
    "context_preset": "medium",
    "anime_consistency": 0.8,
    "voice_enabled": True,
    "voice_pitch": 0,
    "voice_speed": 1.0,
    "verbosity": "short",
}

CONTEXT_PRESETS = {
    "low": {"n_ctx": 1024, "max_new_tokens": 128},
    "medium": {"n_ctx": 2048, "max_new_tokens": 256},
    "high": {"n_ctx": 4096, "max_new_tokens": 512},
}

VERBOSITY_TOKENS = {
    "short": 120,
    "medium": 220,
    "long": 360,
}


@dataclass
class Character:
    name: str
    personality: str
    backstory: str
    anime_mode: bool


class AppState:
    def __init__(self) -> None:
        self.llm: Optional[Llama] = None
        self.llm_status: str = "No model loaded"
        self.voice_session: Optional[ort.InferenceSession] = None
        self.voice_status: str = "Voice not loaded"
        self.voice_provider: str = "CPU"
        self.voice_error: Optional[str] = None
        self.config = load_config()
        self.characters = load_characters()


def load_config() -> Dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_characters() -> List[Character]:
    if os.path.exists(CHARACTERS_PATH):
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return [Character(**item) for item in raw]
    return []


def save_characters(characters: List[Character]) -> None:
    with open(CHARACTERS_PATH, "w", encoding="utf-8") as handle:
        json.dump([asdict(c) for c in characters], handle, indent=2)


def get_temperature(consistency: float) -> float:
    return max(0.5, 1.3 - 0.7 * consistency)


def build_prompt(
    history: List[Tuple[str, str]],
    character: Character,
    group_mode: bool,
    verbosity: str,
) -> str:
    lines = []
    if character.anime_mode:
        lines.append(
            "System: You are writing anime-style dialogue with expressive yet concise replies."
        )
    lines.append(f"System: Character name is {character.name}.")
    lines.append(f"System: Personality: {character.personality}.")
    if character.backstory.strip():
        lines.append(f"System: Backstory: {character.backstory}.")
    lines.append(f"System: Verbosity preference: {verbosity}.")
    lines.append("System: Keep responses short unless asked for more detail.")

    for role, content in history[-10:]:
        lines.append(f"{role}: {content}")

    if group_mode:
        lines.append(f"User: Please respond as {character.name}.")
    return "\n".join(lines)


def stream_generate(
    llm: Llama,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
):
    output_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()

    def worker():
        for token in llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        ):
            text = token["choices"][0]["text"]
            output_queue.put(text)
        done_event.set()

    threading.Thread(target=worker, daemon=True).start()

    while not done_event.is_set() or not output_queue.empty():
        try:
            chunk = output_queue.get(timeout=0.05)
            yield chunk
        except queue.Empty:
            time.sleep(0.01)


def load_llm(path: str, threads: int, context_preset: str) -> Tuple[Optional[Llama], str]:
    if not path:
        return None, "LLM file not found"
    if not os.path.exists(path):
        return None, "LLM file not found"
    preset = CONTEXT_PRESETS.get(context_preset, CONTEXT_PRESETS["medium"])
    llm = Llama(model_path=path, n_ctx=preset["n_ctx"], n_threads=threads)
    return llm, "LLM loaded"


def select_voice_providers(mode: str) -> List[str]:
    if mode in {"igpu", "hybrid"}:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_voice(path: str, mode: str) -> Tuple[Optional[ort.InferenceSession], str, str]:
    if not path:
        return None, "Voice ONNX not valid", "CPU"
    if not os.path.exists(path):
        return None, "Voice ONNX not valid", "CPU"
    providers = select_voice_providers(mode)
    try:
        session = ort.InferenceSession(path, providers=providers)
    except Exception:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return session, "DirectML not available, using CPU", "CPU"
    active_provider = session.get_providers()[0] if session.get_providers() else "CPU"
    if "DmlExecutionProvider" in active_provider:
        return session, "Voice loaded", "DirectML"
    return session, "Voice loaded", "CPU"


def validate_voice_model(session: ort.InferenceSession) -> Optional[str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        return "Voice model format not supported by this build."
    input_type = inputs[0].type
    if "string" not in input_type:
        return "Voice model format not supported by this build."
    return None


def run_voice(
    session: ort.InferenceSession,
    text: str,
    pitch: int,
    speed: float,
    sample_rate: int = 22050,
) -> Tuple[Optional[str], Optional[str]]:
    error = validate_voice_model(session)
    if error:
        return None, error
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.array([text])})
    audio = np.asarray(outputs[0]).squeeze()
    if audio.ndim != 1:
        return None, "Voice model format not supported by this build."

    if speed != 1.0:
        indices = np.arange(0, len(audio), speed)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return path, None


def ensure_state() -> AppState:
    return AppState()


def format_status(state: AppState) -> str:
    return f"LLM: {state.llm_status} | Voice: {state.voice_status}"


def load_models(state: AppState) -> AppState:
    state.llm, state.llm_status = load_llm(
        state.config["llm_gguf_path"],
        int(state.config["threads"]),
        state.config["context_preset"],
    )
    session, status, provider = load_voice(
        state.config["voice_onnx_path"],
        state.config["compute_mode"],
    )
    state.voice_session = session
    state.voice_status = status
    state.voice_provider = provider
    return state


def setup_save(
    state: AppState,
    llm_path: str,
    voice_path: str,
    compute_mode: str,
):
    state.config["llm_gguf_path"] = llm_path
    state.config["voice_onnx_path"] = voice_path
    state.config["compute_mode"] = compute_mode
    save_config(state.config)
    load_models(state)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        format_status(state),
        state,
    )


def update_settings(
    state: AppState,
    compute_mode: str,
    threads: int,
    context_preset: str,
    anime_consistency: float,
    verbosity: str,
):
    state.config["compute_mode"] = compute_mode
    state.config["threads"] = int(threads)
    state.config["context_preset"] = context_preset
    state.config["anime_consistency"] = float(anime_consistency)
    state.config["verbosity"] = verbosity
    save_config(state.config)
    return "Settings saved", state


def update_models(state: AppState, llm_path: str, voice_path: str):
    state.config["llm_gguf_path"] = llm_path
    state.config["voice_onnx_path"] = voice_path
    save_config(state.config)
    return "Paths saved", state


def reload_llm(state: AppState):
    state.llm, state.llm_status = load_llm(
        state.config["llm_gguf_path"],
        int(state.config["threads"]),
        state.config["context_preset"],
    )
    return format_status(state), state


def reload_voice(state: AppState):
    session, status, provider = load_voice(
        state.config["voice_onnx_path"],
        state.config["compute_mode"],
    )
    state.voice_session = session
    state.voice_status = status
    state.voice_provider = provider
    return format_status(state), state


def add_character(
    state: AppState,
    name: str,
    personality: str,
    backstory: str,
    anime_mode: bool,
):
    if not name.strip() or len(state.characters) >= 8:
        return "Character name required (max 8)", gr.update(), state
    state.characters.append(
        Character(
            name=name.strip(),
            personality=personality.strip(),
            backstory=backstory.strip(),
            anime_mode=anime_mode,
        )
    )
    save_characters(state.characters)
    choices = [c.name for c in state.characters]
    return "Character added", gr.update(choices=choices, value=choices), state


def delete_character(state: AppState, selected: List[str]):
    remaining = [c for c in state.characters if c.name not in (selected or [])]
    state.characters = remaining
    save_characters(state.characters)
    choices = [c.name for c in state.characters]
    return "Character deleted", gr.update(choices=choices, value=choices), state


def edit_character(
    state: AppState,
    selected: List[str],
    name: str,
    personality: str,
    backstory: str,
    anime_mode: bool,
):
    if not selected:
        return "Select a character to edit", gr.update(), state
    target = selected[0]
    for character in state.characters:
        if character.name == target:
            character.name = name.strip() or character.name
            character.personality = personality.strip()
            character.backstory = backstory.strip()
            character.anime_mode = anime_mode
            break
    save_characters(state.characters)
    choices = [c.name for c in state.characters]
    return "Character updated", gr.update(choices=choices, value=[name.strip() or target]), state


def fill_character_fields(state: AppState, selected: List[str]):
    if not selected:
        return "", "", "", False
    target = selected[0]
    for character in state.characters:
        if character.name == target:
            return character.name, character.personality, character.backstory, character.anime_mode
    return "", "", "", False


def export_chat(chat_history: List[Tuple[str, str]]):
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as handle:
        for role, message in chat_history:
            handle.write(f"{role}: {message}\n")
    return path


def chat_reply(
    state: AppState,
    chat_history: List[Tuple[str, str]],
    user_message: str,
    selected_characters: List[str],
    group_mode: bool,
    generate_voice: bool,
):
    if not state.llm:
        chat_history.append(("System", "Load your GGUF model in Models tab"))
        return chat_history, "", None, state
    if not user_message.strip():
        return chat_history, "", None, state
    if not selected_characters:
        chat_history.append(("System", "Select at least one character"))
        return chat_history, "", None, state

    chat_history.append(("User", user_message.strip()))
    generated_audio = None
    temperature = get_temperature(float(state.config["anime_consistency"]))
    verbosity = state.config.get("verbosity", "short")
    max_tokens = VERBOSITY_TOKENS.get(verbosity, 160)

    def append_and_yield(name: str, message: str):
        chat_history.append((name, message))

    for name in selected_characters:
        character = next((c for c in state.characters if c.name == name), None)
        if not character:
            continue
        prompt = build_prompt(chat_history, character, group_mode, verbosity)
        response_text = ""
        for chunk in stream_generate(
            state.llm,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "System:"],
        ):
            response_text += chunk
            temp_history = chat_history + [(character.name, response_text)]
            yield temp_history, "", None, state
        append_and_yield(character.name, response_text.strip())

        if (
            generate_voice
            and state.config.get("voice_enabled", True)
            and state.voice_session
        ):
            audio_path, error = run_voice(
                state.voice_session,
                response_text,
                int(state.config.get("voice_pitch", 0)),
                float(state.config.get("voice_speed", 1.0)),
            )
            if error:
                chat_history.append(("System", error))
            else:
                generated_audio = audio_path

        if not group_mode:
            break

    yield chat_history, "", generated_audio, state


def update_voice_settings(state: AppState, enabled: bool, pitch: int, speed: float):
    state.config["voice_enabled"] = enabled
    state.config["voice_pitch"] = int(pitch)
    state.config["voice_speed"] = float(speed)
    save_config(state.config)
    return "Voice settings saved", state


def test_voice(state: AppState):
    if not state.voice_session:
        return None, "Voice ONNX not valid"
    if not state.config.get("voice_enabled", True):
        return None, "Voice is disabled"
    audio_path, error = run_voice(
        state.voice_session,
        "Hello! This is a voice test.",
        int(state.config.get("voice_pitch", 0)),
        float(state.config.get("voice_speed", 1.0)),
    )
    if error:
        return None, error
    return audio_path, "Voice test complete"


def build_ui():
    state = gr.State(ensure_state())

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        status_bar = gr.Markdown("Loading...")

        setup_visible = not os.path.exists(CONFIG_PATH)

        with gr.Column(visible=setup_visible) as setup_screen:
            gr.Markdown("# Setup")
            gr.Markdown("Follow the steps below to start.")
            setup_llm = gr.Textbox(label="Step 1: Select Chat Model (GGUF)")
            setup_voice = gr.Textbox(label="Step 2: Select Voice Model (ONNX)")
            setup_compute = gr.Dropdown(
                choices=["cpu", "igpu", "hybrid"],
                value="cpu",
                label="Step 3: Choose Compute Mode",
            )
            setup_save_btn = gr.Button("Save & Start", variant="primary")

        with gr.Column(visible=not setup_visible) as main_screen:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Characters")
                    characters_list = gr.CheckboxGroup(label="Characters")
                    character_name = gr.Textbox(label="Name")
                    character_personality = gr.Textbox(label="Personality", lines=2)
                    character_backstory = gr.Textbox(label="Backstory", lines=2)
                    character_anime = gr.Checkbox(label="Anime Mode", value=True)
                    character_add = gr.Button("Add Character")
                    character_edit = gr.Button("Edit Character")
                    character_delete = gr.Button("Delete Character")
                    group_mode = gr.Checkbox(label="Group Chat", value=False)
                    active_count = gr.Markdown("Active Characters: 0")

                with gr.Column(scale=5):
                    gr.Markdown("## Chat")
                    chat_box = gr.Chatbot(height=420)
                    user_input = gr.Textbox(label="Message", placeholder="Type your message")
                    send_btn = gr.Button("Send", variant="primary")
                    generate_voice = gr.Checkbox(label="Generate Voice", value=True)
                    export_btn = gr.Button("Export Chat")
                    export_file = gr.File(label="Download", visible=False)

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("Settings"):
                            settings_compute = gr.Dropdown(
                                choices=["cpu", "igpu", "hybrid"],
                                label="Compute mode",
                            )
                            settings_threads = gr.Slider(2, 8, step=1, label="Threads")
                            settings_context = gr.Dropdown(
                                choices=["low", "medium", "high"],
                                label="Context preset",
                            )
                            settings_consistency = gr.Slider(
                                0.5, 1.0, step=0.05, label="Anime consistency"
                            )
                            settings_verbosity = gr.Dropdown(
                                choices=["short", "medium", "long"],
                                label="Verbosity",
                            )
                            settings_save = gr.Button("Save Settings")
                            settings_msg = gr.Markdown()

                        with gr.Tab("Models"):
                            model_llm = gr.Textbox(label="LLM GGUF path")
                            model_voice = gr.Textbox(label="Voice ONNX path")
                            model_save = gr.Button("Save Paths")
                            model_reload_llm = gr.Button("Reload LLM")
                            model_reload_voice = gr.Button("Reload Voice")
                            model_msg = gr.Markdown()

                        with gr.Tab("Voice"):
                            voice_enable = gr.Checkbox(label="Enable Voice", value=True)
                            voice_pitch = gr.Slider(-12, 12, step=1, label="Pitch")
                            voice_speed = gr.Slider(0.5, 2.0, step=0.05, label="Speed")
                            voice_save = gr.Button("Save Voice Settings")
                            voice_test = gr.Button("Test Voice")
                            voice_audio = gr.Audio(label="Last generated audio")
                            voice_msg = gr.Markdown()

                        with gr.Tab("About"):
                            gr.Markdown(
                                "Place your GGUF and ONNX model files anywhere on your PC, then "
                                "set their paths in the Models tab. Use Settings to adjust compute "
                                "mode and context size."
                            )

        def initialize(state_value: AppState):
            state_value = load_models(state_value)
            choices = [c.name for c in state_value.characters]
            return (
                format_status(state_value),
                gr.update(choices=choices, value=choices),
                len(choices),
                state_value,
            )

        demo.load(
            initialize,
            inputs=state,
            outputs=[status_bar, characters_list, active_count, state],
        )

        setup_save_btn.click(
            setup_save,
            inputs=[state, setup_llm, setup_voice, setup_compute],
            outputs=[setup_screen, main_screen, status_bar, state],
        )

        characters_list.change(
            lambda values: f"Active Characters: {len(values or [])}",
            inputs=characters_list,
            outputs=active_count,
        )

        character_add.click(
            add_character,
            inputs=[
                state,
                character_name,
                character_personality,
                character_backstory,
                character_anime,
            ],
            outputs=[model_msg, characters_list, state],
        )

        character_edit.click(
            edit_character,
            inputs=[
                state,
                characters_list,
                character_name,
                character_personality,
                character_backstory,
                character_anime,
            ],
            outputs=[model_msg, characters_list, state],
        )

        character_delete.click(
            delete_character,
            inputs=[state, characters_list],
            outputs=[model_msg, characters_list, state],
        )

        characters_list.change(
            fill_character_fields,
            inputs=[state, characters_list],
            outputs=[
                character_name,
                character_personality,
                character_backstory,
                character_anime,
            ],
        )

        settings_save.click(
            update_settings,
            inputs=[
                state,
                settings_compute,
                settings_threads,
                settings_context,
                settings_consistency,
                settings_verbosity,
            ],
            outputs=[settings_msg, state],
        )

        model_save.click(
            update_models,
            inputs=[state, model_llm, model_voice],
            outputs=[model_msg, state],
        )

        model_reload_llm.click(
            reload_llm,
            inputs=[state],
            outputs=[status_bar, state],
        )

        model_reload_voice.click(
            reload_voice,
            inputs=[state],
            outputs=[status_bar, state],
        )

        voice_save.click(
            update_voice_settings,
            inputs=[state, voice_enable, voice_pitch, voice_speed],
            outputs=[voice_msg, state],
        )

        voice_test.click(
            test_voice,
            inputs=[state],
            outputs=[voice_audio, voice_msg],
        )

        export_btn.click(
            export_chat, inputs=[chat_box], outputs=[export_file]
        ).then(lambda: gr.update(visible=True), outputs=[export_file])

        send_btn.click(
            chat_reply,
            inputs=[
                state,
                chat_box,
                user_input,
                characters_list,
                group_mode,
                generate_voice,
            ],
            outputs=[chat_box, user_input, voice_audio, state],
        )

        def refresh_fields(state_value: AppState):
            config = state_value.config
            return (
                config["llm_gguf_path"],
                config["voice_onnx_path"],
                config["compute_mode"],
                config["threads"],
                config["context_preset"],
                config["anime_consistency"],
                config.get("verbosity", "short"),
                config.get("voice_enabled", True),
                config.get("voice_pitch", 0),
                config.get("voice_speed", 1.0),
            )

        demo.load(
            refresh_fields,
            inputs=state,
            outputs=[
                model_llm,
                model_voice,
                settings_compute,
                settings_threads,
                settings_context,
                settings_consistency,
                settings_verbosity,
                voice_enable,
                voice_pitch,
                voice_speed,
            ],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
