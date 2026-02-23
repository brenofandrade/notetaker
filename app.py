import mimetypes
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import assemblyai as aai
from openai import OpenAI

# =========================================================
# Configuração inicial
# =========================================================
load_dotenv(override=True)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")  # altere se preferir outro modelo

SAVE_DIR = Path("transcricoes_salvas")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Configura a AssemblyAI
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

# Cliente OpenAI (só cria se chave existir)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================================================
# Helpers
# =========================================================
def sanitize_filename(name: str) -> str:
    """Sanitiza nome de arquivo para evitar caracteres inválidos."""
    name = name.strip()
    name = re.sub(r"[^\w\-. ]", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name or "transcricao"


def save_text_locally(content: str, base_name: str, suffix: str = "") -> Path:
    """Salva texto localmente (no filesystem da máquina/servidor que roda o Streamlit)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base = sanitize_filename(base_name)
    final_name = f"{safe_base}{suffix}_{timestamp}.txt"
    file_path = SAVE_DIR / final_name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def guess_audio_mime(filename: str) -> str:
    """Tenta inferir MIME para o st.audio."""
    mime, _ = mimetypes.guess_type(filename or "")
    if mime and (mime.startswith("audio/") or mime.startswith("video/")):
        return mime

    # fallback comum
    ext = (Path(filename).suffix or "").lower()
    if ext in [".mp3", ".mpeg"]:
        return "audio/mpeg"
    if ext in [".wav"]:
        return "audio/wav"
    if ext in [".ogg"]:
        return "audio/ogg"
    if ext in [".m4a", ".mp4"]:
        return "audio/mp4"
    if ext in [".webm"]:
        return "audio/webm"
    if ext in [".flac"]:
        return "audio/flac"
    return "audio/wav"


def extract_utterances_from_transcript(transcript) -> list[dict]:
    """
    Extrai utterances do objeto de transcript da AssemblyAI em formato serializável.
    Cada item: {speaker, text, start, end}
    """
    utterances = []
    raw_utterances = getattr(transcript, "utterances", None) or []

    for utt in raw_utterances:
        speaker = getattr(utt, "speaker", None)
        text = getattr(utt, "text", None)
        start = getattr(utt, "start", None)
        end = getattr(utt, "end", None)

        if text:
            utterances.append(
                {
                    "speaker": str(speaker) if speaker is not None else "Speaker",
                    "text": str(text).strip(),
                    "start": start,
                    "end": end,
                }
            )
    return utterances


def get_detected_speakers(utterances: list[dict]) -> list[str]:
    """Retorna lista única e ordenada de speakers encontrados nas utterances."""
    speakers = []
    seen = set()
    for utt in utterances:
        spk = (utt.get("speaker") or "Speaker").strip()
        if spk and spk not in seen:
            seen.add(spk)
            speakers.append(spk)
    return speakers


def build_speaker_labeled_transcript(
    utterances: list[dict],
    speaker_name_map: dict[str, str] | None = None,
) -> str:
    """
    Monta texto da transcrição no formato:
    Nome do Orador: fala...
    """
    if not utterances:
        return ""

    speaker_name_map = speaker_name_map or {}
    lines = []

    for utt in utterances:
        original_speaker = (utt.get("speaker") or "Speaker").strip()
        display_speaker = (speaker_name_map.get(original_speaker) or original_speaker).strip()
        text = (utt.get("text") or "").strip()

        if text:
            lines.append(f"{display_speaker}: {text}")

    return "\n".join(lines).strip()


def refresh_transcript_text_from_state() -> None:
    """
    Atualiza st.session_state.transcript_text com base em:
    - utterances + mapeamento de oradores (se houver diarização)
    - fallback para transcript_raw_text
    """
    utterances = st.session_state.get("transcript_utterances", [])
    raw_text = st.session_state.get("transcript_raw_text", "")
    speaker_map = st.session_state.get("speaker_name_map", {})

    if utterances:
        st.session_state.transcript_text = build_speaker_labeled_transcript(utterances, speaker_map)
    else:
        st.session_state.transcript_text = raw_text or ""


def transcribe_audio_with_assemblyai(audio_bytes: bytes, original_filename: str) -> dict:
    """
    Salva temporariamente o áudio e envia para transcrição com AssemblyAI.
    Retorna dict com texto bruto + utterances (se speaker diarization estiver ativo).
    """
    suffix = Path(original_filename).suffix or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro", "universal-2"],
            language_detection=True,
            speaker_labels=True,  # habilita diarização (oradores)
        )

        transcript = aai.Transcriber().transcribe(tmp_path, config=config)

        # Compatível com enum/string
        status = getattr(transcript, "status", None)
        if status == aai.TranscriptStatus.error or str(status).lower() == "error":
            raise RuntimeError(f"Falha na transcrição: {getattr(transcript, 'error', 'erro desconhecido')}")

        text = getattr(transcript, "text", "") or ""
        utterances = extract_utterances_from_transcript(transcript)

        return {
            "text": text,
            "utterances": utterances,
        }

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def build_reformat_prompt(style: str, transcript_text: str) -> str:
    """Monta prompt de reformatação em PT-BR."""
    style_instructions = {
        "padrão": (
            "Reescreva a transcrição em português claro, mantendo fidelidade total ao conteúdo, "
            "corrigindo pontuação, capitalização e pequenas repetições típicas de fala."
        ),
        "sumário executivo": (
            "Converta a transcrição em um sumário executivo em português, com foco em clareza, "
            "objetividade e decisões/pontos principais. Se fizer sentido, use tópicos curtos."
        ),
        "informal": (
            "Reescreva a transcrição em tom informal e natural, fácil de ler, sem perder o sentido original."
        ),
        "formal": (
            "Reescreva a transcrição em tom formal/profissional, com linguagem organizada, "
            "precisa e bem pontuada, sem alterar o significado."
        ),
    }

    instruction = style_instructions.get(style, style_instructions["padrão"])

    return f"""
Você é um editor de texto especializado em pós-processamento de transcrições.
Tarefa: {instruction}

Regras obrigatórias:
- Preserve o idioma original principal da transcrição (se estiver em português, mantenha em português).
- Não invente informações.
- Não omita conteúdo relevante.
- Melhore legibilidade e estrutura.
- Se houver trechos incompreensíveis, mantenha o trecho de forma neutra, sem inventar.
- Preserve a identificação dos oradores quando houver (ex.: "Breno:", "Entrevistador:").

Transcrição original:
\"\"\"
{transcript_text}
\"\"\"
""".strip()


def extract_response_text(response) -> str:
    """
    Extrai texto da resposta da OpenAI.
    Usa output_text quando disponível (conveniência do SDK) e tenta fallback.
    """
    text = getattr(response, "output_text", None)
    if text:
        return text

    # Fallback defensivo
    try:
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        parts.append(getattr(c, "text", ""))
        joined = "\n".join([p for p in parts if p]).strip()
        if joined:
            return joined
    except Exception:
        pass

    return ""


def reformat_transcript_with_openai(
    transcript_text: str,
    style: str,
    model_name: str,
    temperature: float = 0.2,
) -> str:
    """Reformata a transcrição com OpenAI Responses API."""
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    prompt = build_reformat_prompt(style, transcript_text)

    response = openai_client.responses.create(
        model=model_name,
        instructions="Você é um assistente de edição e formatação de texto.",
        input=prompt,
        temperature=temperature,
    )

    reformatted = extract_response_text(response).strip()
    if not reformatted:
        raise RuntimeError("A resposta da OpenAI veio vazia.")
    return reformatted


# =========================================================
# Estado inicial
# =========================================================
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

if "transcript_raw_text" not in st.session_state:
    st.session_state.transcript_raw_text = ""

if "transcript_utterances" not in st.session_state:
    st.session_state.transcript_utterances = []

if "speaker_name_map" not in st.session_state:
    st.session_state.speaker_name_map = {}

if "reformatted_text" not in st.session_state:
    st.session_state.reformatted_text = ""

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = "audio"

if "last_saved_path" not in st.session_state:
    st.session_state.last_saved_path = ""

# Mantém bytes do áudio para preview e para transcrição sem depender do ponteiro do UploadedFile
if "uploaded_audio_bytes" not in st.session_state:
    st.session_state.uploaded_audio_bytes = b""

if "uploaded_audio_mime" not in st.session_state:
    st.session_state.uploaded_audio_mime = "audio/wav"

if "last_uploaded_file_token" not in st.session_state:
    st.session_state.last_uploaded_file_token = ""


# =========================================================
# UI Streamlit
# =========================================================
st.set_page_config(page_title="Transcrição de Áudio com Notetaker", page_icon="🎙️", layout="wide")

st.title("🎙️ Transcrição de Áudio com Notetaker")
st.caption("Faça upload de um arquivo de áudio, transcreva, reformate e salve o resultado.")

# (Mantido o estilo que você preferiu: uploader/config na sidebar)
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Envie um arquivo de áudio",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "mpeg", "webm"],
        help="Arquivos grandes podem demorar mais para upload/transcrição.",
    )

    openai_model = st.text_input("Modelo OpenAI (reformatação)", value=DEFAULT_OPENAI_MODEL)
    style_option = st.selectbox(
        "Modelo de formatação",
        options=["padrão", "sumário executivo", "informal", "formal"],
        index=0,
    )

# =========================================================
# Preview do áudio carregado (sem recorte)
# =========================================================
if uploaded_file is not None:
    current_file_bytes = uploaded_file.getvalue()
    current_token = f"{uploaded_file.name}|{len(current_file_bytes)}"

    # Se o arquivo mudou, atualiza estado
    if st.session_state.last_uploaded_file_token != current_token:
        st.session_state.last_uploaded_file_token = current_token
        st.session_state.uploaded_audio_bytes = current_file_bytes
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.uploaded_audio_mime = guess_audio_mime(uploaded_file.name)

    st.subheader("🎧 Pré-visualização do áudio carregado")
    st.audio(
        st.session_state.uploaded_audio_bytes,
        format=st.session_state.uploaded_audio_mime,
    )

st.divider()

# Botões principais
c1, c2, c3 = st.columns(3)

with c1:
    transcribe_clicked = st.button("📝 Transcrever áudio", use_container_width=True)

with c2:
    reformat_clicked = st.button("✨ Reformatar transcrição", use_container_width=True)

with c3:
    clear_clicked = st.button("🧹 Limpar", use_container_width=True)

if clear_clicked:
    st.session_state.transcript_text = ""
    st.session_state.transcript_raw_text = ""
    st.session_state.transcript_utterances = []
    st.session_state.speaker_name_map = {}
    st.session_state.reformatted_text = ""
    st.session_state.uploaded_filename = "audio"
    st.session_state.last_saved_path = ""
    st.session_state.uploaded_audio_bytes = b""
    st.session_state.uploaded_audio_mime = "audio/wav"
    st.session_state.last_uploaded_file_token = ""
    st.rerun()

# Validação de chaves
if not ASSEMBLYAI_API_KEY:
    st.warning("Defina ASSEMBLYAI_API_KEY no .env para habilitar a transcrição.")
if not OPENAI_API_KEY:
    st.warning("Defina OPENAI_API_KEY no .env para habilitar a reformatação.")

# Transcrição
if transcribe_clicked:
    if uploaded_file is None:
        st.error("Envie um arquivo de áudio antes de transcrever.")
    elif not ASSEMBLYAI_API_KEY:
        st.error("ASSEMBLYAI_API_KEY não configurada.")
    else:
        try:
            st.session_state.uploaded_filename = uploaded_file.name

            # Usa bytes persistidos no estado para evitar problema de ponteiro do UploadedFile
            audio_bytes = st.session_state.uploaded_audio_bytes or uploaded_file.getvalue()

            with st.spinner("Transcrevendo áudio com AssemblyAI..."):
                result = transcribe_audio_with_assemblyai(audio_bytes, uploaded_file.name)

            # Guarda texto bruto + utterances
            st.session_state.transcript_raw_text = result.get("text", "") or ""
            st.session_state.transcript_utterances = result.get("utterances", []) or []

            # Inicializa mapeamento dos oradores detectados
            detected_speakers = get_detected_speakers(st.session_state.transcript_utterances)
            st.session_state.speaker_name_map = {spk: spk for spk in detected_speakers}

            # Atualiza transcrição exibida (speaker-labeled se houver diarização)
            refresh_transcript_text_from_state()

            # Limpa reformatação anterior
            st.session_state.reformatted_text = ""

            if detected_speakers:
                st.success(f"Transcrição concluída com sucesso! Oradores detectados: {', '.join(detected_speakers)}")
            else:
                st.success("Transcrição concluída com sucesso!")

        except Exception as e:
            st.exception(e)

# =========================================================
# Renomeação de oradores
# =========================================================
if st.session_state.transcript_utterances:
    with st.expander("👥 Oradores identificados (renomear)", expanded=True):
        detected_speakers = get_detected_speakers(st.session_state.transcript_utterances)

        if detected_speakers:
            st.caption("Edite os nomes abaixo e clique em **Aplicar nomes** para atualizar a transcrição exibida.")

            for spk in detected_speakers:
                default_value = st.session_state.speaker_name_map.get(spk, spk)
                st.text_input(
                    f"Nome para {spk}",
                    value=default_value,
                    key=f"speaker_alias_{spk}",
                    placeholder="Ex.: Entrevistador, Breno, Cliente...",
                )

            col_apply, col_reset = st.columns(2)

            with col_apply:
                if st.button("Aplicar nomes", use_container_width=True):
                    new_map = {}
                    for spk in detected_speakers:
                        alias = (st.session_state.get(f"speaker_alias_{spk}", spk) or "").strip()
                        new_map[spk] = alias if alias else spk

                    st.session_state.speaker_name_map = new_map
                    refresh_transcript_text_from_state()

                    # A reformatação pode ficar desatualizada se os nomes mudarem
                    if st.session_state.reformatted_text.strip():
                        st.session_state.reformatted_text = ""

                    st.success("Nomes dos oradores atualizados.")
                    st.rerun()

            with col_reset:
                if st.button("Restaurar rótulos originais", use_container_width=True):
                    st.session_state.speaker_name_map = {spk: spk for spk in detected_speakers}
                    for spk in detected_speakers:
                        st.session_state[f"speaker_alias_{spk}"] = spk

                    refresh_transcript_text_from_state()

                    if st.session_state.reformatted_text.strip():
                        st.session_state.reformatted_text = ""

                    st.success("Rótulos originais restaurados.")
                    st.rerun()
        else:
            st.info("Nenhum orador foi identificado neste áudio.")

# Reformatação
if reformat_clicked:
    if not st.session_state.transcript_text.strip():
        st.error("Transcreva um áudio primeiro (ou tenha uma transcrição disponível).")
    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY não configurada.")
    else:
        try:
            with st.spinner(f"Reformatando em estilo '{style_option}' com OpenAI..."):
                reformatted = reformat_transcript_with_openai(
                    transcript_text=st.session_state.transcript_text,
                    style=style_option,
                    model_name=openai_model.strip(),
                    temperature=0.2,
                )

            st.session_state.reformatted_text = reformatted
            st.success("Reformatação concluída com sucesso!")
        except Exception as e:
            st.exception(e)

# =========================================================
# Exibição dos textos (corrigido)
# =========================================================
left_text_col, right_text_col = st.columns(2)

with left_text_col:
    st.subheader("Transcrição original")
    st.text_area(
        "Texto transcrito",
        value=st.session_state.transcript_text,
        height=350,
        label_visibility="collapsed",
    )

with right_text_col:
    st.subheader("Transcrição reformulada")
    st.text_area(
        "Texto reformulado",
        value=st.session_state.reformatted_text,
        height=350,
        label_visibility="collapsed",
    )

st.divider()

# Salvamento local e download
st.subheader("💾 Salvar / Baixar transcrição")
save_target = st.radio(
    "Escolha qual versão salvar",
    options=["Transcrição original", "Transcrição reformulada"],
    horizontal=True,
)

default_base_name = Path(st.session_state.uploaded_filename).stem if st.session_state.uploaded_filename else "audio"
custom_base_name = st.text_input("Nome base do arquivo (.txt)", value=f"{default_base_name}_transcricao")

text_to_save = ""
suffix_to_use = ""

if save_target == "Transcrição original":
    text_to_save = st.session_state.transcript_text or ""
    suffix_to_use = "_original"
else:
    text_to_save = st.session_state.reformatted_text or ""
    suffix_to_use = "_reformatada"

save_col1, save_col2 = st.columns([1, 1])

with save_col1:
    if st.button("Salvar em arquivo .txt local", use_container_width=True):
        if not text_to_save.strip():
            st.error("Não há conteúdo para salvar na opção selecionada.")
        else:
            try:
                saved_path = save_text_locally(text_to_save, custom_base_name, suffix=suffix_to_use)
                st.session_state.last_saved_path = str(saved_path)
                st.success(f"Arquivo salvo localmente em: {saved_path}")
                st.info("Obs.: 'local' significa no computador/servidor onde o Streamlit está rodando.")
            except Exception as e:
                st.exception(e)

with save_col2:
    st.download_button(
        label="Baixar .txt no navegador",
        data=text_to_save.encode("utf-8") if text_to_save else b"",
        file_name=f"{sanitize_filename(custom_base_name)}{suffix_to_use}.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not bool(text_to_save.strip()),
    )

if st.session_state.last_saved_path:
    st.caption(f"Último arquivo salvo localmente: {st.session_state.last_saved_path}")