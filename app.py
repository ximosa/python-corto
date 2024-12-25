import streamlit as st
import os
import json
import logging
import time
import re
from google.cloud import texttospeech
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, concatenate_audioclips
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
import requests
from io import BytesIO
from google.api_core import exceptions

logging.basicConfig(level=logging.INFO)

# Cargar credenciales de GCP desde secrets
credentials = dict(st.secrets.gcp_service_account)
with open("google_credentials.json", "w") as f:
    json.dump(credentials, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

# Configuración de voces
VOCES_DISPONIBLES = {
    'es-ES-Journey-D': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Journey-F': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Journey-O': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-A': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-B': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Neural2-C': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-D': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-E': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Neural2-F': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Polyglot-1': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Standard-A': texttospeech.SsmlVoiceGender.FEMALE,
    'es-ES-Standard-B': texttospeech.SsmlVoiceGender.MALE,
    'es-ES-Standard-C': texttospeech.SsmlVoiceGender.FEMALE
}

# Función de creación de texto
def create_text_image(text, size=(1080, 1920), font_size=48, line_height=60):
    img = Image.new('RGB', size, 'black')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        left, top, right, bottom = draw.textbbox((0, 0), test_line, font=font)
        if right > size[0] - 60:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))

    total_height = len(lines) * line_height
    y = (size[1] - total_height) // 2

    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        x = (size[0] - (right - left)) // 2
        draw.text((x, y), line, font=font, fill="white")
        y += line_height

    return np.array(img)


def split_text_into_segments(text, max_segment_length=450, max_time=15):
    frases = [f.strip() + "." for f in text.split('.') if f.strip()]
    segments = []
    current_segment = ""
    for frase in frases:
        if len(current_segment) + len(frase) <= max_segment_length:
            current_segment += " " + frase
        else:
            segments.append(current_segment.strip())
            current_segment = frase
    if current_segment:
        segments.append(current_segment.strip())

    adjusted_segments = []
    for segment in segments:
        words = segment.split()
        current_segment_time = ""
        for word in words:
            if len(current_segment_time.split(" ")) < max_time:
                current_segment_time += word + " "
            else:
                adjusted_segments.append(current_segment_time.strip())
                current_segment_time = word + " "
        if current_segment_time:
            adjusted_segments.append(current_segment_time.strip())

    return adjusted_segments


# Función de creación de video
def create_simple_video(texto, nombre_salida, voz):
    archivos_temp = []
    clips_audio = []
    clips_finales = []

    try:
        logging.info("Iniciando proceso de creación de video...")
        segments = split_text_into_segments(texto)
        client = texttospeech.TextToSpeechClient()

        tiempo_acumulado = 0

        for i, segment in enumerate(segments):
            logging.info(f"Procesando segmento {i + 1} de {len(segments)}")

            synthesis_input = texttospeech.SynthesisInput(text=segment)
            voice = texttospeech.VoiceSelectionParams(
                language_code="es-ES",
                name=voz,
                ssml_gender=VOCES_DISPONIBLES[voz]
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                try:
                    response = client.synthesize_speech(
                        input=synthesis_input,
                        voice=voice,
                        audio_config=audio_config
                    )
                    break
                except Exception as e:
                    logging.error(f"Error al solicitar audio (intento {retry_count + 1}): {str(e)}")
                    if "429" in str(e):
                        retry_count += 1
                        time.sleep(2 ** retry_count)
                    else:
                        raise

            if retry_count > max_retries:
                raise Exception("Maximos intentos de reintento alcanzado")

            temp_filename = f"temp_audio_{i}.mp3"
            archivos_temp.append(temp_filename)
            with open(temp_filename, "wb") as out:
                out.write(response.audio_content)

            audio_clip = AudioFileClip(temp_filename)
            clips_audio.append(audio_clip)
            duracion = audio_clip.duration

            text_img = create_text_image(segment)
            txt_clip = (ImageClip(text_img)
                        .set_start(tiempo_acumulado)
                        .set_duration(duracion)
                        .set_position('center')
                        .crossfadein(0.25))

            video_segment = txt_clip.set_audio(audio_clip.set_start(tiempo_acumulado))
            clips_finales.append(video_segment)

            tiempo_acumulado += duracion
            time.sleep(0.2)

        video_final = concatenate_videoclips(clips_finales, method="compose")

        if video_final.duration > 60:
            video_final = video_final.subclip(0, 60)

        video_final.write_videofile(
            nombre_salida,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=4
        )

        video_final.close()

        for clip in clips_audio:
            clip.close()

        for clip in clips_finales:
            clip.close()

        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.close(os.open(temp_file, os.O_RDONLY))
                    os.remove(temp_file)
            except:
                pass

        return True, "Video generado exitosamente"

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        for clip in clips_audio:
            try:
                clip.close()
            except:
                pass

        for clip in clips_finales:
            try:
                clip.close()
            except:
                pass

        for temp_file in archivos_temp:
            try:
                if os.path.exists(temp_file):
                    os.close(os.open(temp_file, os.O_RDONLY))
                    os.remove(temp_file)
            except:
                pass

        return False, str(e)


def main():
    st.title("Creador de Shorts Automático")

    uploaded_file = st.file_uploader("Carga un archivo de texto", type="txt")
    voz_seleccionada = st.selectbox("Selecciona la voz", options=list(VOCES_DISPONIBLES.keys()))

    if uploaded_file:
        texto = uploaded_file.read().decode("utf-8")
        nombre_salida = st.text_input("Nombre del Video (sin extensión)", "video_generado")

        if st.button("Generar Video"):
            with st.spinner('Generando video...'):
                nombre_salida_completo = f"{nombre_salida}.mp4"
                success, message = create_simple_video(texto, nombre_salida_completo, voz_seleccionada)
                if success:
                    st.success(message)
                    st.video(nombre_salida_completo)
                    with open(nombre_salida_completo, 'rb') as file:
                        st.download_button(label="Descargar video", data=file, file_name=nombre_salida_completo)

                    st.session_state.video_path = nombre_salida_completo
                else:
                    st.error(f"Error al generar video: {message}")

        if st.session_state.get("video_path"):
            st.markdown(f'<a href="https://www.youtube.com/upload" target="_blank">Subir video a YouTube</a>', unsafe_allow_html=True)


if __name__ == "__main__":
    # Inicializar session state
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    main()
