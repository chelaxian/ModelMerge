#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Полный скрипт для обработки различных типов сообщений в Telegram-боте
Включает функции из scripts_new.py и дополнен недостающей функциональностью из scripts.py
"""

# Импорты
import os
import sys
import json
import base64
import tiktoken
import requests
import urllib.parse
import logging
import asyncio
import time
import imghdr
from io import BytesIO

# Настройка логирования
logger = logging.getLogger(__name__)

# Функции для работы с текстом из scripts.py
def get_encode_text(text, model_name):
    tiktoken.get_encoding("cl100k_base")
    model_name = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model_name)
    encode_text = encoding.encode(text, disallowed_special=())
    return encoding, encode_text

def get_text_token_len(text, model_name):
    encoding, encode_text = get_encode_text(text, model_name)
    return len(encode_text)

def cut_message(message: str, max_tokens: int, model_name: str):
    if type(message) != str:
        message = str(message)
    encoding, encode_text = get_encode_text(message, model_name)
    if len(encode_text) > max_tokens:
        encode_text = encode_text[:max_tokens]
        message = encoding.decode(encode_text)
    encode_text = encoding.encode(message)
    return message, len(encode_text)

# Функции для работы с изображениями из scripts.py
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        file_content = image_file.read()
        file_type = imghdr.what(None, file_content)
        base64_encoded = base64.b64encode(file_content).decode('utf-8')

        if file_type == 'png':
            return f"data:image/png;base64,{base64_encoded}"
        elif file_type in ['jpeg', 'jpg']:
            return f"data:image/jpeg;base64,{base64_encoded}"
        else:
            raise ValueError(f"Неподдерживаемый формат изображения: {file_type}")

def get_doc_from_url(url):
    filename = urllib.parse.unquote(url.split("/")[-1])
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename

def get_encode_image(image_url):
    filename = get_doc_from_url(image_url)
    image_path = os.getcwd() + "/" + filename
    base64_image = encode_image(image_path)
    os.remove(image_path)
    return base64_image

def get_image_message(image_url, message, engine = None):
    if image_url:
        base64_image = get_encode_image(image_url)
        colon_index = base64_image.index(":")
        semicolon_index = base64_image.index(";")
        image_type = base64_image[colon_index + 1:semicolon_index]

        if "gpt-4" in engine \
        or (os.environ.get('claude_api_key', None) is None and "claude-3" in engine) \
        or (os.environ.get('GOOGLE_AI_API_KEY', None) is None and "gemini" in engine) \
        or (os.environ.get('GOOGLE_AI_API_KEY', None) is None and os.environ.get('VERTEX_CLIENT_EMAIL', None) is None and os.environ.get('VERTEX_PRIVATE_KEY', None) is None and os.environ.get("VERTEX_PROJECT_ID", None) is None and "gemini" in engine):
            message.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                }
            )
        if os.environ.get('claude_api_key', None) and "claude-3" in engine:
            message.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_type,
                        "data": base64_image.split(",")[1],
                    }
                }
            )
        if (
            os.environ.get('GOOGLE_AI_API_KEY', None) \
            or (os.environ.get('VERTEX_CLIENT_EMAIL', None) and os.environ.get('VERTEX_PRIVATE_KEY', None) and os.environ.get("VERTEX_PROJECT_ID", None))
        ) \
        and "gemini" in engine:
            message.append(
                {
                    "inlineData": {
                        "mimeType": image_type,
                        "data": base64_image.split(",")[1],
                    }
                }
            )
    return message

# Функция для работы с аудио из scripts.py
def get_audio_message(file_bytes):
    try:
        # Создаем байтовый поток
        audio_stream = BytesIO(file_bytes)

        # Используем поток для транскрипции
        import config
        transcript = config.whisperBot.generate(audio_stream)

        return transcript

    except Exception as e:
        return f"Ошибка при обработке аудиофайла: {str(e)}"

# Функция для работы с документами из scripts.py
def Document_extract(docurl, docpath=None, engine = None):
    filename = docpath
    text = None
    prompt = None
    if docpath and docurl and "paper.pdf" != docpath:
        filename = get_doc_from_url(docurl)
        docpath = os.getcwd() + "/" + filename
    if filename and filename[-3:] == "pdf":
        from pdfminer.high_level import extract_text
        text = extract_text(docpath)
    if filename and (filename[-3:] == "txt" or filename[-3:] == ".md" or filename[-3:] == ".py" or filename[-3:] == "yml"):
        with open(docpath, 'r') as f:
            text = f.read()
    if text:
        prompt = (
            "Here is the document, inside <document></document> XML tags:"
            "<document>"
            "{}"
            "</document>"
        ).format(text)
    if filename and filename[-3:] == "jpg" or filename[-3:] == "png" or filename[-4:] == "jpeg":
        prompt = get_image_message(docurl, [], engine)
    if filename and filename[-3:] == "wav" or filename[-3:] == "mp3":
        with open(docpath, "rb") as file:
            file_bytes = file.read()
        prompt = get_audio_message(file_bytes)
        prompt = (
            "Here is the text content after voice-to-text conversion, inside <voice-to-text></voice-to-text> XML tags:"
            "<voice-to-text>"
            "{}"
            "</voice-to-text>"
        ).format(prompt)
    if os.path.exists(docpath):
        os.remove(docpath)
    return prompt

# Функции для работы с JSON из scripts.py
def split_json_strings(input_string):
    # Инициализируем список результатов и текущую JSON строку
    json_strings = []
    current_json = ""
    brace_count = 0

    # Обходим входную строку посимвольно
    for char in input_string:
        current_json += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

            # Если фигурные скобки сбалансированы, найдена полная JSON строка
            if brace_count == 0:
                # Пытаемся разобрать текущую JSON строку
                try:
                    json.loads(current_json)
                    json_strings.append(current_json)
                    current_json = ""
                except json.JSONDecodeError:
                    # Если разбор не удался, продолжаем добавлять символы
                    pass
    if json_strings == []:
        json_strings.append(input_string)
    return json_strings

def check_json(json_data):
    while True:
        try:
            result = split_json_strings(json_data)
            if len(result) > 0:
                json_data = result[0]
            json.loads(json_data)
            break
        except json.decoder.JSONDecodeError as e:
            logger.error(f"JSON error: {e}")
            logger.error(f"JSON body: {repr(json_data)}")
            if "Invalid control character" in str(e):
                json_data = json_data.replace("\n", "\\n")
            elif "Unterminated string starting" in str(e):
                json_data += '"}'
            elif "Expecting ',' delimiter" in str(e):
                json_data =  {"prompt": json_data}
            elif "Expecting ':' delimiter" in str(e):
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
            elif "Expecting value: line 1 column 1" in str(e):
                if json_data.startswith("prompt: "):
                    json_data = json_data.replace("prompt: ", "")
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
            else:
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
    return json_data

# Функции для работы с китайским текстом из scripts.py
def is_surrounded_by_chinese(text, index):
    left_char = text[index - 1]
    if 0 < index < len(text) - 1:
        right_char = text[index + 1]
        return '\u4e00' <= left_char <= '\u9fff' or '\u4e00' <= right_char <= '\u9fff'
    if index == len(text) - 1:
        return '\u4e00' <= left_char <= '\u9fff'
    return False

def replace_char(string, index, new_char):
    return string[:index] + new_char + string[index+1:]

def claude_replace(text):
    Punctuation_mapping = {",": "，", ":": "：", "!": "！", "?": "？", ";": "；"}
    key_list = list(Punctuation_mapping.keys())
    for i in range(len(text)):
        if is_surrounded_by_chinese(text, i) and (text[i] in key_list):
            text = replace_char(text, i, Punctuation_mapping[text[i]])
    return text

# Асинхронная утилита из scripts.py
def async_generator_to_sync(async_gen):
    """
    Функция для преобразования асинхронного генератора в синхронный
    
    Args:
        async_gen: Асинхронная функция-генератор
        
    Yields:
        Каждое значение, созданное асинхронным генератором
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async def collect_chunks():
            chunks = []
            async for chunk in async_gen:
                chunks.append(chunk)
            return chunks

        chunks = loop.run_until_complete(collect_chunks())
        for chunk in chunks:
            yield chunk

    except Exception as e:
        logger.error(f"Error during async execution: {e}")
        raise
    finally:
        try:
            # Очищаем все ожидающие задачи
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Функции из scripts_new.py
def CutNICK(update_text, update_message):
    import config
    botNick = config.NICK.lower() if config.NICK else None
    botNicKLength = len(botNick) if botNick else 0

    update_chat = update_message.chat
    update_reply_to_message = update_message.reply_to_message
    if botNick is None:
        return update_text
    else:
        if update_text[:botNicKLength].lower() == botNick:
            return update_text[botNicKLength:].strip()
        else:
            if update_chat.type == 'private' or (botNick and update_reply_to_message and update_reply_to_message.text and update_reply_to_message.from_user.is_bot and update_reply_to_message.sender_chat == None):
                return update_text
            else:
                return None

time_out = 600
async def get_file_url(file, context):
    file_id = file.file_id
    new_file = await context.bot.get_file(file_id, read_timeout=time_out, write_timeout=time_out, connect_timeout=time_out, pool_timeout=time_out)
    file_url = new_file.file_path
    return file_url

async def get_voice(file_id: str, context) -> str:
    file_unique_id = file_id
    filename_mp3 = f'{file_unique_id}.mp3'

    try:
        # Пробуем загрузить файл до 3 раз, чтобы избежать пустых файлов
        file_bytes = None
        for attempt in range(3):
            try:
                logger.info(f"Downloading voice file (attempt {attempt+1}/3)")
                file = await context.bot.get_file(file_id)
                file_bytes = await file.download_as_bytearray()
                
                # Проверяем размер файла
                if len(file_bytes) > 100:  # Минимальный размер валидного аудиофайла
                    logger.info(f"Successfully downloaded voice file, size: {len(file_bytes)} bytes")
                    break
                else:
                    logger.warning(f"Downloaded empty or too small voice file on attempt {attempt+1}/3, size: {len(file_bytes)} bytes")
                    # Добавляем небольшую задержку перед повторной попыткой
                    await asyncio.sleep(1)
                    file_bytes = None
            except Exception as e:
                logger.error(f"Error downloading voice file (attempt {attempt+1}/3): {str(e)}")
                await asyncio.sleep(1)
        
        # Если после всех попыток файл всё еще пустой или слишком маленький
        if file_bytes is None or len(file_bytes) <= 100:
            logger.error(f"Failed to download valid voice file after 3 attempts")
            return "Не удалось загрузить голосовое сообщение. Пожалуйста, попробуйте ещё раз."

        # Создаем объект байтового потока для передачи файла
        audio_stream = BytesIO(file_bytes)
        audio_stream.name = "audio.mp3"  # Задаем имя файла для multipart/form-data

        # Получаем API ключ из конфигурации
        import config
        api_key = config.API

        # Если локальный whisperBot доступен, попробуем его использовать сначала
        if hasattr(config, 'whisperBot') and config.whisperBot:
            try:
                logger.info("Using local whisperBot for transcription")
                # Создаем новый BytesIO, чтобы сбросить позицию чтения на начало
                local_audio_stream = BytesIO(file_bytes)
                transcript = config.whisperBot.generate(local_audio_stream)
                if transcript and not transcript.startswith("error:"):
                    return transcript
                logger.warning(f"Local whisperBot failed: {transcript}")
            except Exception as local_e:
                logger.error(f"Local whisperBot error: {str(local_e)}")
                # Продолжаем с внешним API
        
        # Используем 1min-relay API для транскрипции
        logger.info("Using 1min-relay API for transcription")
        # Устанавливаем 1min-relay API URL
        # По умолчанию предполагаем, что он запущен на локальном хосте на порту 5001
        api_url = "http://localhost:5001/v1/audio/transcriptions"
        
        # До 3 попыток отправки запроса к API
        for api_attempt in range(3):
            try:
                # Сбрасываем позицию чтения на начало файла перед каждой попыткой
                audio_stream.seek(0)
                
                # Подготавливаем файл для отправки
                files = {
                    'file': ('audio.mp3', audio_stream, 'audio/mpeg')
                }
                
                # Подготавливаем данные формы
                data = {
                    'model': 'whisper-1',
                }
                
                # Подготавливаем заголовки
                headers = {
                    'Authorization': f'Bearer {api_key}',
                }
                
                # Отправляем запрос
                logger.info(f"Sending transcription request to {api_url} (attempt {api_attempt+1}/3)")
                response = requests.post(api_url, files=files, data=data, headers=headers, timeout=30)
                
                # Проверяем ответ
                if response.status_code == 200:
                    response_data = response.json()
                    logger.info(f"Received transcription response: {json.dumps(response_data)[:200]}...")
                    
                    # Сохраняем ответ для отладки
                    try:
                        with open('last_transcription_response.json', 'w', encoding='utf-8') as f:
                            json.dump(response_data, f, ensure_ascii=False, indent=2)
                    except Exception as save_e:
                        logger.error(f"Error saving response to file: {str(save_e)}")
                    
                    # Извлекаем текст из ответа в формате OpenAI API
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        transcript = response_data['choices'][0]['message']['content']
                        logger.info(f"Extracted full transcript from choices: '{transcript}'")
                        # Проверяем, есть ли в транскрипции полезная информация
                        if not transcript or transcript.strip() in ["", ".", ",", "?", "!"]:
                            logger.warning(f"Empty or minimal transcript detected: '{transcript}'")
                            if api_attempt < 2:
                                continue
                            return "Не удалось распознать текст в голосовом сообщении. Пожалуйста, попробуйте говорить чётче."
                        return transcript
                    # Вторая возможная структура ответа
                    elif 'aiRecord' in response_data and 'aiRecordDetail' in response_data['aiRecord']:
                        details = response_data['aiRecord']['aiRecordDetail']
                        if 'resultObject' in details:
                            result_obj = details['resultObject']
                            if isinstance(result_obj, list) and result_obj:
                                transcript = "".join(result_obj)
                                logger.info(f"Extracted full transcript from aiRecord list: '{transcript}'")
                                # Проверка на пустую транскрипцию
                                if not transcript or transcript.strip() in ["", ".", ",", "?", "!"]:
                                    logger.warning(f"Empty or minimal transcript from aiRecord detected: '{transcript}'")
                                    if api_attempt < 2:
                                        continue
                                    return "Не удалось распознать текст в голосовом сообщении. Пожалуйста, попробуйте говорить чётче."
                                return transcript
                            elif isinstance(result_obj, str):
                                transcript = result_obj
                                logger.info(f"Extracted full transcript from aiRecord string: '{transcript}'")
                                # Проверка на пустую транскрипцию
                                if not transcript or transcript.strip() in ["", ".", ",", "?", "!"]:
                                    logger.warning(f"Empty or minimal transcript from aiRecord string detected: '{transcript}'")
                                    if api_attempt < 2:
                                        continue
                                    return "Не удалось распознать текст в голосовом сообщении. Пожалуйста, попробуйте говорить чётче."
                                return transcript
                    else:
                        logger.warning(f"Invalid response format (attempt {api_attempt+1}/3): {response_data}")
                        if api_attempt == 2:  # Последняя попытка
                            return "Не удалось распознать голосовое сообщение. Попробуйте еще раз."
                else:
                    logger.error(f"Error from API (attempt {api_attempt+1}/3): {response.status_code} - {response.text}")
                    if api_attempt < 2:  # Не последняя попытка
                        await asyncio.sleep(2)  # Ждем перед повторной попыткой
                    else:
                        return f"Ошибка распознавания голосового сообщения: HTTP {response.status_code}"
            
            except Exception as api_e:
                logger.error(f"API request error (attempt {api_attempt+1}/3): {str(api_e)}")
                if api_attempt < 2:  # Не последняя попытка
                    await asyncio.sleep(2)  # Ждем перед повторной попыткой
                else:
                    return f"Ошибка при отправке запроса на распознавание: {str(api_e)}"

        # Если мы дошли до этой точки, значит все попытки не удались
        return "Не удалось распознать голосовое сообщение после нескольких попыток. Пожалуйста, попробуйте позже."

    except Exception as e:
        logger.error(f"Voice transcription error: {str(e)}")
        return f"Временно невозможно использовать голосовую функцию: {str(e)}"
    finally:
        if os.path.exists(filename_mp3):
            os.remove(filename_mp3)

# Функции для обработки сообщений
async def GetMesage(update_message, context, voice=True):
    image_url = None
    file_url = None
    reply_to_message_text = None
    message = None
    rawtext = None
    voice_text = None
    reply_to_message_file_content = None

    chatid = str(update_message.chat_id)
    if update_message.is_topic_message:
        message_thread_id = update_message.message_thread_id
    else:
        message_thread_id = None
    if message_thread_id:
        convo_id = str(chatid) + "_" + str(message_thread_id)
    else:
        convo_id = str(chatid)

    messageid = update_message.message_id

    # Добавляем логирование для отладки голосовых сообщений
    if update_message.voice:
        logger.info(f"Voice message detected: ID={update_message.voice.file_id}, Duration={update_message.voice.duration}s")
    
    if update_message.text:
        message = rawtext = CutNICK(update_message.text, update_message)

    if update_message.reply_to_message:
        reply_to_message_text = update_message.reply_to_message.text
        reply_to_message_file = update_message.reply_to_message.document

        if update_message.reply_to_message.photo:
            photo = update_message.reply_to_message.photo[-1]
            image_url = await get_file_url(photo, context)

        if reply_to_message_file:
            reply_to_message_file_url = await get_file_url(reply_to_message_file, context)
            reply_to_message_file_content = Document_extract(reply_to_message_file_url, reply_to_message_file_url, None)

    if update_message.photo:
        photo = update_message.photo[-1]

        image_url = await get_file_url(photo, context)

        if update_message.caption:
            message = rawtext = CutNICK(update_message.caption, update_message)

    if voice and update_message.voice:
        voice_id = update_message.voice.file_id
        voice_text = await get_voice(voice_id, context)

        if update_message.caption:
            message = rawtext = CutNICK(update_message.caption, update_message)

    if update_message.document:
        file = update_message.document

        file_url = await get_file_url(file, context)

        if image_url == None and file_url and (file_url[-3:] == "jpg" or file_url[-3:] == "png" or file_url[-4:] == "jpeg"):
            image_url = file_url

        if update_message.caption:
            message = rawtext = CutNICK(update_message.caption, update_message)

    if update_message.audio:
        file = update_message.audio

        file_url = await get_file_url(file, context)

        if image_url == None and file_url and (file_url[-3:] == "jpg" or file_url[-3:] == "png" or file_url[-4:] == "jpeg"):
            image_url = file_url

        if update_message.caption:
            message = rawtext = CutNICK(update_message.caption, update_message)

    return message, rawtext, image_url, chatid, messageid, reply_to_message_text, message_thread_id, convo_id, file_url, reply_to_message_file_content, voice_text

async def GetMesageInfo(update, context, voice=True):
    if update.edited_message:
        message, rawtext, image_url, chatid, messageid, reply_to_message_text, message_thread_id, convo_id, file_url, reply_to_message_file_content, voice_text = await GetMesage(update.edited_message, context, voice)
        update_message = update.edited_message
    elif update.callback_query:
        message, rawtext, image_url, chatid, messageid, reply_to_message_text, message_thread_id, convo_id, file_url, reply_to_message_file_content, voice_text = await GetMesage(update.callback_query.message, context, voice)
        update_message = update.callback_query.message
    elif update.message:
        message, rawtext, image_url, chatid, messageid, reply_to_message_text, message_thread_id, convo_id, file_url, reply_to_message_file_content, voice_text = await GetMesage(update.message, context, voice)
        update_message = update.message
    else:
        return None, None, None, None, None, None, None, None, None, None, None, None
    return message, rawtext, image_url, chatid, messageid, reply_to_message_text, update_message, message_thread_id, convo_id, file_url, reply_to_message_file_content, voice_text

# Переопределение safe_get из scripts_new.py, сохраняем формат оригинала
def safe_get(data, *keys):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return None
    return data

# Функция для определения эмодзи
def is_emoji(character):
    if len(character) != 1:
        return False

    code_point = ord(character)

    # Определяем диапазоны Юникода для эмодзи
    emoji_ranges = [
        (0x1F300, 0x1F5FF),  # Различные символы и пиктограммы
        (0x1F600, 0x1F64F),  # Эмодзи
        (0x1F680, 0x1F6FF),  # Транспорт и карты
        (0x2600, 0x26FF),    # Различные символы
        (0x2700, 0x27BF),    # Декоративные символы
        (0x1F900, 0x1F9FF)   # Дополнительные символы и пиктограммы
    ]

    # Проверяем, входит ли код символа в любой из диапазонов эмодзи
    return any(start <= code_point <= end for start, end in emoji_ranges)

if __name__ == "__main__":
    os.system("clear" if os.name == "posix" else "cls") 
