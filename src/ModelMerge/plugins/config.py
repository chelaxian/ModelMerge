import os
import json

from . import *
from ..utils.scripts import cut_message
from ..utils.prompt import search_key_word_prompt, arxiv_doc_assistant_prompt, arxiv_doc_user_prompt

PLUGINS = {
    "SEARCH" : (os.environ.get('SEARCH', "True") == "False") == False,
    "URL"    : (os.environ.get('URL', "True") == "False") == False,
    "ARXIV"  : (os.environ.get('ARXIV', "False") == "False") == False,
    "CODE"   : (os.environ.get('CODE', "False") == "False") == False,
    "IMAGE"  : (os.environ.get('IMAGE', "False") == "False") == False,
    "DATE"   : (os.environ.get('DATE', "False") == "False") == False,
    # "VERSION": (os.environ.get('VERSION', "False") == "False") == False,
    # "TARVEL" : (os.environ.get('TARVEL', "False") == "False") == False,
    # "FLIGHT" : (os.environ.get('FLIGHT', "False") == "False") == False,
}

async def get_tools_result_async(function_call_name, function_full_response, function_call_max_tokens, engine, robot, api_key, api_url, use_plugins, model, add_message, convo_id, language):
    function_response = ""
    if function_call_name == "get_search_results":
        prompt = json.loads(function_full_response)["prompt"]
        yield "message_search_stage_1"
        llm = robot(api_key=api_key, api_url=api_url.source_api_url, engine=engine, use_plugins=use_plugins)
        keywords = (await llm.ask_async(search_key_word_prompt.format(source=prompt), model=model)).split("\n")
        async for chunk in eval(function_call_name)(prompt, keywords):
            if type(chunk) == str:
                yield chunk
            else:
                function_response = "\n\n".join(chunk)
            # function_response = yield chunk
        # function_response = yield from eval(function_call_name)(prompt, keywords)
        function_call_max_tokens = 32000
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
        if function_response:
            function_response = (
                f"You need to response the following question: {prompt}. Search results is provided inside <Search_results></Search_results> XML tags. Your task is to think about the question step by step and then answer the above question in {language} based on the Search results provided. Please response in {language} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive. Directly response markdown format, without using markdown code blocks. For each sentence quoting search results, a markdown ordered superscript number url link must be used to indicate the source, e.g., [¹](https://www.example.com)"
                "Here is the Search results, inside <Search_results></Search_results> XML tags:"
                "<Search_results>"
                "{}"
                "</Search_results>"
            ).format(function_response)
        else:
            function_response = "无法找到相关信息，停止使用 tools"
        # user_prompt = f"You need to response the following question: {prompt}. Search results is provided inside <Search_results></Search_results> XML tags. Your task is to think about the question step by step and then answer the above question in {config.language} based on the Search results provided. Please response in {config.language} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive. Directly response markdown format, without using markdown code blocks"
        # self.add_to_conversation(user_prompt, "user", convo_id=convo_id)
    if function_call_name == "get_url_content":
        url = json.loads(function_full_response)["url"]
        print("\n\nurl", url)
        # function_response = jina_ai_Web_crawler(url)
        # function_response = Web_crawler(url)
        function_response = compare_and_choose_content(url)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "get_city_tarvel_info":
        city = json.loads(function_full_response)["city"]
        function_response = eval(function_call_name)(city)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
        function_response = (
            f"Tarvel infomation is provided inside <infomation></infomation> XML tags. Your task is to think about the question step by step and then answer the above question in {language} based on the tarvel infomation provided. Please response in {language} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive."
            "Here is the tarvel infomation, inside <infomation></infomation> XML tags:"
            "<infomation>"
            "{}"
            "</infomation>"
        ).format(function_response)
    if function_call_name == "get_Round_trip_flight_price":
        departcity = json.loads(function_full_response)["departcity"]
        arrivalcity = json.loads(function_full_response)["arrivalcity"]
        function_response = eval(function_call_name)(departcity, arrivalcity)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
        function_response = (
            # f"Tarvel infomation is provided inside <infomation></infomation> XML tags. Your task is to think about the question step by step and then answer the above question in {language} based on the tarvel infomation provided. Please response in {language} and adopt a style that is logical, in-depth, and detailed. Note: In order to make the answer appear highly professional, you should be an expert in textual analysis, aiming to make the answer precise and comprehensive."
            "Here is the Round-trip flight price infomation, inside <infomation></infomation> XML tags:"
            "<infomation>"
            "{}"
            "</infomation>"
        ).format(function_response)
    if function_call_name == "generate_image":
        prompt = json.loads(function_full_response)["prompt"]
        function_response = eval(function_call_name)(prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "download_read_arxiv_pdf":
        add_message(arxiv_doc_user_prompt, "user", convo_id=convo_id)
        # add_message(arxiv_doc_assistant_prompt, "assistant", convo_id=convo_id)
        prompt = json.loads(function_full_response)["prompt"]
        function_response = eval(function_call_name)(prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "run_python_script":
        prompt = json.loads(function_full_response)["prompt"]
        function_response = await eval(function_call_name)(prompt)
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "get_date_time_weekday":
        function_response = eval(function_call_name)()
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    if function_call_name == "get_version_info":
        function_response = eval(function_call_name)()
        function_response, text_len = cut_message(function_response, function_call_max_tokens, engine)
    function_response = (
        f"function_response:{function_response}"
    )
    yield function_response
    # return function_response