import os
import re
import json
import copy
import httpx
import requests
import tiktoken
from typing import Set
from typing import Union
from pathlib import Path


from .base import BaseLLM
from ..tools import function_call_list
from ..plugins import PLUGINS, get_tools_result_async
from ..utils.scripts import check_json, safe_get, async_generator_to_sync
from ..core.request import prepare_request_payload

def get_filtered_keys_from_object(obj: object, *keys: str) -> Set[str]:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return set(class_keys)

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(
            f"Invalid keys: {invalid_keys}",
        )
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}

class chatgpt(BaseLLM):
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-4o",
        api_url: str = (os.environ.get("API_URL") or "https://api.openai.com/v1/chat/completions"),
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        proxy: str = None,
        timeout: float = 600,
        max_tokens: int = None,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        truncate_limit: int = None,
        use_plugins: bool = True,
        print_log: bool = False,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        super().__init__(api_key, engine, api_url, system_prompt, proxy, timeout, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, reply_count, truncate_limit, use_plugins=use_plugins, print_log=print_log)
        self.conversation: dict[str, list[dict]] = {
            "default": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
            ],
        }
        self.function_calls_counter = {}
        self.function_call_max_loop = 3

        if self.get_token_count("default") > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(
        self,
        message: Union[str, list],
        role: str,
        convo_id: str = "default",
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        pass_history: int = 9999,
        function_call_id: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        # print("role", role, "function_name", function_name, "message", message)
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        if function_name == "" and message and message != None:
            self.conversation[convo_id].append({"role": role, "content": message})
        elif function_name != "" and message and message != None:
            self.conversation[convo_id].append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": function_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments,
                        },
                    }
                ],
                })
            self.conversation[convo_id].append({"role": role, "tool_call_id": function_call_id, "content": message})
        else:
            print('\033[31m')
            print("error: add_to_conversation message is None or empty")
            print("role", role, "function_name", function_name, "message", message)
            print('\033[0m')

        conversation_len = len(self.conversation[convo_id]) - 1
        message_index = 0
        # print(json.dumps(self.conversation[convo_id], indent=4, ensure_ascii=False))
        while message_index < conversation_len:
            if self.conversation[convo_id][message_index]["role"] == self.conversation[convo_id][message_index + 1]["role"]:
                if self.conversation[convo_id][message_index].get("content") and self.conversation[convo_id][message_index + 1].get("content"):
                    if type(self.conversation[convo_id][message_index + 1]["content"]) == str \
                    and type(self.conversation[convo_id][message_index]["content"]) == list:
                        self.conversation[convo_id][message_index + 1]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index + 1]["content"]}]
                    if type(self.conversation[convo_id][message_index]["content"]) == str \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == list:
                        self.conversation[convo_id][message_index]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index]["content"]}]
                    self.conversation[convo_id][message_index]["content"] += self.conversation[convo_id][message_index + 1]["content"]
                self.conversation[convo_id].pop(message_index + 1)
                conversation_len = conversation_len - 1
            else:
                message_index = message_index + 1

        history_len = len(self.conversation[convo_id])

        history = pass_history
        if pass_history < 2:
            history = 2
        while history_len > history:
            mess_body = self.conversation[convo_id].pop(1)
            history_len = history_len - 1
            if mess_body.get("role") == "user":
                assistant_body = self.conversation[convo_id].pop(1)
                history_len = history_len - 1
                if assistant_body.get("tool_calls"):
                    self.conversation[convo_id].pop(1)
                    history_len = history_len - 1

        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count(convo_id) > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                mess = self.conversation[convo_id].pop(1)
                print("Truncate message:", mess)
            else:
                break

    # def truncate_conversation(self, convo_id: str = "default") -> None:
    #     """
    #     Truncate the conversation
    #     """
    #     while True:

    #         try:
    #             message_token = {
    #                 "total": self.get_token_count(convo_id),
    #             }
    #         except:
    #             print('\033[31m')
    #             print("error: get_message_token")
    #             print('\033[0m')
    #             message_token = {
    #                 "total": 0,
    #             }
    #         if self.print_log:
    #             print("message_token", message_token, "truncate_limit", self.truncate_limit)
    #         if (
    #             message_token["total"] > self.truncate_limit
    #             and len(self.conversation[convo_id]) > 1
    #         ):
    #             # Don't remove the first message
    #             mess = self.conversation[convo_id].pop(1)
    #             print("Truncate message:", mess)
    #         else:
    #             break
    #     return json_post, message_token

    def extract_values(self, obj):
        if isinstance(obj, dict):
            for value in obj.values():
                yield from self.extract_values(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from self.extract_values(item)
        else:
            yield obj

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        # encoding = tiktoken.get_encoding("cl100k_base")

        # num_tokens = 0
        # for message in self.conversation[convo_id]:
        #     # every message follows <im_start>{role/name}\n{content}<im_end>\n
        #     num_tokens += 5
        #     for key, value in message.items():
        #         values = list(self.extract_values(value))
        #         if "image_url" in values:
        #             continue
        #         if values:
        #             for value in values:
        #                 # print("value", value)
        #                 try:
        #                     num_tokens += len(encoding.encode(value))
        #                 except:
        #                     print('\033[31m')
        #                     print("error value:", value)
        #                     print('\033[0m')
        #                     num_tokens += 0
        #         if key == "name":  # if there's a name, the role is omitted
        #             num_tokens += 5  # role is always required and always 1 token
        # num_tokens += 5  # every reply is primed with <im_start>assistant
        num_tokens = self.tokens_usage[convo_id]
        return num_tokens

    async def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        **kwargs,
    ):
        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt}

        # 构造 provider 信息
        provider = {
            "provider": "openai",
            "base_url": kwargs.get('api_url', self.api_url.chat_url),
            "api": kwargs.get('api_key', self.api_key),
            "model": [model or self.engine],
            "tools": True if self.use_plugins else False,
            "image": True
        }

        # 构造请求数据
        request_data = {
            "model": model or self.engine,
            "messages": copy.deepcopy(self.conversation[convo_id]) if pass_history else [
                {"role": "system","content": self.system_prompt},
                {"role": role, "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
            "temperature": kwargs.get("temperature", self.temperature)
        }

        # 添加工具相关信息
        plugins = kwargs.get("plugins", PLUGINS)
        if not (all(value == False for value in plugins.values()) or self.use_plugins == False):
            tools = []
            # tools.append(copy.deepcopy(function_call_list["base"]))
            for item in plugins.keys():
                try:
                    if plugins[item]:
                        tools.append({"type": "function", "function": function_call_list[item]})
                except:
                    pass
            if tools:
                request_data["tools"] = tools
                request_data["tool_choice"] = "auto"

        # print("request_data", json.dumps(request_data, indent=4, ensure_ascii=False))

        # 调用核心模块的 prepare_request_payload 函数
        url, headers, json_post_body = await prepare_request_payload(provider, request_data)

        return json_post_body

    def ask_stream(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        function_call_id: str = "",
        language: str = "English",
        system_prompt: str = None,
        **kwargs,
    ):
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, function_call_id=function_call_id, pass_history=pass_history)

        # 创建异步包装函数获取 post body
        json_post = None

        async def get_post_body_async():
            nonlocal json_post
            return await self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)

        # 使用封装后的函数
        json_post = next(async_generator_to_sync(get_post_body_async()))

        self.truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])
        if self.print_log:
            print("api_url", kwargs.get('api_url', self.api_url.chat_url))
            print("api_key", kwargs.get('api_key', self.api_key))

        for _ in range(3):
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
                print(json.dumps(replaced_text, indent=4, ensure_ascii=False))
            response = None
            try:
                response = self.session.post(
                    kwargs.get('api_url', self.api_url.chat_url),
                    headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
                    json=json_post,
                    timeout=kwargs.get("timeout", self.timeout),
                    stream=True,
                )
            except ConnectionError:
                print("连接错误，请检查服务器状态或网络连接。")
                return
            except requests.exceptions.ReadTimeout:
                print("请求超时，请检查网络连接或增加超时时间。{e}")
                return
            except Exception as e:
                print(f"发生了未预料的错误：{e}")
                if "Invalid URL" in str(e):
                    e = "You have entered an invalid API URL, please use the correct URL and use the `/start` command to set the API URL again. Specific error is as follows:\n\n" + str(e)
                    raise Exception(f"{e}")
            # print("response.text", response.text)
            # print("response.status_code", response.status_code, response.status_code == 503, response != None and response.status_code == 503, response.text[:400])
            if response != None and (response.status_code == 400 or response.status_code == 422):
                print("response.text", response.text)
                if "function calling" in response.text:
                    if "tools" in json_post:
                        del json_post["tools"]
                    if "tool_choice" in json_post:
                        del json_post["tool_choice"]
                elif "invalid_request_error" in response.text:
                    for index, mess in enumerate(json_post["messages"]):
                        if type(mess["content"]) == list and "text" in mess["content"][0]:
                            json_post["messages"][index] = {
                                "role": mess["role"],
                                "content": mess["content"][0]["text"]
                            }
                elif "'function' is not an allowed role" in response.text:
                    if json_post["messages"][-1]["role"] == "tool":
                        mess = json_post["messages"][-1]
                        json_post["messages"][-1] = {
                            "role": "assistant",
                            "name": mess["name"],
                            "content": mess["content"]
                        }
                else:
                    if "tools" in json_post:
                        del json_post["tools"]
                    if "tool_choice" in json_post:
                        del json_post["tool_choice"]
                continue
            if response != None and response.status_code == 503:
                # print("response.text", response.text)
                if "Sorry, server is busy" in response.text:
                    for index, mess in enumerate(json_post["messages"]):
                        if type(mess["content"]) == list and "text" in mess["content"][0]:
                            json_post["messages"][index] = {
                                "role": mess["role"],
                                "content": mess["content"][0]["text"]
                            }
                continue
            if response != None and response.status_code == 200 and "is not possible because the prompts occupy" in response.text:
                max_tokens = re.findall(r"only\s(\d+)\stokens", response.text)
                # print("max_tokens", max_tokens)
                if max_tokens:
                    json_post["max_tokens"] = int(max_tokens[0])
                    continue
            if response != None and response.status_code == 200:
                if response.text == "":
                    for index, mess in enumerate(json_post["messages"]):
                        if type(mess["content"]) == list and "text" in mess["content"][0]:
                            json_post["messages"][index] = {
                                "role": mess["role"],
                                "content": mess["content"][0]["text"]
                            }
                    continue
                else:
                    break
        # print("response.status_code", response.status_code, response.text)
        if response != None and response.status_code != 200:
            raise Exception(f"{response.status_code} {response.reason} {response.text[:400]}")
        if response is None:
            raise Exception(f"response is None, please check the connection or network.")

        response_role: str = None
        full_response: str = ""
        function_full_response: str = ""
        function_call_name: str = ""
        need_function_call: bool = False
        total_tokens = 0
        for line in response.iter_lines():
            line = line.decode("utf-8")
            if not line or line.startswith(':'):
                continue
            # print(line)
            if line.startswith('data:'):
                line = line.lstrip("data: ")
                if line == "[DONE]":
                    break
            else:
                line = json.loads(line)
                if safe_get(line, "choices", 0, "message", "content"):
                    full_response = line["choices"][0]["message"]["content"]
                    yield full_response
                else:
                    yield str(line)
                break
            resp: dict = json.loads(line)
            if "error" in resp:
                raise Exception(f"{resp}")
            total_tokens = total_tokens or safe_get(resp, "usage", "total_tokens", default=0)
            delta = safe_get(resp, "choices", 0, "delta")
            if not delta:
                continue
            response_role = response_role or safe_get(delta, "role")
            if safe_get(delta, "content"):
                need_function_call = False
                content = delta["content"]
                full_response += content
                yield content
            if safe_get(delta, "tool_calls"):
                need_function_call = True
                function_call_name = function_call_name or safe_get(delta, "tool_calls", 0, "function", "name")
                function_full_response += safe_get(delta, "tool_calls", 0, "function", "arguments", default="")
                function_call_id = function_call_id or safe_get(delta, "tool_calls", 0, "id")
        if self.print_log:
            print("\n\rtotal_tokens", total_tokens)
        if response_role == None:
            response_role = "assistant"
        if need_function_call:
            function_full_response = check_json(function_full_response)
            print("function_full_response", function_full_response)
            function_response = ""
            # print(self.function_calls_counter)
            if not self.function_calls_counter.get(function_call_name):
                self.function_calls_counter[function_call_name] = 1
            else:
                self.function_calls_counter[function_call_name] += 1
            if self.function_calls_counter[function_call_name] <= self.function_call_max_loop and function_full_response != "{}":
                function_call_max_tokens = self.truncate_limit - 1000
                if function_call_max_tokens <= 0:
                    function_call_max_tokens = int(self.truncate_limit / 2)
                print("\033[32m function_call", function_call_name, "max token:", function_call_max_tokens, "\033[0m")

                # # function_response = yield from get_tools_result(function_call_name, function_full_response, function_call_max_tokens, model or self.engine, chatgpt, kwargs.get('api_key', self.api_key), self.api_url, use_plugins=False, model=model or self.engine, add_message=self.add_to_conversation, convo_id=convo_id, language=language)

                async def run_async():
                    nonlocal function_response
                    async for chunk in get_tools_result_async(
                        function_call_name, function_full_response, function_call_max_tokens,
                        model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
                        self.api_url, use_plugins=False, model=model or self.engine,
                        add_message=self.add_to_conversation, convo_id=convo_id, language=language
                    ):
                        if "function_response:" in chunk:
                            function_response = chunk.replace("function_response:", "")
                        else:
                            yield chunk

                # 使用封装后的函数
                for chunk in async_generator_to_sync(run_async()):
                    yield chunk

            else:
                function_response = "无法找到相关信息，停止使用 tools"
            response_role = "tool"
            # print(self.conversation[convo_id][-1])
            # if self.conversation[convo_id][-1]["role"] == "tool" and self.conversation[convo_id][-1]["name"] == "get_search_results":
            #     mess = self.conversation[convo_id].pop(-1)
                # print("Truncate message:", mess)
            yield from self.ask_stream(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens, model=model or self.engine, function_arguments=function_full_response, function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key), plugins=kwargs.get("plugins", PLUGINS), system_prompt=system_prompt)
        else:
            # if self.conversation[convo_id][-1]["role"] == "tool" and self.conversation[convo_id][-1]["name"] == "get_search_results":
            #     mess = self.conversation[convo_id].pop(-1)
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}
            if pass_history <= 2 and len(self.conversation[convo_id]) >= 2 \
            and (
                "You are a translation engine" in self.conversation[convo_id][-2]["content"] \
                or "You are a translation engine" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="") \
                or "你是一位精通简体中文的专业翻译" in self.conversation[convo_id][-2]["content"] \
                or "你是一位精通简体中文的专业翻译" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="")
            ):
                self.conversation[convo_id].pop(-1)
                self.conversation[convo_id].pop(-1)

    async def ask_stream_async(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        function_call_id: str = "",
        language: str = "English",
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        **kwargs,
    ):
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, pass_history=pass_history, function_call_id=function_call_id)
        json_post = await self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)
        self.truncate_conversation(convo_id=convo_id)
        # print(self.conversation[convo_id])
        if self.print_log:
            print("api_url", kwargs.get('api_url', self.api_url.chat_url))
            print("api_key", kwargs.get('api_key', self.api_key))

        response_role: str = None
        full_response: str = ""
        function_full_response: str = ""
        function_call_name: str = ""
        need_function_call: bool = False
        total_tokens = 0

        for _ in range(3):
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
                print(json.dumps(replaced_text, indent=4, ensure_ascii=False))
            try:
                async with self.aclient.stream(
                    "post",
                    self.api_url.chat_url,
                    headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
                    json=json_post,
                    timeout=kwargs.get("timeout", self.timeout),
                ) as response:
                    # print("response.text", response.text)
                    if response != None:
                        # await response.aread()
                        # print("response.status_code", response.status_code, response.status_code == 200, response != None and response.status_code == 200, response.text == "", response.text[:400])
                        if response.status_code == 400 or response.status_code == 422:
                            await response.aread()
                            print("response.text", response.text)
                            if "Content did not pass the moral check" in response.text:
                                raise Exception(f"{response.status_code} {response.reason_phrase} {response.text[:400]}")
                            if "function calling" in response.text:
                                if "tools" in json_post:
                                    del json_post["tools"]
                                if "tool_choice" in json_post:
                                    del json_post["tool_choice"]
                            elif "invalid_request_error" in response.text:
                                for index, mess in enumerate(json_post["messages"]):
                                    if safe_get(mess, "content", 0) == "text":
                                        json_post["messages"][index] = {
                                            "role": mess["role"],
                                            "content": mess["content"][0]["text"]
                                        }
                            elif "'function' is not an allowed role" in response.text:
                                if json_post["messages"][-1]["role"] == "tool":
                                    mess = json_post["messages"][-1]
                                    json_post["messages"][-1] = {
                                        "role": "assistant",
                                        "name": mess["name"],
                                        "content": mess["content"]
                                    }
                            else:
                                if "tools" in json_post:
                                    del json_post["tools"]
                                if "tool_choice" in json_post:
                                    del json_post["tool_choice"]
                            continue
                        if response.status_code == 503:
                            await response.aread()
                            # print("response.text", response.text)
                            if "Sorry, server is busy" in response.text:
                                for index, mess in enumerate(json_post["messages"]):
                                    if type(mess["content"]) == list and "text" in mess["content"][0]:
                                        json_post["messages"][index] = {
                                            "role": mess["role"],
                                            "content": mess["content"][0]["text"]
                                        }
                            continue
                        # if response.status_code == 200 and "is not possible because the prompts occupy" in response.text:
                        #     max_tokens = re.findall(r"only\s(\d+)\stokens", response.text)
                        #     # print("max_tokens", max_tokens)
                        #     if max_tokens:
                        #         json_post["max_tokens"] = int(max_tokens[0])
                        #         continue
                        # if response.status_code == 200 and response.text == "":
                        #     for index, mess in enumerate(json_post["messages"]):
                        #         if type(mess["content"]) == list and "text" in mess["content"][0]:
                        #             json_post["messages"][index] = {
                        #                 "role": mess["role"],
                        #                 "content": mess["content"][0]["text"]
                        #             }
                        #     continue
                        if response.status_code != 200:
                            await response.aread()
                            raise Exception(f"{response.status_code} {response.reason_phrase} {response.text[:400]}")
                    else:
                        raise Exception(f"response is None, please check the connection or network.")

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(':'):
                            continue
                        # print("line", line)
                        if line.startswith('data:'):
                            line = line.lstrip("data: ")
                            if line == "[DONE]":
                                break
                        else:
                            try:
                                line = json.loads(line)
                                full_response = safe_get(line, "choices", 0, "message", "content")
                                if full_response:
                                    yield full_response
                                else:
                                    yield line
                                break
                            except:
                                print("json.loads error:", repr(line))
                                continue
                        resp: dict = json.loads(line)
                        if "error" in resp:
                            raise Exception(f"{resp}")
                        total_tokens = total_tokens or safe_get(resp, "usage", "total_tokens", default=0)
                        delta = safe_get(resp, "choices", 0, "delta")
                        if not delta:
                            continue
                        response_role = response_role or safe_get(delta, "role")
                        if safe_get(delta, "content"):
                            need_function_call = False
                            content = delta["content"]
                            full_response += content
                            yield content
                        if safe_get(delta, "tool_calls"):
                            need_function_call = True
                            function_call_name = function_call_name or safe_get(delta, "tool_calls", 0, "function", "name")
                            function_full_response += safe_get(delta, "tool_calls", 0, "function", "arguments", default="")
                            function_call_id = function_call_id or safe_get(delta, "tool_calls", 0, "id")

                if full_response == "" and function_full_response == "":
                    print("error: full_response or is empty", "full_response", full_response, "function_full_response", function_full_response)
                    continue
                else:
                    break
            except Exception as e:
                print(f"发生了未预料的错误：{e}")
                import traceback
                traceback.print_exc()
                if "Invalid URL" in str(e):
                    e = "You have entered an invalid API URL, please use the correct URL and use the `/start` command to set the API URL again. Specific error is as follows:\n\n" + str(e)
                    raise Exception(f"{e}")
                raise Exception(f"{e}")
        if self.print_log:
            print("\n\rtotal_tokens", total_tokens)
        if response_role == None:
            response_role = "assistant"
        if need_function_call:
            function_full_response = check_json(function_full_response)
            print("function_full_response", function_full_response)
            function_response = ""
            # print(self.function_calls_counter)
            if not self.function_calls_counter.get(function_call_name):
                self.function_calls_counter[function_call_name] = 1
            else:
                self.function_calls_counter[function_call_name] += 1
            if self.function_calls_counter[function_call_name] <= self.function_call_max_loop and (function_full_response != "{}" or function_call_name == "get_date_time_weekday"):
                function_call_max_tokens = self.truncate_limit - 1000
                if function_call_max_tokens <= 0:
                    function_call_max_tokens = int(self.truncate_limit / 2)
                print("\033[32m function_call", function_call_name, "max token:", function_call_max_tokens, "\033[0m")
                async for chunk in get_tools_result_async(function_call_name, function_full_response, function_call_max_tokens, model or self.engine, chatgpt, kwargs.get('api_key', self.api_key), self.api_url, use_plugins=False, model=model or self.engine, add_message=self.add_to_conversation, convo_id=convo_id, language=language):
                    if "function_response:" in chunk:
                        function_response = chunk.replace("function_response:", "")
                    else:
                        yield chunk
            else:
                function_response = "无法找到相关信息，停止使用 tools"
            response_role = "tool"
            # print(self.conversation[convo_id][-1])
            # if self.conversation[convo_id][-1]["role"] == "tool" and self.conversation[convo_id][-1]["name"] == "get_search_results":
            #     mess = self.conversation[convo_id].pop(-1)
                # print("Truncate message:", mess)
            async for chunk in self.ask_stream_async(function_response, response_role, convo_id=convo_id, function_name=function_call_name, total_tokens=total_tokens, model=model or self.engine, function_arguments=function_full_response, function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key), plugins=kwargs.get("plugins", PLUGINS), system_prompt=system_prompt):
                yield chunk
        else:
            # if self.conversation[convo_id][-1]["role"] == "tool" and self.conversation[convo_id][-1]["name"] == "get_search_results":
            #     mess = self.conversation[convo_id].pop(-1)
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}
            if pass_history <= 2 and len(self.conversation[convo_id]) >= 2 \
            and (
                "You are a translation engine" in self.conversation[convo_id][-2]["content"] \
                or "You are a translation engine" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="") \
                or "你是一位精通简体中文的专业翻译" in self.conversation[convo_id][-2]["content"] \
                or "你是一位精通简体中文的专业翻译" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="")
            ):
                self.conversation[convo_id].pop(-1)
                self.conversation[convo_id].pop(-1)

    async def ask_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream_async(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            pass_history=pass_history,
            model=model or self.engine,
            **kwargs,
        )
        full_response: str = "".join([r async for r in response])
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally") -> None:
        """
        Reset the conversation
        """
        self.system_prompt = system_prompt or self.system_prompt
        self.conversation[convo_id] = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.tokens_usage[convo_id] = 0

    def save(self, file: str, *keys: str) -> None:
        """
        Save the Chatbot configuration to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            data = {
                key: self.__dict__[key]
                for key in get_filtered_keys_from_object(self, *keys)
            }
            # saves session.proxies dict as session
            # leave this here for compatibility
            data["session"] = data["proxy"]
            del data["aclient"]
            json.dump(
                data,
                f,
                indent=2,
            )

    def load(self, file: Path, *keys_: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            # load json, if session is in keys, load proxies
            loaded_config = json.load(f)
            keys = get_filtered_keys_from_object(self, *keys_)

            if (
                "session" in keys
                and loaded_config["session"]
                or "proxy" in keys
                and loaded_config["proxy"]
            ):
                self.proxy = loaded_config.get("session", loaded_config["proxy"])
                self.session = httpx.Client(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
                self.aclient = httpx.AsyncClient(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
            if "session" in keys:
                keys.remove("session")
            if "aclient" in keys:
                keys.remove("aclient")
            self.__dict__.update({key: loaded_config[key] for key in keys})