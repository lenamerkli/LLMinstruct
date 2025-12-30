from jinja2 import Template, Environment
from json import load, dump, loads, JSONDecodeError
from os import listdir, environ
from os.path import getsize, exists
from requests import request, RequestException
from subprocess import Popen, PIPE, run
from threading import Lock
from datetime import datetime
import math
import typing as t
import socket
import sys


if not exists('/opt/llms/index.json'):
    with open('/opt/llms/index.json', 'w') as _f:
        dump({}, _f)
with open('/opt/llms/index.json') as _f:
    LOCAL_MODELS = load(_f)


class BaseLLM:

    def __init__(self):
        pass

    def set_model(self, model_name: str) -> None:
        raise NotImplementedError

    def get_model(self) -> str:
        raise NotImplementedError

    def load_model(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: t.Union[str, t.List[t.Dict[str, str]]], enable_thinking: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, min_p: float = None, n_predict: int = None, grammar: str = None, seed: int = None, chat_template: str = None, template_env: t.Dict[str, t.Any] = None) -> str:
        raise NotImplementedError


class LLaMaCPP:

    def __init__(self):
        """
        Initialize a new instance of LLaMaCPP
        """
        super().__init__()
        self._model_name = None
        self._process = None
        self._readers = 0
        self._read_lock = Lock()
        self._write_lock = Lock()
        self._port = 8432

    def _add_reader(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def _remove_reader(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    @staticmethod
    def min_none(a: t.Any, b: t.Any) -> t.Any:
        """
        Returns the minimum of two values, or the single value if one of them is None.

        :param a: First value
        :param b: Second value
        :return: The minimum of a and b, or a/b if one of them is None
        """
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    def calculate_offload_layers(self, model_name: str, short_model_name: str) -> int:
        """
        Calculates the number of layers to offload

        :param model_name: The name of the model
        :param short_model_name: The short name of the model
        :return: The number of layers to offload
        """
        free_vram = self.check_free_vram() - 256
        llm_size = getsize(f"/opt/llms/{model_name}") / (1024 ** 2)  # from bytes to MiB
        llm_size = llm_size * 1.1  # Adjust for fluctuation
        layers = LOCAL_MODELS[short_model_name]['layers']
        vram_per_layer = llm_size / layers
        return min(int(free_vram / vram_per_layer), layers)

    @staticmethod
    def check_free_vram() -> int:
        """
        Checks the amount of free VRAM on the GPU

        :return: The amount of free VRAM in MiB
        :raises Exception: If `nvidia-smi` fails
        """
        # BEGIN https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560
        nvidia_smi = run([
            'nvidia-smi',
            '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], stdout=PIPE, text=True)
        # END https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560
        if nvidia_smi.returncode != 0:
            raise Exception(nvidia_smi.stderr)
        return int(nvidia_smi.stdout)

    def set_model(self, model_name: str) -> None:
        """
        Sets the model to use

        :param model_name: The file name of the model to use, including the `.gguf` extension
        :return: None
        :raises Exception: If the model is not found
        """
        if model_name not in self.list_available_models():
            raise Exception(f"Model {model_name} not found")
        with self._write_lock:
            self._model_name = model_name

    def get_model(self) -> str:
        return self._model_name

    def load_model(
            self,
            print_log: bool = False,
            seed: int = None,
            threads: int = None,
            kv_cache_type: t.Optional[t.Literal['f16', 'bf16', 'q8_0', 'q5_0', 'q4_0']] = None,
            context: int = None,
            temperature: float = None,
            top_p: float = None,
            top_k: int = None,
            min_p: float = None,
            cpu_only: bool = False,
    ) -> None:
        """
        Load the selected model into memory

        :param print_log: Whether to print the stdout from llama.cpp into the stdout
        :param seed: Random seed for reproducible outputs
        :param threads: The number of threads to use (default: 16)
        :param kv_cache_type: The type of key-value cache to use (default: q8_0)
        :param context: The maximum context size to allocate (depends on the model's default)
        :param temperature: Controls randomness in generation. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic
        :param top_k: Limits sampling to the k most likely tokens at each step. If set to 0 or None, no limit is applied
        :param top_p: Nucleus sampling - only considers tokens whose cumulative probability exceeds the probability threshold p
        :param min_p: Minimum probability threshold for token sampling - excludes tokens below this probability
        :param cpu_only: Use CPU-only mode
        :return: None
        :raises Exception: If a model is already loaded, the model name is not set, or the model is not found
        """
        if self.process_is_alive():
            raise Exception("A model is already loaded. Use stop() before loading a new model.")
        if self._model_name is None:
            raise Exception("Model not set")
        short_name = self.short_model_name(self._model_name)
        if short_name is None:
            raise Exception(f"Model {self._model_name} not found")
        # BEGIN https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
        if seed is None:
            seed = -1
        if threads is None:
            threads = 16
        if kv_cache_type is None:
            kv_cache_type = 'q8_0'
        context = self.min_none(context, LOCAL_MODELS[short_name]['context'])
        index = 'sampling'
        if index not in LOCAL_MODELS[short_name]:
            index = 'sampling_thinking'
        if temperature is None:
            temperature = LOCAL_MODELS[short_name][index]['temperature']
        if top_p is None:
            top_p = LOCAL_MODELS[short_name][index]['top_p']
        if top_k is None:
            top_k = LOCAL_MODELS[short_name][index]['top_k']
        if min_p is None:
            min_p = LOCAL_MODELS[short_name][index]['min_p']
        while is_port_in_use(self._port):
            self._port += 1
        with self._write_lock:
            if not cpu_only:
                offload_layers = self.calculate_offload_layers(self._model_name, short_name)
            else:
                offload_layers = 0
            print(f"Loading model {self._model_name} with {offload_layers} layers offloaded")
            command = [
                '/opt/llama.cpp/bin/llama-server',
                '--threads', str(threads),
                '--ctx-size', str(context),
                '--no-escape',
                '--cache-type-k', kv_cache_type,
                '--cache-type-v', kv_cache_type,
                '--mlock',
                '--n-gpu-layers', str(offload_layers),
                '--model', f'/opt/llms/{self._model_name}',
                '--seed', str(seed),
                '--temp', str(temperature),
                '--top-k', str(top_k),
                '--top-p', str(top_p),
                '--min-p', str(min_p),
                '--host', '127.0.0.1',
                '--port', str(self._port),
                '--alias', short_name,
                '--slots',
                '--metrics',
                '--parallel', '2',
            ]
            # END https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
            if print_log:
                stdout = None
                stderr = None
                print(command)
            else:
                stdout = PIPE
                stderr = PIPE
            self._process = Popen(command, stdout=stdout, stderr=stderr, text=True)
        return None

    def apply_chat_template(self, conversation: t.List[t.Dict[str, str]], enable_thinking: bool = False, chat_template: str = None, template_env: t.Dict[str, t.Any] = None) -> str:
        """
        Applies the chat template to the conversation

        :param conversation: The conversation in ChatML format
        :param enable_thinking: Whether to enable thinking (only supported on certain models)
        :param chat_template: Optional custom chat template to override the default
        :param template_env: Optional dict containing template environment settings and variables
        :return: The conversation as a string
        """
        short_name = self.short_model_name(self._model_name)
        template_str: str = chat_template if chat_template is not None else LOCAL_MODELS[short_name]['chat_template']

        # Create Jinja2 environment with default strftime_now function
        def strftime_now(format_str):
            return datetime.now().strftime(format_str)

        env = Environment()
        env.globals['strftime_now'] = strftime_now

        # Apply custom environment settings from template_env
        if template_env is not None:
            for key, value in template_env.items():
                if key == 'strftime_now':
                    env.globals['strftime_now'] = value
                else:
                    env.globals[key] = value

        template = env.from_string(template_str)

        # Set up default options
        options: t.Dict[str, t.Any] = {
            'messages': conversation,
            'tools': [],
            'add_generation_prompt': True,
            'enable_thinking': False,
        }

        # Apply template_env overrides to options
        if template_env is not None:
            for key, value in template_env.items():
                if key not in ['strftime_now']:  # Skip environment globals
                    options[key] = value

        # Handle enable_thinking based on model capabilities and user preference
        if LOCAL_MODELS[short_name]['thinking']:
            if LOCAL_MODELS[short_name]['optional_thinking']:
                options['enable_thinking'] = enable_thinking
            else:
                options['enable_thinking'] = True
        else:
            options['enable_thinking'] = False

        return template.render(**options)

    def generate(self, prompt: t.Union[str, t.List[t.Dict[str, str]]], enable_thinking: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, min_p: float = None, n_predict: int = None, grammar: str = None, seed: int = None, chat_template: str = None, template_env: t.Dict[str, t.Any] = None, stop: t.List[str] = None) -> str:
        """
        Generate an answer or completion using the large language model based on the prompt
        :param prompt: Either a string containing the prompt text or a list of message dictionaries in ChatML format
        :param enable_thinking: Whether to enable the model's thinking mode, if supported by the model
        :param temperature: Controls randomness in generation. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic
        :param top_k: Limits sampling to the k most likely tokens at each step. If set to 0 or None, no limit is applied
        :param top_p: Nucleus sampling - only considers tokens whose cumulative probability exceeds the probability threshold p
        :param min_p: Minimum probability threshold for token sampling - excludes tokens below this probability
        :param n_predict: Maximum number of tokens to predict/generate
        :param grammar: Optional grammar constraints for structured generation
        :param seed: Random seed for reproducible outputs
        :param chat_template: Optional custom chat template to override the default
        :param template_env: Optional dict containing template environment settings and variables
        :param stop: Optional list of additional stop reasons
        :return: The generated text response from the model
        :raises Exception: If the model is not loaded or the request fails
        """
        if isinstance(prompt, list):
            prompt = self.apply_chat_template(prompt, enable_thinking, chat_template, template_env)
        json_data: t.Dict[str, t.Any] = {
            'prompt': prompt,
        }
        # BEGIN https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
        if temperature is not None:
            json_data['temperature'] = temperature
        if top_k is not None:
            json_data['top_k'] = top_k
        if top_p is not None:
            json_data['top_p'] = top_p
        if min_p is not None:
            json_data['min_p'] = min_p
        if n_predict is not None:
            json_data['n_predict'] = n_predict
        if grammar is not None:
            json_data['grammar'] = grammar
        if seed is not None:
            json_data['seed'] = seed
        if stop is not None:
            json_data['stop'] = stop
            short_name = self.short_model_name(self._model_name)
            json_data['stop'].append(LOCAL_MODELS[short_name]['stop'])
        self._add_reader()
        try:
            # req = request('POST', f"http://127.0.0.1:{self._port}/completion", json=json_data)
            # if req.status_code != 200:
            #     raise Exception(req.text)
            # json_return = req.json()
            # return json_return['content']
            value = stream_passthrough_llama_cpp(json_data, self._port)
            return value
        # END https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
        finally:
            self._remove_reader()

    def generate_probabilities(self, prompt: t.Union[str, t.List[t.Dict[str, str]]], enable_thinking: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, min_p: float = None, n_predict: int = 1, grammar: str = None, seed: int = None, n_probs: int = 1, post_sampling_probs: bool = False, id_slot: int = 0, cache_prompt: bool = True) -> dict:
        """
                Return next-token probabilities (top-N) for each generated step.

                This uses llama.cpp server native endpoint POST /completion with:
                  - n_probs > 0 to request top token probabilities
                  - post_sampling_probs=True to get probabilities (0..1) instead of logprobs

                Notes:
                  - This returns probabilities for the tokens the server *actually generates* (n_predict steps).
                  - To “peek” only the immediate next token distribution, keep n_predict=1.
                  - Setting cache_prompt=False avoids reusing KV cache across unrelated calls.
                """
        if isinstance(prompt, list):
            prompt = self.apply_chat_template(prompt, enable_thinking, None, None)

        json_data: t.Dict[str, t.Any] = {
            "prompt": prompt,
            "stream": False,  # easier: we want a single JSON response
            "n_predict": int(n_predict),
            "n_probs": int(n_probs),
            "post_sampling_probs": bool(post_sampling_probs),
            "id_slot": int(id_slot),
            "cache_prompt": bool(cache_prompt),
        }

        # Optional sampling overrides
        if temperature is not None:
            json_data["temperature"] = float(temperature)
        if top_k is not None:
            json_data["top_k"] = int(top_k)
        if top_p is not None:
            json_data["top_p"] = float(top_p)
        if min_p is not None:
            json_data["min_p"] = float(min_p)
        if grammar is not None:
            json_data["grammar"] = grammar
        if seed is not None:
            json_data["seed"] = int(seed)

        self._add_reader()
        try:
            resp = request("POST", f"http://127.0.0.1:{self._port}/completion", json=json_data)
            if resp.status_code != 200:
                raise Exception(resp.text)

            try:
                data = resp.json()
            except JSONDecodeError as e:
                raise Exception(f"Invalid JSON from llama-server: {e}\nRaw: {resp.text[:5000]}")

            # The README calls it completion_probabilities, but some builds/fields may expose "probs".
            steps = (
                    data.get("completion_probabilities")
                    or data.get("probs")
                    or []
            )

            # Normalize shape so callers can just read:
            #   steps[i]["token"], steps[i]["prob"], steps[i]["top_probs"]
            #
            # If post_sampling_probs=False, server returns logprob/top_logprobs.
            # Convert to prob/top_probs for convenience.
            if steps and not post_sampling_probs:
                for step in steps:
                    if "logprob" in step and "prob" not in step:
                        step["prob"] = float(math.exp(step["logprob"]))
                    if "top_logprobs" in step and "top_probs" not in step:
                        step["top_probs"] = []
                        for tp in step["top_logprobs"]:
                            prob = float(math.exp(tp["logprob"])) if "logprob" in tp else None
                            step["top_probs"].append({
                                "id": tp.get("id"),
                                "token": tp.get("token"),
                                "bytes": tp.get("bytes"),
                                "prob": prob,
                            })

            return {
                "content": data.get("content", ""),
                "model": data.get("model"),
                "prompt": data.get("prompt", prompt),
                "steps": steps,  # per generated token, with top_probs/top_logprobs
                "raw": data,  # keep full response in case you want timings etc.
                "request": json_data,  # helpful for debugging
                "stop_type": data.get("stop_type"),
                "stopping_word": data.get("stopping_word", ""),
            }

        except RequestException as e:
            raise Exception(f"Request to llama-server failed: {e}")
        finally:
            self._remove_reader()

    def process_is_alive(self) -> bool:
        """
        Checks if the process is still running

        :return: True if the process is running, False otherwise
        """
        self._add_reader()
        try:
            if self._process is None:
                return False
            return self._process.poll() is None
        finally:
            self._remove_reader()

    def is_loading(self) -> bool:
        """
        Checks if the model is loading

        :return: True if the model is loading, False otherwise
        """
        self._add_reader()
        try:
            req = request('GET', f"http://127.0.0.1:{self._port}/health")
            return req.status_code == 503
        except RequestException:
            return False
        finally:
            self._remove_reader()

    def is_running(self) -> bool:
        """
        Checks if the model is running

        :return: True if the model is running, False otherwise
        """
        self._add_reader()
        try:
            req = request('GET', f"http://127.0.0.1:{self._port}/health")
            return req.status_code == 200
        except RequestException:
            return False
        finally:
            self._remove_reader()

    def has_error(self) -> bool:
        """
        Checks if the model has an error

        :return: True if the model has an error, False otherwise
        """
        self._add_reader()
        try:
            req = request('GET', f"http://127.0.0.1:{self._port}/health")
            return req.status_code not in [200, 503]
        except RequestException:
            return True
        finally:
            self._remove_reader()

    def stop(self) -> None:
        """
        Stop the model
        """
        with self._write_lock:
            if self._process is None:
                return None
            self._process.terminate()
            return None

    def kill(self):
        """
        Kill the model using SIGKILL
        """
        with self._write_lock:
            if self._process is None:
                return None
            self._process.kill()
            return None

    def get_system_message(self) -> t.List[t.Dict[str, str]]:
        """
        Get the system message for the selected model

        :return: The system message in ChatML format
        """
        short_name = self.short_model_name(self._model_name)
        system_message = LOCAL_MODELS[short_name]['system_message']
        if system_message == '':
            return []
        return [{'role': 'system', 'content': system_message}]

    @staticmethod
    def list_available_models() -> t.List[str]:
        """
        List available models
        """
        directory_list = listdir('/opt/llms/')
        model_list = []
        for entry in directory_list:
            if entry.endswith('.gguf') and LLaMaCPP.short_model_name(entry) is not None:
                model_list.append(entry)
        return model_list

    @staticmethod
    def short_model_name(model_name: str) -> t.Optional[str]:
        """
        Extract the short model name from the long model file name

        :param model_name: The long model file name
        :return: The short model name (or None if not found)
        """
        for model in sorted(LOCAL_MODELS.keys(), key=lambda x: len(x) , reverse=True):
            if model_name.startswith(model):
                return model
        return None


class OpenRouterLLM(BaseLLM):

    def __init__(self) -> None:
        """
        Initialize a new instance of OpenRouterLLM
        """
        super().__init__()
        self._model_name = ''

    def set_model(self, model_name: str) -> None:
        """
        Sets the model to use

        :param model_name: The name of the model to use
        """
        self._model_name = model_name

    def get_model(self) -> str:
        return self._model_name

    def load_model(self) -> None:
        pass

    def is_loading(self) -> bool:  # noqa
        return False

    def is_running(self) -> bool:  # noqa
        return True

    def apply_chat_template(self, conversation: t.List[t.Dict[str, str]], enable_thinking: bool = False) -> str:
        raise NotImplementedError

    def generate(self, prompt: t.Union[str, t.List[t.Dict[str, str]]], enable_thinking: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, min_p: float = None, n_predict: int = None, grammar: str = None, seed: int = None) -> str:
        """
        Generate an answer or completion using the large language model based on the prompt

        :param prompt: Either a string containing the prompt text or a list of message dictionaries in ChatML format
        :param enable_thinking: (UNUSED) Whether to enable the model's thinking mode, if supported by the model
        :param temperature: (UNUSED) Controls randomness in generation. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic
        :param top_k: (UNUSED) Limits sampling to the k most likely tokens at each step. If set to 0 or None, no limit is applied
        :param top_p: (UNUSED) Nucleus sampling - only considers tokens whose cumulative probability exceeds the probability threshold p
        :param min_p: (UNUSED) Minimum probability threshold for token sampling - excludes tokens below this probability
        :param n_predict: (UNUSED) Maximum number of tokens to predict/generate
        :param grammar: (UNUSED) Optional grammar constraints for structured generation
        :param seed: (UNUSED) Random seed for reproducible outputs
        :return: The generated text response from the model
        :raises Exception: If the model is not loaded or the request fails
        """
        if isinstance(prompt, str):
            data: t.Dict[str, t.Any] = {'prompt': prompt}
        else:
            data: t.Dict[str, t.Any] = {'messages': prompt}
        model_name = self._model_name.replace('OpenRouter', '').replace('openrouter', '')
        if model_name[0] == '/':
            model_name = model_name[1:]
        if model_name == 'deepseek/deepseek-chat-v3.1:free':
            data['provider'] = {'ignore': ['OpenInference']}
        # BEGIN https://openrouter.ai/docs/api-reference/overview
        data['model'] = model_name
        response = request(
            method='POST',
            url='https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + environ['OPENROUTER_API_KEY'],
            },
            json=data,
            verify='/etc/pki/tls/certs/ca-bundle.crt',
        )
        if response.status_code != 200:
            raise Exception(response.text)
        response_json = response.json()
        if 'choices' not in response_json:
            raise RequestException(response.text)
        if 'message' in response_json['choices'][0]:
            return response_json['choices'][0]['message']['content']
        elif 'text' in response_json['choices'][0]:
            return response_json['choices'][0]['text']
        # END https://openrouter.ai/docs/api-reference/overview
        else:
            raise RequestException(response.text)

    def stop(self) -> None:
        pass


class PublicAiLLM(BaseLLM):

    def __init__(self) -> None:
        """
        Initialize a new instance of PublicAiLLM
        """
        super().__init__()
        self._model_name = ''

    def set_model(self, model_name: str) -> None:
        """
        Sets the model to use

        :param model_name: The name of the model to use
        """
        self._model_name = model_name

    def get_model(self) -> str:
        return self._model_name

    def load_model(self) -> None:
        pass

    def is_loading(self) -> bool:  # noqa
        return False

    def is_running(self) -> bool:  # noqa
        return True

    def apply_chat_template(self, conversation: t.List[t.Dict[str, str]], enable_thinking: bool = False) -> str:
        raise NotImplementedError

    def generate(self, prompt: t.Union[str, t.List[t.Dict[str, str]]], enable_thinking: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, min_p: float = None, n_predict: int = None, grammar: str = None, seed: int = None) -> str:
        """
        Generate an answer or completion using the large language model based on the prompt

        :param prompt: Either a string containing the prompt text or a list of message dictionaries in ChatML format
        :param enable_thinking: (UNUSED) Whether to enable the model's thinking mode, if supported by the model
        :param temperature: (UNUSED) Controls randomness in generation. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic
        :param top_k: (UNUSED) Limits sampling to the k most likely tokens at each step. If set to 0 or None, no limit is applied
        :param top_p: (UNUSED) Nucleus sampling - only considers tokens whose cumulative probability exceeds the probability threshold p
        :param min_p: (UNUSED) Minimum probability threshold for token sampling - excludes tokens below this probability
        :param n_predict: (UNUSED) Maximum number of tokens to predict/generate
        :param grammar: (UNUSED) Optional grammar constraints for structured generation
        :param seed: (UNUSED) Random seed for reproducible outputs
        :return: The generated text response from the model
        :raises Exception: If the model is not loaded or the request fails
        """
        if isinstance(prompt, str):
            data: t.Dict[str, t.Any] = {'prompt': prompt}
        else:
            data: t.Dict[str, t.Any] = {'messages': prompt}
        model_name = self._model_name.replace('publicai', '')
        if model_name[0] == '/':
            model_name = model_name[1:]
        data['model'] = model_name
        response = request(
            method='POST',
            url='https://api.publicai.co/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + environ['PUBLICAI_API_KEY'],
                'User-Agent': 'SyntheticData',
            },
            json=data,
            verify='/etc/pki/tls/certs/ca-bundle.crt',
        )
        if response.status_code != 200:
            raise Exception(response.text)
        response_json = response.json()
        if 'choices' not in response_json:
            raise RequestException(response.text)
        if 'message' in response_json['choices'][0]:
            return response_json['choices'][0]['message']['content']
        elif 'text' in response_json['choices'][0]:
            return response_json['choices'][0]['text']
        else:
            raise RequestException(response.text)

    def stop(self) -> None:
        pass


def stream_passthrough_llama_cpp(json_data: dict, port: int) -> str:
    json_data['stream'] = True
    url = f"http://127.0.0.1:{port}/completion"
    response = request('POST', url, json=json_data, stream=True)
    if response.status_code != 200:
        raise Exception(response.text)
    full_content = ""
    print(json_data['prompt'], end='\n', flush=True)
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith(b'data: '):
            try:
                event_data = loads(line[len('data: '):])
                token = event_data.get('content', '')
                full_content += token
                if token:
                    print(token, end='', flush=True)
                if event_data.get('stop', False):
                    break
            except JSONDecodeError:
                print(f"[ERROR] Invalid JSON: {line}", file=sys.stderr)
    print('\n')
    return full_content


def get_llm(model_name: str) -> t.Union[LLaMaCPP, OpenRouterLLM]:
    if model_name.startswith('OpenRouter'):
        llm = OpenRouterLLM()
    elif model_name.startswith('publicai'):
        llm = PublicAiLLM()
    elif model_name.endswith('.gguf'):
        llm = LLaMaCPP()
    else:
        raise Exception(f"Model {model_name} not found")
    llm.set_model(model_name)
    return llm

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
