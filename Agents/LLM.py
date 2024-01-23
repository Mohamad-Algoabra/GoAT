from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LLM:
    def __init__(self, model_name, base_url, api_key, temperature=0,
                 max_tokens=0, verbose=False):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.model = self._create_model()

    def _create_model(self):
        parameters = {
            'model_name': self.model_name,
            'base_url': self.base_url,
            'api_key': self.api_key,
            'temperature': self.temperature,
        }

        if self.max_tokens:
            parameters['max_tokens'] = self.max_tokens

        if self.verbose:
            parameters['streaming'] = self.verbose
            parameters['callback_manager'] = CallbackManager([StreamingStdOutCallbackHandler()])

        return ChatOpenAI(**parameters)

    def get_model(self):
        return self.model
