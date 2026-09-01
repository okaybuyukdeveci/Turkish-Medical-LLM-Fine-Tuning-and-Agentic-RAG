"""Language model online vllm interface for text generation"""

from loguru import logger
from langchain_openai import ChatOpenAI
import json
import traceback

class OnlineInferance:
    """Interface to language model"""

    def __init__(self, config_path):
        self._init_vllm(config_path)

    def _init_vllm(self, config_path):
        with open(config_path, "r") as file:
            data = json.load(file)
        data = data["qwen35"]
        self.api_url = data["api_url"]
        self.token = data["token"]
        self.model_id = data["model_name"]

    def return_model(self):
        """Generate text from prompt"""
        self.client = ChatOpenAI(
            base_url=self.api_url,
            api_key=self.token,
            model=self.model_id,
            temperature=1.0,
            max_tokens=6000,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                #"add_generation_prompt": False,
                #"continue_final_message": True
            },
            
        )
        return self.client

