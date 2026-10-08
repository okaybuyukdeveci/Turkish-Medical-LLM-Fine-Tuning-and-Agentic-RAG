"""Local Qwen generation with the message interface used by the existing graph."""
from __future__ import annotations

import re
from contextlib import nullcontext
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


def parse_response(text, tools=()):
    """Convert generated Qwen text into a LangChain AIMessage.

    Args:
        text: Newly generated text, excluding the input conversation.
        tools: Bound tool definitions in OpenAI function-schema format.

    Returns:
        An AIMessage containing ordinary text and any validated retrieval call.

    Raises:
        ValueError: A tool call is malformed, unsupported, or repeated.

    Removes completed thinking blocks and parses Qwen's XML-style tool syntax.
    Only retrieve_documents with one nonempty query argument is supported.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        function = re.fullmatch(r"\s*<function=([^>]+)>(.*?)</function>\s*", block, re.S)
        if not function:
            raise ValueError("Malformed Qwen tool call")
        name, body = function.groups()
        args = dict(re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.S))
        allowed = {tool['function']['name'] for tool in tools}
        if name not in allowed or name != 'retrieve_documents' or set(args) != {'query'} or not args['query'].strip():
            raise ValueError("Invalid retrieval tool name or arguments")
        calls.append({'name': name, 'args': args, 'id': uuid4().hex, 'type': 'tool_call'})
    if '<tool_call>' in text and not calls:
        raise ValueError("Incomplete Qwen tool call")
    if len(calls) > 1:
        raise ValueError("The graph supports one retrieval call per turn")
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return AIMessage(content=content, tool_calls=calls)


class QwenChat:
    """Expose a shared local Qwen model through the graph's chat interface.

    Instances share model weights but keep separate adapter and tool settings.
    Generation is synchronous; shared instances should be invoked sequentially.
    """

    def __init__(self, model, tokenizer, *, adapter=False, max_new_tokens=512, max_input_tokens=8192, tools=()):
        """Configure an already loaded model without loading additional weights.

        Args:
            model: Transformers model, optionally wrapped by PEFT.
            tokenizer: Base model tokenizer with its Qwen chat template.
            adapter: Whether to keep the loaded PEFT adapter active during calls.
                False temporarily disables it to evaluate the base model.
            max_new_tokens: Maximum number of tokens generated per invocation.
            max_input_tokens: Maximum input length; longer inputs raise an error.
            tools: OpenAI-format tool schemas exposed to the model.
        """
        self.model, self.tokenizer = model, tokenizer
        self.adapter = adapter
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        self.tools = tools

    def bind_tools(self, tools):
        """Return a wrapper sharing the weights with the supplied tools bound.

        Converts LangChain tools to function schemas for Qwen's chat template.
        The original wrapper keeps its tool configuration unchanged.
        """
        return QwenChat(self.model, self.tokenizer, adapter=self.adapter,
                        max_new_tokens=self.max_new_tokens, max_input_tokens=self.max_input_tokens,
                        tools=[convert_to_openai_tool(tool) for tool in tools])

    def invoke(self, messages):
        """Generate one AIMessage from LangChain conversation messages.

        Converts human, system, AI, and tool messages to Qwen's chat format,
        preserves previous tool calls, and generates without sampling or thinking.
        Uses this wrapper's adapter setting only for the duration of generation.

        Raises:
            ValueError: Input exceeds the configured limit or the generated tool
                call cannot be parsed safely. Inputs are not silently truncated.
        """
        import torch
        history = []
        roles = {'human': 'user', 'ai': 'assistant', 'system': 'system', 'tool': 'tool'}
        for message in messages:
            item = {'role': roles[message.type], 'content': message.content}
            if getattr(message, 'tool_calls', None):
                item['tool_calls'] = [{'type': 'function', 'function': {'name': c['name'], 'arguments': c['args']}} for c in message.tool_calls]
            history.append(item)
        inputs = self.tokenizer.apply_chat_template(history, tools=self.tools or None,
            enable_thinking=False, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt')
        if inputs['input_ids'].shape[-1] > self.max_input_tokens:
            raise ValueError("Qwen context exceeds max_input_tokens; input was not silently truncated")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        context = nullcontext() if self.adapter or not hasattr(self.model, 'disable_adapter') else self.model.disable_adapter()
        with context, torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return parse_response(text, self.tools)


def load_qwen(config, generation, *, with_adapter=True):
    """Load base weights and an optional local LoRA adapter for evaluation.

    Args:
        config: Model settings containing base, dtype, optional device_map, and
            adapter when with_adapter is True. Adapter paths must be resolved.
        generation: QwenChat generation settings such as max_new_tokens and
            max_input_tokens, normally read from benchmark.yaml.
        with_adapter: Whether to load the adapter through PEFT. The returned
            wrapper initially evaluates the base model; enable adapter=True on
            a wrapper sharing these weights to evaluate SFT.

    Returns:
        A QwenChat wrapper with evaluation-mode weights and the base tokenizer.
        Missing base weights may be downloaded by Transformers.
    """
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[config['dtype']]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(config['base'], dtype=dtype,
        device_map=config.get('device_map', 'auto'), attn_implementation='sdpa')
    tokenizer = AutoTokenizer.from_pretrained(config['base'])
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config['adapter'])
    model.eval()
    return QwenChat(model, tokenizer, **generation)
