"""ToolRL prompts and parser with request-local turn type and explicit decoding."""
import os
import time
from overrides import override
from toolrl_adapter import RLLAHandler as ToolRLHandler


class RLLAHandler(ToolRLHandler):
    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        data = super()._pre_query_processing_prompting(test_entry)
        data['turn_type'] = 'multi_turn' if 'multi_turn' in test_entry['id'] else 'single_turn'
        return data

    @override
    def _query_prompting(self, inference_data):
        prompt = self._format_prompt(inference_data['message'], inference_data['function'],
                                     turn_type=inference_data['turn_type'])
        input_tokens = len(self.tokenizer.tokenize(prompt))
        maximum = int(os.environ.get('BFCL_MAX_NEW_TOKENS', '4096'))
        available = 1000 if self.max_context_length < input_tokens + 2 else min(maximum, self.max_context_length-input_tokens-2)
        request = dict(model=self.model_path_or_id, prompt=prompt, temperature=0, top_p=1,
                       max_tokens=available, seed=0, extra_body={'top_k': -1, 'repetition_penalty': 1})
        inference_data['inference_input_log'] = dict(formatted_prompt=prompt, turn_type=inference_data['turn_type'],
                                                    temperature=0, top_p=1, max_tokens=available, seed=0)
        started = time.monotonic()
        response = self.client.completions.create(**request, timeout=300)
        return response, time.monotonic()-started


def register():
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, local_inference_model_map, ModelConfig
    cfg = ModelConfig(model_name='rlla-eval', display_name='RLLA evaluation',
                      url='https://github.com/qiancheng0/ToolRL', org='MSI', license='Apache-2.0',
                      model_handler=RLLAHandler, is_fc_model=False, underscore_to_dot=False)
    MODEL_CONFIG_MAPPING['rlla-eval'] = cfg
    local_inference_model_map['rlla-eval'] = cfg
