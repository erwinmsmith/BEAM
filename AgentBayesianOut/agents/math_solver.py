from typing import List, Any, Dict

from AgentDropout.graph.node import Node
from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.tools.coding.python_executor import execute_code_get_return
from datasets.gsm8k_dataset import gsm_get_predict

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "MathSolver", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        
    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        if self.role == "Math Solver":
            hints = []
            for id, info in spatial_info.items():
                predict = gsm_get_predict(info.get("output", ""))
                if predict:
                    hints.append(predict)
            for id, info in temporal_info.items():
                predict = gsm_get_predict(info.get("output", ""))
                if predict:
                    hints.append(predict)
            if hints:
                user_prompt += "(Hint: The answer is near to " + " ".join(hints) + ")."
        else:
            for id, info in spatial_info.items():
                output = info.get("output", "")
                if output:
                    spatial_str += f"Agent {id} as a {info.get('role', '')} his answer to this question is:\n\n{output}\n\n"
            for id, info in temporal_info.items():
                output = info.get("output", "")
                if output:
                    temporal_str += f"Agent {id} as a {info.get('role', '')} his answer to this question was:\n\n{output}\n\n"
            user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if spatial_str else ""
            user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if temporal_str else ""
        return [system_prompt, user_prompt]
    
    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self.llm.agen(message)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
        return response