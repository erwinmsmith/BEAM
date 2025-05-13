from typing import List, Any, Dict
import re
import asyncio

from AgentDropout.graph.node import Node
from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.tools.search.wiki import search_wiki_main

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "AnalyzeAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""
        
    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            output = info.get('output', '')
            if self.role == 'Wiki Searcher' and info.get('role') == 'Knowledgeable Expert':
                queries = find_strings_between_pluses(output)
                if queries:
                    wiki = await search_wiki_main(queries)
                    if wiki:
                        self.wiki_summary = ".\n".join(wiki)
                        user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            if 'None.' not in output:
                spatial_str += f"Agent {id}, role is {info.get('role', '')}, output is:\n\n {output}\n\n"
        for id, info in temporal_info.items():
            output = info.get('output', '')
            if 'None.' not in output:
                temporal_str += f"Agent {id}, role is {info.get('role', '')}, output is:\n\n {output}\n\n"
            
        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if spatial_str else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if temporal_str else ""
        return [system_prompt, user_prompt]
                
    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self.llm.agen(message)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        return response