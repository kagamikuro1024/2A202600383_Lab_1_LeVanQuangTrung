"""
Agent v1 - ReAct Loop Implementation
Implements Thought -> Action -> Observation cycle
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.openai_provider import OpenAIProvider
from src.tools.teaching_assistant_tools import (
    SearchLearningMaterial,
    GetCoursePolicy,
    CalculateGradePenalty
)

# Simple Logger
class Logger:
    """Simple logger for agent"""
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        Path(log_dir).mkdir(exist_ok=True)
    
    def log_event(self, event_data: Dict[str, Any]):
        """Log event to file"""
        log_file = Path(self.log_dir) / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")


class ReActAgent:
    """
    ReAct Agent - Reasoning + Acting
    
    Workflow:
    1. Receive user query
    2. Thought: LLM thinks about what to do
    3. Action: LLM decides which tool to call
    4. Observation: Tool executes and returns result
    5. Loop back to 2 until LLM outputs "Final Answer"
    """
    
    def __init__(self, provider: str = "openai", max_steps: int = 10):
        """Initialize agent with tools"""
        self.provider_name = provider
        self.max_steps = max_steps
        self.logger = Logger()
        
        # Initialize LLM
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.llm = OpenAIProvider(model_name="gpt-4o", api_key=api_key)
        else:
            raise ValueError(f"Provider {provider} not supported")
        
        # Initialize tools
        self.tools = {
            "search_learning_material": SearchLearningMaterial(),
            "get_course_policy": GetCoursePolicy(),
            "calculate_grade_penalty": CalculateGradePenalty(),
        }
        
        # Tool descriptions for system prompt
        self.tool_descriptions = self._build_tool_descriptions()
        self.system_prompt = self._build_system_prompt()
    
    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for LLM"""
        descriptions = []
        for tool_name, tool_obj in self.tools.items():
            descriptions.append(f"- {tool_name}: {tool_obj.description}")
        return "\n".join(descriptions)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with ReAct instructions"""
        return f"""Bạn là một AI Teaching Assistant thông minh cho khóa học Lập trình C.

Bạn có quyền truy cập vào các công cụ (tools) sau:
{self.tool_descriptions}

IMPORTANT: Bạn PHẢI sử dụng công cụ thích hợp để trả lời câu hỏi, đặc biệt đối với:
- Tài liệu học tập → sử dụng search_learning_material
- Quy định môn học → sử dụng get_course_policy
- Tính điểm trừ → sử dụng calculate_grade_penalty

Hãy suy luận từng bước (Thought - Action - Observation):

1. **Thought**: Bạn suy nghĩ về câu hỏi và xác định cần dùng tool nào
2. **Action**: Bạn gọi tool với format JSON sau:
   {{"action": "tool_name", "input": {{"param1": "value1", "param2": "value2"}}}}
3. **Observation**: Bạn nhận kết quả từ tool
4. **Thought**: Lặp lại nếu cần thêm thông tin
5. **Final Answer**: Khi bạn có đủ thông tin, output "Final Answer: " + câu trả lời chi tiết

Luôn ghép nối các thông tin từ tools lại với nhau một cách hợp lý.
Nếu tool trả về lỗi, hãy thử cách khác hoặc giải thích tại sao không thể trả lời."""
    
    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse action from LLM output
        Looks for JSON format: {"action": "tool_name", "input": {...}}
        """
        # First, try to parse the entire response as JSON
        try:
            data = json.loads(text.strip())
            if "action" in data and "input" in data:
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the text
        json_matches = re.findall(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        
        for json_str in json_matches:
            try:
                action = json.loads(json_str)
                if "action" in action and "input" in action:
                    return action
            except json.JSONDecodeError:
                continue
        
        # Try a more flexible regex for nested JSON
        json_matches = re.findall(r'\{.*?"action".*?\}', text, re.DOTALL)
        for json_str in json_matches:
            try:
                action = json.loads(json_str)
                if "action" in action and "input" in action:
                    return action
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool and return the result as string
        """
        if tool_name not in self.tools:
            return json.dumps({
                "success": False,
                "error": f"Tool '{tool_name}' không tồn tại"
            }, ensure_ascii=False)
        
        tool = self.tools[tool_name]
        
        try:
            # Execute tool with all input parameters
            result = tool.execute(**tool_input)
            return result
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)
    
    def _is_final_answer(self, text: str) -> bool:
        """Check if LLM has provided final answer"""
        return "final answer:" in text.lower()
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from LLM output"""
        # Find "Final Answer:" and get everything after it
        match = re.search(r'final answer:\s*(.*?)(?:$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run the ReAct agent loop
        
        Returns:
        {
            "success": bool,
            "query": str,
            "answer": str,
            "steps": int,
            "total_latency_ms": int,
            "total_tokens": int,
            "trace": [...]  # All reasoning steps
        }
        """
        start_time = time.time()
        
        # Log request
        self.logger.log_event({
            "event": "AGENT_REQUEST",
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "provider": self.provider_name,
            "max_steps": self.max_steps
        })
        
        # Initialize
        trace = []
        loop_count = 0
        total_tokens = 0
        total_latency = 0
        conversation_prompt = ""
        
        print(f"\n{'='*90}")
        print(f"🤖 AGENT v1 - REASONING LOOP")
        print(f"{'='*90}")
        print(f"❓ Query: {user_query}\n")
        
        try:
            # Build initial prompt
            conversation_prompt = f"Question: {user_query}"
            
            while loop_count < self.max_steps:
                loop_count += 1
                step_start = time.time()
                
                print(f"\n{'─'*90}")
                print(f"Step {loop_count}")
                print(f"{'─'*90}")
                
                # Call LLM
                llm_result = self.llm.generate(
                    prompt=conversation_prompt,
                    system_prompt=self.system_prompt
                )
                
                llm_response = llm_result["content"]
                total_latency += llm_result["latency_ms"]
                total_tokens += llm_result["usage"]["total_tokens"]
                
                print(f"🧠 LLM Output:")
                response_preview = llm_response[:250] + "..." if len(llm_response) > 250 else llm_response
                print(f"{response_preview}")
                
                # Check if final answer
                if self._is_final_answer(llm_response):
                    final_answer = self._extract_final_answer(llm_response)
                    
                    trace.append({
                        "step": loop_count,
                        "type": "FINAL_ANSWER",
                        "content": final_answer,
                        "latency_ms": int((time.time() - step_start) * 1000)
                    })
                    
                    print(f"\n✅ FINAL ANSWER:")
                    print(f"{final_answer[:300] if len(final_answer) > 300 else final_answer}")
                    
                    # Log success
                    self.logger.log_event({
                        "event": "AGENT_SUCCESS",
                        "timestamp": datetime.now().isoformat(),
                        "steps": loop_count,
                        "answer": final_answer[:100] + "..."
                    })
                    
                    total_time = int((time.time() - start_time) * 1000)
                    
                    return {
                        "success": True,
                        "query": user_query,
                        "answer": final_answer,
                        "steps": loop_count,
                        "total_latency_ms": total_time,
                        "total_tokens": total_tokens,
                        "trace": trace,
                        "type": "AGENT_v1"
                    }
                
                # Parse action
                action = self._parse_action(llm_response)
                
                if not action:
                    # No valid action found
                    print(f"⚠️  No valid action found. LLM response doesn't contain proper JSON action.")
                    
                    trace.append({
                        "step": loop_count,
                        "type": "PARSE_ERROR",
                        "content": "No valid action found in LLM response",
                        "latency_ms": int((time.time() - step_start) * 1000)
                    })
                    
                    # Add LLM response and ask for retry
                    conversation_prompt += f"\n\nAssistant: {llm_response}\n\nUser: Lỗi: Không tìm thấy action JSON hợp lệ. Vui lòng cung cấp action với format: {{\"action\": \"tool_name\", \"input\": {{...}}}}"
                    
                    continue
                
                # Execute action
                tool_name = action.get("action", "")
                tool_input = action.get("input", {})
                
                print(f"🔧 Action: {tool_name}")
                print(f"   Input: {tool_input}")
                
                observation = self._execute_tool(tool_name, tool_input)
                
                obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"👁️  Observation:")
                print(f"{obs_preview}")
                
                trace.append({
                    "step": loop_count,
                    "type": "TOOL_CALL",
                    "tool": tool_name,
                    "input": tool_input,
                    "observation": observation[:100] + "..." if len(observation) > 100 else observation,
                    "latency_ms": int((time.time() - step_start) * 1000)
                })
                
                # Add to conversation
                conversation_prompt += f"\n\nAssistant: {llm_response}\n\nObservation từ tool {tool_name}:\n{observation}"
            
            # Max steps reached
            self.logger.log_event({
                "event": "AGENT_MAX_STEPS_REACHED",
                "timestamp": datetime.now().isoformat(),
                "max_steps": self.max_steps
            })
            
            return {
                "success": False,
                "query": user_query,
                "error": f"Agent reached max steps ({self.max_steps}) without providing final answer",
                "steps": loop_count,
                "total_latency_ms": int((time.time() - start_time) * 1000),
                "total_tokens": total_tokens,
                "trace": trace,
                "type": "AGENT_v1"
            }
        
        except Exception as e:
            self.logger.log_event({
                "event": "AGENT_ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "steps": loop_count
            })
            
            return {
                "success": False,
                "query": user_query,
                "error": str(e),
                "steps": loop_count,
                "total_latency_ms": int((time.time() - start_time) * 1000),
                "total_tokens": total_tokens,
                "trace": trace,
                "type": "AGENT_v1"
            }
