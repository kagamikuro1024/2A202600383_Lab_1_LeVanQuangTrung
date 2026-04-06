"""
Agent v2 - Improved ReAct Implementation
Cải tiến so với v1:
1. Support 5 tools (thay vì 3)
2. Better JSON parsing (xử lý markdown JSON blocks)
3. Improved prompt engineering
4. Retry logic when parsing fails
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.openai_provider import OpenAIProvider
from src.tools.teaching_assistant_tools import (
    SearchLearningMaterial,
    GetCoursePolicy,
    CalculateGradePenalty,
    CreateCodeExample,
    CreateLearningRoadmap
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


class ReActAgentV2:
    """
    ReAct Agent v2 - Enhanced Version
    
    Improvements:
    - 5 tools (vs 3 in v1)
    - Better JSON parsing (handles markdown blocks)
    - Improved system prompt
    - Retry mechanism for parsing failures
    """
    
    def __init__(self, provider: str = "openai", max_steps: int = 5):
        """Initialize agent with all 5 tools"""
        self.provider_name = provider
        self.max_steps = max_steps
        self.logger = Logger()
        self.tool_failure_count = 0  # Track consecutive tool failures
        
        # Initialize LLM
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.llm = OpenAIProvider(model_name="gpt-4o", api_key=api_key)
        else:
            raise ValueError(f"Provider {provider} not supported")
        
        # Initialize ALL 5 tools
        self.tools = {
            "search_learning_material": SearchLearningMaterial(),
            "get_course_policy": GetCoursePolicy(),
            "calculate_grade_penalty": CalculateGradePenalty(),
            "create_code_example": CreateCodeExample(),
            "create_learning_roadmap": CreateLearningRoadmap(),
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
        """Build enhanced system prompt with fallback instructions"""
        return f"""Bạn là một AI Teaching Assistant thông minh cho khóa học Lập trình C.

📚 AVAILABLE TOOLS (5 công cụ):
{self.tool_descriptions}

⚡ CRITICAL INSTRUCTIONS:
1. Analyze the question carefully to determine which tool(s) to use
2. Output ONLY raw JSON (no markdown, no backticks, no extra text)
3. Format: {{"action": "tool_name", "input": {{"param": "value"}}}}
4. If question needs multiple steps, use one tool at a time then ask for next
5. ALWAYS finish with "Final Answer:" when you have all information needed
6. If tools fail to find data, you can FALLBACK to answer directly using your knowledge
7. When fallback requested, provide "Final Answer:" with your best answer based on knowledge

🔄 EXAMPLE FLOW:
Question: "Tôi nộp muộn 2 ngày, bài tôi bị trừ mấy?"
Response: {{"action": "get_course_policy", "input": {{"policy_type": "late_submission"}}}}
(Tool returns penalty info)
Final Answer: Bạn bị trừ 20% vì nộp trễ 2-3 ngày...

⚡ FALLBACK EXAMPLE:
If tool can't find data:
User: "Tools not helpful. Answer directly"
Final Answer: Based on my knowledge about C programming, the best practice is...

GOLDEN RULES:
✓ Output JSON on first line only
✓ Use "Final Answer:" to conclude
✓ Don't explain before taking action
✓ Keep JSON clean and valid
✓ Fallback mode: Direct answer without tools"""
    
    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced JSON parser - handles markdown blocks better
        """
        # Clean up text first
        text = text.strip()
        
        # Method 1: Try parsing whole response as JSON
        try:
            data = json.loads(text)
            if "action" in data and "input" in data:
                return data
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract JSON from markdown code blocks
        markdown_json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if markdown_json_match:
            try:
                json_str = markdown_json_match.group(1)
                data = json.loads(json_str)
                if "action" in data and "input" in data:
                    return data
            except json.JSONDecodeError:
                pass
        
        # Method 3: Find raw JSON object
        json_matches = re.findall(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        for json_str in json_matches:
            try:
                action = json.loads(json_str)
                if "action" in action and "input" in action:
                    return action
            except json.JSONDecodeError:
                continue
        
        # Method 4: More flexible regex for nested JSON
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*"action"(?:[^{}]|(?:\{[^{}]*\}))*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        for json_str in json_matches:
            try:
                # Clean potential issues
                json_str = json_str.replace('\n', ' ').replace('\\n', ' ')
                action = json.loads(json_str)
                if "action" in action and "input" in action:
                    return action
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return result as dict"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' không tồn tại. Các tools có sẵn: {list(self.tools.keys())}",
                "tool_not_found": True
            }
        
        tool = self.tools[tool_name]
        
        try:
            # Execute tool with all input parameters
            result = tool.execute(**tool_input)
            result_dict = json.loads(result) if isinstance(result, str) else result
            return result_dict
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _is_tool_failure(self, tool_result: Dict[str, Any]) -> bool:
        """Check if tool failed or couldn't find data"""
        return (
            tool_result.get("success") == False 
            or "error" in tool_result
            or "not found" in str(tool_result).lower()
        )
    
    def _is_final_answer(self, text: str) -> bool:
        """Check if LLM has provided final answer"""
        return "final answer:" in text.lower()
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from LLM output"""
        match = re.search(r'final answer:\s*(.*?)(?:$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run the enhanced ReAct agent loop with 5 tools
        """
        start_time = time.time()
        
        # Log request
        self.logger.log_event({
            "event": "AGENT_V2_REQUEST",
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "provider": self.provider_name,
            "max_steps": self.max_steps,
            "tools_available": list(self.tools.keys())
        })
        
        # Initialize
        trace = []
        loop_count = 0
        total_tokens = 0
        total_latency = 0
        conversation_prompt = ""
        parse_error_count = 0
        tool_failure_count = 0
        fallback_triggered = False  # Track if we've triggered fallback mode
        
        print(f"\n{'='*90}")
        print(f"🤖 AGENT v2 - ENHANCED REASONING (5 Tools)")
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
                
                # Check if final answer first
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
                        "event": "AGENT_V2_SUCCESS",
                        "timestamp": datetime.now().isoformat(),
                        "steps": loop_count,
                        "parse_errors": parse_error_count,
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
                        "parse_errors": parse_error_count,
                        "trace": trace,
                        "type": "AGENT_v2"
                    }
                
                # Parse action with improved parser
                action = self._parse_action(llm_response)
                
                if not action:
                    # No valid action found
                    parse_error_count += 1
                    print(f"⚠️  Parse Error #{parse_error_count}: No valid JSON action found.")
                    
                    trace.append({
                        "step": loop_count,
                        "type": "PARSE_ERROR",
                        "content": "No valid action found in LLM response",
                        "latency_ms": int((time.time() - step_start) * 1000)
                    })
                    
                    if parse_error_count >= 3:
                        # Give up after 3 parse errors
                        print(f"\n❌ Too many parse errors. Asking LLM to provide answer directly.")
                        conversation_prompt += f"\n\nAssistant: {llm_response}\n\nUser: Lỗi parse action JSON! Hãy cung cấp 'Final Answer:' với câu trả lời trực tiếp."
                    else:
                        # Ask for retry
                        conversation_prompt += f"\n\nAssistant: {llm_response}\n\nUser: Lỗi: Hãy output JSON action với format: {{\"action\": \"tool_name\", \"input\": {{...}}}}"
                    
                    continue
                
                # Execute action
                tool_name = action.get("action", "")
                tool_input = action.get("input", {})
                
                print(f"🔧 Action: {tool_name}")
                print(f"   Input: {tool_input}")
                
                tool_result = self._execute_tool(tool_name, tool_input)
                observation = json.dumps(tool_result, ensure_ascii=False)
                
                obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"👁️  Observation:")
                print(f"{obs_preview}")
                
                # Check if tool failed
                is_failure = self._is_tool_failure(tool_result)
                if is_failure:
                    tool_failure_count += 1
                    print(f"⚠️  Tool failure #{tool_failure_count}")
                else:
                    tool_failure_count = 0  # Reset on success
                
                trace.append({
                    "step": loop_count,
                    "type": "TOOL_CALL",
                    "tool": tool_name,
                    "input": tool_input,
                    "observation": observation[:100] + "..." if len(observation) > 100 else observation,
                    "latency_ms": int((time.time() - step_start) * 1000)
                })
                
                # FALLBACK MECHANISM: If too many tool failures OR near max steps
                # Ask LLM to answer directly without more tools
                if (tool_failure_count >= 2 or loop_count >= self.max_steps - 1) and not fallback_triggered:
                    fallback_triggered = True
                    print(f"\n🔄 FALLBACK TRIGGERED: Tools not helpful. Asking LLM for direct answer...\n")
                    
                    conversation_prompt += f"\n\nAssistant: {llm_response}\n\nObservation từ tool {tool_name}:\n{observation}"
                    conversation_prompt += f"\n\nUser: Các tools không tìm được dữ liệu liên quan. Hãy trả lời trực tiếp dựa trên kiến thức của bạn. Cung cấp 'Final Answer:' với câu trả lời hoàn chỉnh."
                else:
                    # Normal flow: add observation and continue
                    conversation_prompt += f"\n\nAssistant: {llm_response}\n\nObservation từ tool {tool_name}:\n{observation}"
            
            # Max steps reached
            self.logger.log_event({
                "event": "AGENT_V2_MAX_STEPS",
                "timestamp": datetime.now().isoformat(),
                "max_steps": self.max_steps,
                "parse_errors": parse_error_count
            })
            
            return {
                "success": False,
                "query": user_query,
                "error": f"Agent reached max steps ({self.max_steps}) without providing final answer",
                "steps": loop_count,
                "parse_errors": parse_error_count,
                "total_latency_ms": int((time.time() - start_time) * 1000),
                "total_tokens": total_tokens,
                "trace": trace,
                "type": "AGENT_v2"
            }
        
        except Exception as e:
            self.logger.log_event({
                "event": "AGENT_V2_ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "steps": loop_count,
                "parse_errors": parse_error_count
            })
            
            return {
                "success": False,
                "query": user_query,
                "error": str(e),
                "steps": loop_count,
                "parse_errors": parse_error_count,
                "total_latency_ms": int((time.time() - start_time) * 1000),
                "total_tokens": total_tokens,
                "trace": trace,
                "type": "AGENT_v2"
            }
