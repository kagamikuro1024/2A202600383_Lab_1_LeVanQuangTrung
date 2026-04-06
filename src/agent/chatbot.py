"""
Chatbot Baseline - Without tools
A simple chatbot that tries to answer questions without using tools
This will FAIL on multi-step reasoning tasks
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.openai_provider import OpenAIProvider

# Simple Logger for this module
class Logger:
    """Simple logger for chatbot"""
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        Path(log_dir).mkdir(exist_ok=True)
    
    def log_event(self, event_data: Dict[str, Any]):
        """Log event to file"""
        log_file = Path(self.log_dir) / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")

class MetricsCollector:
    """Simple metrics collector"""
    def __init__(self):
        self.metrics = []
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric"""
        self.metrics.append({
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        })

class ChatbotBaseline:
    """Simple chatbot without tools - No reasoning, just Q&A"""
    
    def __init__(self, api_key: Optional[str] = None, log_dir: str = "logs", provider: str = "openai"):
        """Initialize chatbot"""
        self.provider_name = provider
        
        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.llm = OpenAIProvider(model_name="gpt-4o", api_key=api_key)
        else:
            raise ValueError(f"Provider {provider} not supported yet")
        
        # Setup logging
        self.log_dir = log_dir
        Path(log_dir).mkdir(exist_ok=True)
        self.logger = Logger(log_dir=log_dir)
        self.metrics = MetricsCollector()
        
        # System prompt for teaching assistant
        self.system_prompt = """Bạn là một trợ giảng AI thông minh cho khóa học Lập trình C.
Hãy trả lời các câu hỏi của sinh viên một cách chi tiết, rõ ràng và hữu ích.
- Sử dụng tiếng Việt
- Cung cấp ví dụ code khi cần thiết
- Giải thích chi tiết các khái niệm
- KHÔNG gọi bất kỳ công cụ/tool nào
- Chỉ dùng kiến thức của bạn để trả lời"""
    
    def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query and return response WITHOUT using tools
        
        Returns:
        {
            "success": bool,
            "query": str,
            "response": str,
            "latency_ms": int,
            "tokens": {
                "prompt": int,
                "completion": int,
                "total": int
            },
            "type": "CHATBOT_BASELINE"
        }
        """
        start_time = time.time()
        
        # Log request
        self.logger.log_event({
            "event": "CHATBOT_REQUEST",
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "provider": self.provider_name
        })
        
        try:
            # Call LLM without any tools
            result = self.llm.generate(user_query, self.system_prompt)
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            response_data = {
                "success": True,
                "query": user_query,
                "response": result["content"],
                "latency_ms": latency_ms,
                "tokens": result["usage"],
                "provider": result["provider"],
                "type": "CHATBOT_BASELINE"
            }
            
            # Log response
            self.logger.log_event({
                "event": "CHATBOT_RESPONSE",
                "timestamp": datetime.now().isoformat(),
                "response": result["content"][:100] + "...",
                "latency_ms": latency_ms,
                "tokens": result["usage"]
            })
            
            # Collect metrics
            self.metrics.record_metric(
                name="chatbot_latency_ms",
                value=latency_ms,
                tags={"provider": self.provider_name}
            )
            self.metrics.record_metric(
                name="chatbot_tokens_total",
                value=result["usage"]["total_tokens"],
                tags={"provider": self.provider_name}
            )
            
            return response_data
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            self.logger.log_event({
                "event": "CHATBOT_ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "query": user_query
            })
            
            return {
                "success": False,
                "query": user_query,
                "error": str(e),
                "latency_ms": latency_ms,
                "type": "CHATBOT_BASELINE"
            }
    
    def interactive_chat(self):
        """Run interactive chat session"""
        print("\n" + "="*80)
        print("🤖 AI TEACHING ASSISTANT - CHATBOT BASELINE (No Tools)")
        print("="*80)
        print("Nhập câu hỏi của bạn (hoặc 'exit' để thoát)\n")
        
        while True:
            user_input = input("👤 Bạn: ").strip()
            if user_input.lower() in ['exit', 'quit', 'thoát']:
                print("\n👋 Tạm biệt!")
                break
            
            if not user_input:
                continue
            
            response = self.chat(user_input)
            
            if response["success"]:
                print(f"\n🤖 Trợ giảng:")
                print("-" * 80)
                print(response["response"])
                print("-" * 80)
                print(f"⏱️  Latency: {response['latency_ms']}ms")
                print(f"📊 Tokens: {response['tokens']['total']}\n")
            else:
                print(f"\n❌ Lỗi: {response['error']}\n")
