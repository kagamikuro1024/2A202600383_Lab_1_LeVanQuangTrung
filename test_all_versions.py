#!/usr/bin/env python3
"""
Comprehensive Test Suite
Test Chatbot Baseline vs Agent v1 vs Agent v2
With all 5 scenarios
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.chatbot import ChatbotBaseline
from src.agent.agent import ReActAgent as AgentV1
from src.agent.agent_v2 import ReActAgentV2 as AgentV2


class ComprehensiveTest:
    """Test all versions with 5 scenarios"""
    
    def __init__(self):
        self.test_scenarios = self._load_scenarios()
        self.all_results = {
            "chatbot": [],
            "agent_v1": [],
            "agent_v2": []
        }
    
    def _load_scenarios(self) -> list:
        """Load 5 scenarios from scenarios_teaching_assistant.txt"""
        # Define scenarios inline (from scenarios_teaching_assistant.txt)
        return [
            {
                "id": 1,
                "question": "Tôi không hiểu con trỏ hoạt động như thế nào. Cho tôi ví dụ thực tế và giải thích chi tiết.",
                "context": "Sinh viên đang học Chapter 5 (Pointers and Arrays) từ K&R",
                "expected_tools": ["search_learning_material"],
                "expected_key": "pointer"
            },
            {
                "id": 2,
                "question": "Tôi nộp bài muộn 2 ngày. Bài của tôi sẽ bị trừ bao nhiêu điểm?",
                "context": "Sinh viên cần biết chính sách nộp bài trễ",
                "expected_tools": ["get_course_policy", "calculate_grade_penalty"],
                "expected_key": "20"
            },
            {
                "id": 3,
                "question": "Làm thế nào để tôi tránh buffer overflow khi dùng strings?",
                "context": "Sinh viên gặp lỗi khi xử lý chuỗi",
                "expected_tools": ["search_learning_material"],
                "expected_key": "buffer"
            },
            {
                "id": 4,
                "question": "Tôi muốn học thêm về đệ quy. Cho tôi roadmap từ cơ bản đến nâng cao.",
                "context": "Sinh viên muốn tự học chủ động",
                "expected_tools": ["create_learning_roadmap", "search_learning_material"],
                "expected_key": "roadmap"
            },
            {
                "id": 5,
                "question": "Tôi có 3 bài tập chưa nộp. Deadline là bao giờ? Tôi còn bao nhiêu thời gian?",
                "context": "Sinh viên cần quản lý thời gian nộp bài",
                "expected_tools": ["get_course_policy"],
                "expected_key": "deadline"
            }
        ]
    
    def run_all_tests(self):
        """Run all versions with all scenarios"""
        print("\n" + "="*120)
        print("🧪 COMPREHENSIVE TEST SUITE: All 3 Versions × 5 Scenarios")
        print("="*120)
        print(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Initialize systems
        print("🚀 Initializing systems...")
        chatbot = ChatbotBaseline(provider="openai")
        agent_v1 = AgentV1(provider="openai", max_steps=10)
        agent_v2 = AgentV2(provider="openai", max_steps=10)
        print("✅ All systems ready!\n")
        
        # Test each scenario
        for scenario in self.test_scenarios:
            self._test_scenario(scenario, chatbot, agent_v1, agent_v2)
        
        # Print summary
        self._print_summary()
        
        # Save results
        self._save_results()
    
    def _test_scenario(self, scenario: dict, chatbot, agent_v1, agent_v2):
        """Test single scenario with all 3 versions"""
        scenario_id = scenario["id"]
        question = scenario["question"]
        
        print(f"\n{'='*120}")
        print(f"📋 SCENARIO {scenario_id}/5")
        print(f"{'='*120}")
        print(f"Question: {question}")
        print(f"Context: {scenario['context']}")
        print(f"Expected tools: {scenario['expected_tools']}\n")
        
        # Test Chatbot
        print(f"{'─'*120}")
        print(f"1️⃣  CHATBOT BASELINE")
        print(f"{'─'*120}")
        chatbot_result = chatbot.chat(question)
        chatbot_result['scenario_id'] = scenario_id
        self.all_results["chatbot"].append(chatbot_result)
        self._print_result_summary(chatbot_result, "chatbot")
        
        # Test Agent v1
        print(f"\n{'─'*120}")
        print(f"2️⃣  AGENT v1")
        print(f"{'─'*120}")
        agent_v1_result = agent_v1.run(question)
        agent_v1_result['scenario_id'] = scenario_id
        self.all_results["agent_v1"].append(agent_v1_result)
        self._print_result_summary(agent_v1_result, "agent_v1")
        
        # Test Agent v2
        print(f"\n{'─'*120}")
        print(f"3️⃣  AGENT v2")
        print(f"{'─'*120}")
        agent_v2_result = agent_v2.run(question)
        agent_v2_result['scenario_id'] = scenario_id
        self.all_results["agent_v2"].append(agent_v2_result)
        self._print_result_summary(agent_v2_result, "agent_v2")
    
    def _print_result_summary(self, result: dict, version: str):
        """Print summary of single result"""
        if result.get("success"):
            print(f"✅ SUCCESS")
            print(f"   Latency: {result.get('total_latency_ms', result.get('latency_ms', '?'))}ms")
            if version.startswith("agent"):
                print(f"   Steps: {result.get('steps', '?')}")
                print(f"   Parse errors: {result.get('parse_errors', 0)}")
            print(f"   Tokens: {result.get('total_tokens', result.get('tokens', {}).get('total_tokens', '?'))}")
            print(f"   Answer preview: {result.get('answer', result.get('response', ''))[:100]}...")
        else:
            print(f"❌ FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            if version.startswith("agent"):
                print(f"   Steps completed: {result.get('steps', 0)}")
    
    def _print_summary(self):
        """Print overall summary"""
        print(f"\n\n{'='*120}")
        print("📊 OVERALL SUMMARY")
        print(f"{'='*120}\n")
        
        # Success rates
        print("✅ SUCCESS RATES:")
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self.all_results[version]
            success = sum(1 for r in results if r.get("success", False))
            print(f"   {version.upper():<15} {success}/5 ({success*20}%)")
        
        # Latency
        print("\n⏱️  LATENCY (milliseconds):")
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self.all_results[version]
            latencies = []
            for r in results:
                if r.get("success"):
                    lat = r.get('total_latency_ms', r.get('latency_ms', 0))
                    latencies.append(lat)
            
            if latencies:
                total = sum(latencies)
                avg = total / len(latencies)
                print(f"   {version.upper():<15} Total: {total}ms | Avg: {avg:.0f}ms | Per: {latencies}")
            else:
                print(f"   {version.upper():<15} N/A (all failed)")
        
        # Tokens
        print("\n📝 TOKEN USAGE:")
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self.all_results[version]
            tokens_list = []
            for r in results:
                if r.get("success"):
                    if version == "chatbot":
                        tok = r.get('tokens', {}).get('total_tokens', 0)
                    else:
                        tok = r.get('total_tokens', 0)
                    tokens_list.append(tok)
            
            if tokens_list:
                total = sum(tokens_list)
                avg = total / len(tokens_list)
                print(f"   {version.upper():<15} Total: {total} | Avg: {avg:.0f} | Per: {tokens_list}")
            else:
                print(f"   {version.upper():<15} N/A (all failed)")
        
        # Steps (agents only)
        print("\n🔄 STEPS (Agents only):")
        for version in ["agent_v1", "agent_v2"]:
            results = self.all_results[version]
            steps_list = [r.get('steps', 0) for r in results if r.get("success")]
            if steps_list:
                avg_steps = sum(steps_list) / len(steps_list)
                print(f"   {version.upper():<15} Avg: {avg_steps:.1f} | Per: {steps_list}")
        
        # Key wins
        print(f"\n{'='*120}")
        print("🏆 KEY FINDINGS:")
        print(f"{'='*120}")
        
        # Q2 accuracy (grade penalty)
        q2_chatbot = self.all_results["chatbot"][1]
        q2_v1 = self.all_results["agent_v1"][1]
        q2_v2 = self.all_results["agent_v2"][1]
        
        print(f"\nQ2 Accuracy Test (Grade Penalty):")
        if q2_chatbot.get("success"):
            resp = q2_chatbot.get("response", "").lower()
            has_20 = "20" in resp or "trừ 20" in resp
            print(f"   Chatbot: {'✅' if has_20 else '❌'} {('Correct' if has_20 else 'No access to real data')}")
        
        if q2_v1.get("success"):
            resp = q2_v1.get("answer", "").lower()
            has_20 = "20" in resp
            print(f"   Agent v1: {'✅' if has_20 else '❌'} {('Correct' if has_20 else 'Wrong')}")
        
        if q2_v2.get("success"):
            resp = q2_v2.get("answer", "").lower()
            has_20 = "20" in resp
            print(f"   Agent v2: {'✅' if has_20 else '❌'} {('Correct' if has_20 else 'Wrong')}")
        
        print(f"\n{'='*120}\n")
    
    def _save_results(self):
        """Save test results to JSON"""
        log_file = Path("logs") / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file.parent.mkdir(exist_ok=True)
        
        save_data = {
            "test_type": "COMPREHENSIVE_ALL_VERSIONS",
            "timestamp": datetime.now().isoformat(),
            "scenarios": len(self.test_scenarios),
            "results": self.all_results
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to: {log_file}")


def main():
    """Main entry point"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    # Run comprehensive tests
    tester = ComprehensiveTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
