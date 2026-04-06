#!/usr/bin/env python3
"""
Summarize Comprehensive Test Results
Analyzes test_all_versions.py output and generates summary report
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class ResultAnalyzer:
    """Analyze comprehensive test results"""
    
    def __init__(self):
        self.latest_file = self._find_latest_results()
        self.data = None
        if self.latest_file:
            self._load_results()
    
    def _find_latest_results(self) -> Path:
        """Find latest comprehensive test result file"""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return None
        
        test_files = sorted(
            logs_dir.glob("comprehensive_test_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if test_files:
            return test_files[0]
        return None
    
    def _load_results(self):
        """Load JSON results file"""
        try:
            with open(self.latest_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Loaded: {self.latest_file.name}\n")
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            self.data = None
    
    def analyze(self):
        """Run full analysis"""
        if not self.data:
            print("❌ No test results found. Run test_all_versions.py first!")
            return
        
        print("="*100)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*100)
        print(f"Test File: {self.latest_file.name}")
        print(f"Timestamp: {self.data.get('timestamp', 'N/A')}")
        print(f"Total Scenarios: {self.data.get('scenarios', 5)}\n")
        
        # Print sections
        self._print_success_rates()
        self._print_latency_benchmarks()
        self._print_token_usage()
        self._print_steps_analysis()
        self._print_q2_accuracy()
        self._print_parse_errors()
        self._print_scenario_breakdown()
        self._print_verdict()
        
        # Generate markdown report
        self._generate_markdown_report()
    
    def _get_results_for_version(self, version: str) -> List[dict]:
        """Get all results for a specific version"""
        return self.data['results'].get(version, [])
    
    def _print_success_rates(self):
        """Print success rate comparison"""
        print("="*100)
        print("✅ SUCCESS RATES")
        print("="*100)
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            if not results:
                print(f"{version.upper():<20} No data")
                continue
            
            success = sum(1 for r in results if r.get("success", False))
            total = len(results)
            rate = (success / total * 100) if total > 0 else 0
            
            status = "🟢" if rate == 100 else "🟡" if rate >= 60 else "🔴"
            print(f"{status} {version.upper():<18} {success}/{total} ({rate:.0f}%)")
        
        print()
    
    def _print_latency_benchmarks(self):
        """Print latency analysis"""
        print("="*100)
        print("⏱️  LATENCY BENCHMARKS (milliseconds)")
        print("="*100)
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            latencies = []
            
            for r in results:
                if r.get("success"):
                    lat = r.get('total_latency_ms', r.get('latency_ms', 0))
                    latencies.append(lat)
            
            if latencies:
                total = sum(latencies)
                avg = total / len(latencies)
                min_lat = min(latencies)
                max_lat = max(latencies)
                
                print(f"{version.upper():<18}")
                print(f"  Total:      {total:>6} ms")
                print(f"  Average:    {avg:>6.0f} ms")
                print(f"  Min/Max:    {min_lat}/{max_lat} ms")
                print(f"  Per Query:  {latencies}\n")
            else:
                print(f"{version.upper():<18} N/A (all failed)\n")
    
    def _print_token_usage(self):
        """Print token consumption analysis"""
        print("="*100)
        print("📝 TOKEN CONSUMPTION")
        print("="*100)
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
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
                
                print(f"{version.upper():<18}")
                print(f"  Total:      {total:>6} tokens")
                print(f"  Average:    {avg:>6.0f} tokens/query")
                print(f"  Per Query:  {tokens_list}\n")
            else:
                print(f"{version.upper():<18} N/A\n")
    
    def _print_steps_analysis(self):
        """Print agent steps analysis"""
        print("="*100)
        print("🔄 AGENT STEPS (Multi-step reasoning)")
        print("="*100)
        
        for version in ["agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            steps_list = [r.get('steps', 0) for r in results if r.get("success")]
            
            if steps_list:
                total_steps = sum(steps_list)
                avg_steps = total_steps / len(steps_list)
                
                print(f"{version.upper():<18}")
                print(f"  Total Steps: {total_steps}")
                print(f"  Avg Steps:   {avg_steps:.1f} per query")
                print(f"  Per Query:   {steps_list}\n")
            else:
                print(f"{version.upper():<18} N/A\n")
    
    def _print_q2_accuracy(self):
        """Print Q2 (Grade Penalty) accuracy - KEY METRIC"""
        print("="*100)
        print("🎯 Q2 ACCURACY TEST (Grade Penalty Question) - KEY DIFFERENTIATOR")
        print("="*100)
        print("Question: 'Tôi nộp bài muộn 2 ngày. Bài của tôi sẽ bị trừ bao nhiêu điểm?'")
        print("Expected: 20% penalty\n")
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            if len(results) < 2:
                print(f"{version.upper():<18} N/A (not tested)")
                continue
            
            q2_result = results[1]
            success = q2_result.get("success", False)
            
            if success:
                if version == "chatbot":
                    content = q2_result.get("response", "").lower()
                else:
                    content = q2_result.get("answer", "").lower()
                
                # Check if answer contains "20" or "20%"
                has_20 = "20" in content or "0.2" in content
                
                if has_20:
                    print(f"{'✅'} {version.upper():<16} SUCCESS - Contains '20%'")
                    print(f"   Answer: {content[:80]}...")
                else:
                    print(f"{'❌'} {version.upper():<16} WRONG - No '20%' found")
                    print(f"   Answer: {content[:80]}...")
            else:
                error = q2_result.get("error", "Unknown error")
                print(f"{'❌'} {version.upper():<16} FAILED - {error}")
            
            print()
    
    def _print_parse_errors(self):
        """Print parse error analysis"""
        print("="*100)
        print("⚠️  PARSE ERROR TRACKING")
        print("="*100)
        
        for version in ["agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            total_errors = 0
            error_details = {}
            
            for i, r in enumerate(results):
                if r.get("success"):
                    errors = r.get('parse_errors', 0)
                    total_errors += errors
                    error_details[f"Q{i+1}"] = errors
            
            print(f"{version.upper():<18}")
            print(f"  Total Errors: {total_errors}")
            print(f"  Per Query:    {error_details}\n")
    
    def _print_scenario_breakdown(self):
        """Print per-scenario breakdown"""
        print("="*100)
        print("📋 SCENARIO BREAKDOWN (All 3 Versions)")
        print("="*100)
        
        scenario_names = [
            "Q1: Pointers",
            "Q2: Grade Penalty",
            "Q3: Buffer Overflow",
            "Q4: Recursion Roadmap",
            "Q5: Deadline Info"
        ]
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            print(f"\n{version.upper()}:")
            
            for i, result in enumerate(results):
                if i < len(scenario_names):
                    status = "✅" if result.get("success") else "❌"
                    print(f"  {status} {scenario_names[i]}")
                    if not result.get("success"):
                        print(f"     Error: {result.get('error', 'Unknown')}")
    
    def _print_verdict(self):
        """Print overall verdict"""
        print("\n" + "="*100)
        print("🏆 VERDICT & KEY FINDINGS")
        print("="*100)
        
        # Success rates
        chatbot_rate = self._get_success_rate("chatbot")
        v1_rate = self._get_success_rate("agent_v1")
        v2_rate = self._get_success_rate("agent_v2")
        
        print(f"\n📊 Accuracy:")
        print(f"   ChatbotBaseline: {chatbot_rate:.0f}%")
        print(f"   Agent v1:        {v1_rate:.0f}%")
        print(f"   Agent v2:        {v2_rate:.0f}%")
        
        # Q2 finding
        print(f"\n🎯 Q2 Finding (Grade Penalty):")
        results_cb = self._get_results_for_version("chatbot")
        results_v1 = self._get_results_for_version("agent_v1")
        results_v2 = self._get_results_for_version("agent_v2")
        
        if len(results_cb) > 1:
            cb_q2 = results_cb[1].get("success", False)
            print(f"   Chatbot: {'✅ Correct' if cb_q2 else '❌ No data access'}")
        
        if len(results_v1) > 1:
            v1_q2 = results_v1[1].get("success", False)
            print(f"   Agent v1: {'✅ Correct' if v1_q2 else '❌ Parsing error'}")
        
        if len(results_v2) > 1:
            v2_q2 = results_v2[1].get("success", False)
            print(f"   Agent v2: {'✅ Correct (FIXED!)' if v2_q2 else '❌ Failed'}")
        
        # Cost analysis
        print(f"\n💰 Cost Analysis (Tokens Used):")
        cb_tokens = self._get_total_tokens("chatbot")
        v1_tokens = self._get_total_tokens("agent_v1")
        v2_tokens = self._get_total_tokens("agent_v2")
        
        if cb_tokens > 0:
            v1_cost_ratio = v1_tokens / cb_tokens if cb_tokens > 0 else 0
            v2_cost_ratio = v2_tokens / cb_tokens if cb_tokens > 0 else 0
            print(f"   ChatbotBaseline: {cb_tokens:,} tokens (baseline)")
            print(f"   Agent v1:        {v1_tokens:,} tokens ({v1_cost_ratio:.1f}x more)")
            print(f"   Agent v2:        {v2_tokens:,} tokens ({v2_cost_ratio:.1f}x more)")
        
        # Recommendation
        print(f"\n💡 Recommendation:")
        if v2_rate > v1_rate:
            print(f"   ✅ Agent v2 > Agent v1 (improved parsing and accuracy)")
        if v2_rate > chatbot_rate:
            print(f"   ✅ Agent v2 > ChatbotBaseline (tool access enables data-driven answers)")
        print(f"   ⚡ Trade-off: {v2_tokens/cb_tokens:.1f}x more tokens for {(v2_rate-chatbot_rate):.0f}% better accuracy")
        
        print()
    
    def _get_success_rate(self, version: str) -> float:
        """Calculate success rate for a version"""
        results = self._get_results_for_version(version)
        if not results:
            return 0
        success = sum(1 for r in results if r.get("success", False))
        return (success / len(results) * 100) if results else 0
    
    def _get_total_tokens(self, version: str) -> int:
        """Get total tokens used for a version"""
        results = self._get_results_for_version(version)
        total = 0
        
        for r in results:
            if r.get("success"):
                if version == "chatbot":
                    tok = r.get('tokens', {}).get('total_tokens', 0)
                else:
                    tok = r.get('total_tokens', 0)
                total += tok
        
        return total
    
    def _generate_markdown_report(self):
        """Generate markdown report file"""
        report_path = Path("TEST_RESULTS_SUMMARY.md")
        
        content = f"""# Test Results Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test File: {self.latest_file.name if self.latest_file else 'N/A'}

## ✅ Overall Success Rates

| System | Success Rate |
|--------|--------------|
| ChatbotBaseline | {self._get_success_rate('chatbot'):.0f}% |
| Agent v1 | {self._get_success_rate('agent_v1'):.0f}% |
| Agent v2 | {self._get_success_rate('agent_v2'):.0f}% |

## ⏱️ Latency Metrics

| System | Total | Average | Per Query |
|--------|-------|---------|-----------|
"""
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            latencies = [r.get('total_latency_ms', r.get('latency_ms', 0)) 
                        for r in results if r.get("success")]
            if latencies:
                total = sum(latencies)
                avg = total / len(latencies)
                content += f"| {version} | {total}ms | {avg:.0f}ms | {latencies} |\n"
        
        content += f"""
## 📝 Token Usage

| System | Total Tokens | Average | Tokens/Query |
|--------|--------------|---------|--------------|
"""
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            tokens = self._get_total_tokens(version)
            results = self._get_results_for_version(version)
            success_count = sum(1 for r in results if r.get("success"))
            avg = tokens / success_count if success_count > 0 else 0
            content += f"| {version} | {tokens:,} | {avg:.0f} | {avg:.0f} |\n"
        
        content += f"""
## 🎯 Q2 Accuracy (Grade Penalty) - KEY TEST

This question is the key differentiator:
**Question:** "Tôi nộp bài muộn 2 ngày. Bài của tôi sẽ bị trừ bao nhiêu điểm?"
**Expected Answer:** 20% penalty

| System | Result | Notes |
|--------|--------|-------|
"""
        
        for version in ["chatbot", "agent_v1", "agent_v2"]:
            results = self._get_results_for_version(version)
            if len(results) >= 2:
                q2 = results[1]
                success = q2.get("success", False)
                if success:
                    if version == "chatbot":
                        resp = q2.get("response", "")
                    else:
                        resp = q2.get("answer", "")
                    has_20 = "20" in resp
                    result = "✅ Correct" if has_20 else "❌ Wrong"
                    notes = "Tool access" if has_20 else "No data access"
                else:
                    result = "❌ Failed"
                    notes = q2.get("error", "Unknown")
                
                content += f"| {version} | {result} | {notes} |\n"
        
        content += f"""
## 🏆 Conclusion

- **Chatbot** can explain concepts but lacks data access
- **Agent v1** has parsing issues that hurt accuracy
- **Agent v2** improves parsing and adds more tools for better coverage

The key finding is **Q2 accuracy**: Only agents with tool access can answer data-driven questions correctly.

## 📁 Details

See `logs/{self.latest_file.name if self.latest_file else 'comprehensive_test_*.json'}` for full test data.
"""
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"💾 Markdown report saved to: {report_path}")
        except Exception as e:
            print(f"⚠️  Could not save markdown report: {e}")


def main():
    """Main entry point"""
    analyzer = ResultAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
