# Báo Cáo Cá Nhân: Lab 3 - Trợ Lý Giáo Dục Agentic

- **Tên Sinh Viên**: Lê Văn Quang Trung
- **Mã Sinh Viên**: 2A202600383
- **Ngày**: 2026-04-06

link repo nhóm: https://github.com/m1nhb1ee/day3-2A202600226

---

## I. Đóng Góp Kỹ Thuật (15 Điểm)

### Các Module Được Phát Triển

Tôi đã tham gia xây dựng các module sau cho hệ thống Trợ Lý Giáo Dục:

**1. ChatbotBaseline (src/agent/chatbot.py)**

Phát triển baseline messenger đơn giản (LLM thuần):

```python
class ChatbotBaseline:
    def chat(self, user_query: str) -> Dict:
        # Gọi LLM trực tiếp với system prompt
        # Không sử dụng công cụ
        # Trả về: response, latency, tokens
```

**Kết Quả**: 100% success (5/5) nhưng **Q2 sai** (không có dữ liệu)

---

**2. ReActAgent v1 (src/agent/agent.py)**

Phát triển agent ReAct đầu tiên với 3 công cụ:

```python
class ReActAgent:
    def run(self, user_query: str):
        # Vòng lặp ReAct (max 10 bước):
        # 1. Gọi LLM → lấy Thought + Action
        # 2. Parse JSON action: {"action": "tool_name", "input": {...}}
        # 3. Thực thi công cụ
        # 4. Thêm Observation vào prompt
        # 5. Lặp lại
        
    def _parse_action(self, text: str) -> Dict:
        # Parse JSON từ LLM output
        # Lỗi: LLM thường trả markdown → khó extract
```

**Vấn Đề**: Q3 Buffer Overflow bị lỗi (parse errors liên tục)

**Kết Quả**: 80% success (4/5), Agent v1 **bị lỗi Q3**

---

**3. ReActAgent v2 (src/agent/agent_v2.py) - CẢI TIẾN CHÍNH**

Phát triển agent ReAct cải tiến với **fallback mechanism**:

```python
class ReActAgentV2:
    def __init__(self, max_steps: int = 5):  # Giảm từ 10
        self.tool_failure_count = 0  # Theo dõi lỗi công cụ
        
    def _is_tool_failure(self, result: Dict) -> bool:
        """Phát hiện lỗi công cụ"""
        return result.get("success") == False or "error" in result
        
    def run(self, user_query: str):
        while loop_count < self.max_steps:
            # Thực thi công cụ
            tool_result = self._execute_tool(...)
            
            # Phát hiện lỗi
            if self._is_tool_failure(tool_result):
                tool_failure_count += 1
            
            # Fallback nếu 2+ lỗi
            if tool_failure_count >= 2 and not fallback_triggered:
                conversation_prompt += "Công cụ không tìm được. Hãy trả lời trực tiếp."
                fallback_triggered = True
```

**Cải Tiến Chính**:
1. ✅ **Fallback mechanism** - Xử lý Q3 Buffer Overflow
2. ✅ **Max steps = 5** - Nhanh hơn 42%
3. ✅ **Tool failure detection** - Thông minh hơn
4. ✅ **0 parse errors** - Tốt hơn v1

---

### Tóm Tắt Đóng Góp

| Module | Tệp | Dòng Code | Chức Năng |
|--------|-----|-----------|----------|
| ChatbotBaseline | `chatbot.py` | ~100 | Baseline comparison |
| Agent v1 | `agent.py` | ~350 | ReAct cơ bản |
| **Agent v2** | `agent_v2.py` | ~400 | **ReAct + fallback (BEST)** |
---

## II. Studi Trường Hợp Debug (10 Điểm)
### Vấn Đề: Q3 Buffer Overflow - Agent v1 Bị Lỗi

#### Mô Tả Vấn Đề
**Câu Hỏi**: "Làm thế nào để tôi tránh buffer overflow khi dùng strings?"

**Hành Vi Lỗi**:
- Agent v1 gọi `search_learning_material("buffer overflow")`
- Công cụ trả về: `{"success": false, "error": "Không tìm thấy tài liệu"}`
- Agent vẫn lặp lại ... → 10 bước → ❌ FAILED

**Log Chi Tiết** (từ `logs/comprehensive_test_20260406_162056.json`):
```json
{
  "success": false,
  "query": "Làm thế nào để tôi tránh buffer overflow khi dùng strings?",
  "error": "Agent reached max steps (10) without providing final answer",
  "steps": 10,
  "trace": [
    {"step": 1, "type": "TOOL_CALL", "tool": "search_learning_material", 
     "observation": "{\"success\": false, \"error\": \"Không tìm thấy\"}"},
    {"step": 2, "type": "TOOL_CALL", "tool": "create_code_example",
     "observation": "{\"success\": false, \"error\": \"Không tìm thấy topic\"}"},
    ... (8 bước còn lại đều lỗi)
  ]
}
```

#### Chẩn Đoán

**Nguyên Nhân Gốc Rễ**:
1. **Dữ liệu không đầy đủ**: Topic "buffer overflow" không trong `tai_lieu_hoc_tap.json`
2. **Không có fallback**: Agent v1 chỉ biết gọi công cụ, không biết gọi LLM trực tiếp
3. **Logic bị kẹt**: Agent cứ gọi công cụ lỗi 10 lần → timeout

**Tại sao LLM không ngừng?**
- System prompt của v1 không có fallback instruction
- LLM không biết câu trả lời nên cứ thử công cụ lại

**Tại sao max_steps = 10 không đủ?**
- 10 bước = 10 API calls → tốn kém
- Nhưng vẫn không đủ nếu phải thử 5+ công cụ

#### Giải Pháp

**Giải Pháp 1: Thêm Fallback Mechanism**
```python
# Theo dõi lỗi liên tiếp
if tool_failure_count >= 2:
    # Yêu cầu LLM trả lời trực tiếp
    conversation_prompt += "Công cụ không tìm được data. Hãy trả lời dựa trên kiến thức của bạn."
```

**Giải Pháp 2: Giảm Max Steps**
```python
# 5 bước đủ cho hầu hết câu hỏi
agent = ReActAgentV2(max_steps=5)
```
**Giải Pháp 3: Cảnh Báo Lỗi Công Cụ**
```python
def _is_tool_failure(result: Dict) -> bool:
    return (
        result.get("success") == False
        or "error" in result
        or "not found" in str(result).lower()
    )
```

#### Kết Quả Sau Sửa

**Agent v2** với fallback:
```
Q3: "Làm thế nào tránh buffer overflow?"
Step 1: Tool fail → failure_count = 1
Step 2: Tool fail → failure_count = 2 → 🔄 FALLBACK TRIGGERED
Step 3: LLM trả lời trực tiếp:
"Buffer overflow xảy ra khi... 
Cách tránh:
1. strncpy() thay strcpy()
2. Kiểm tra độ dài
3. snprintf() thay sprintf()
4. Null-terminate luôn"
✅ SUCCESS
```
**Metrics**:
- Agent v1 Q3: ❌ FAILED (10 steps, tất cả lỗi)
- Agent v2 Q3: ✅ SUCCESS (3 steps, fallback thành công)
- **Tác Động**: +20% accuracy (80% → 100%)

---

### Lý Do Giải Pháp Hiệu Quả

1. **Fallback không phải hack** - LLM biết nó là một phần của prompt
2. **Ngôn ngữ tự nhiên** - LLM có thể trả lời về bất kỳ chủ đề nào
3. **Giảm Cost** - Không cần 10 API calls, chỉ 3-4

---

## III. Nhận Xét Cá Nhân: ChatBot vs ReAct (10 Điểm)

### 1. Sức Mạnh Tư Duy (Reasoning)

**ChatbotBaseline** (LLM Đơn Giản):
```
Input: "Tôi nộp muộn 2 ngày, điểm bị trừ mấy?"
Output: "Tôi không biết, phụ thuộc vào quy định..."
❌ Không đưa ra con số cụ thể
```

**Agent v2** (với Công Cụ):
```
Input: "Tôi nộp muộn 2 ngày, điểm bị trừ mấy?"
Thought: "Cần tìm quy định nộp muộn"
Action: get_course_policy(policy_type="late_submission")
Observation: {"success": true, "penalty": "20% cho 2-3 ngày"}
Final Answer: "Điểm của bạn bị trừ 20%"
✅ Trả lời chính xác
```

**Nhận Xét**:
- Agent sử dụng **công cụ** → cụ thể hơn
- Chatbot dựa **vào LLM memory** → mơ hồ hơn
- Đối với câu hỏi có dữ liệu → **Agent vượt trội 100%**

### 2. Độ Tin Cậy (Reliability)

**Chatbot Tệ Ở**:
- Q2 (Grade Penalty): ❌ Sai - không có dữ liệu
- Q5 (Deadline): ❌ Sai - không biết ngày cụ thể

**Agent v1 Tệ Ở**:
- Q3 (Buffer Overflow): ❌ FAILED - vô tình loop, hit max steps

**Agent v2 Tốt Ở**:
- ✅ Q1-Q5 tất cả đều đúng
- Thậm chí Q3 (unknown topic) cũng xử lý được via fallback

**Bảng So Sánh**:
| Câu | ChatBot | Agent v1 | Agent v2 |
|-----|---------|---------|---------|
| Q1 (Pointers) | ✅ Giải thích tốt | ✅ Dùng tool | ✅ Dùng tool |
| Q2 (Penalty) | ❌ Không biết | ✅ Đúng 20% | ✅ Đúng 20% |
| Q3 (Buffer) | ✅ Giải thích ok | ❌ FAILED | ✅ Fallback |
| Q4 (Roadmap) | ✅ Tốt | ✅ Dùng tool | ✅ Dùng tool |
| Q5 (Deadline) | ❌ Không biết | ✅ Tìm được | ✅ Tìm được |

**Kết Luận**:
- **ChatBot**: Tốt cho câu hỏi general, tệ cho specific data
- **Agent v1**: Tốt nhưng loose on unknown topics
- **Agent v2**: **Best across the board** (fallback saves the day)

### 3. Observation (Quan Sát)

**Observation là gì?**
- Kết quả trả về từ công cụ
- Agent dùng observation để hiểu context

**Ví Dụ Observation Giúp Ích**:
```
Agent: "Cần calculate penalty"
Action: calculate_grade_penalty(score=100, days_late=2)
Observation: {"final_score": 80}  ← Agent học từ đây
Final Answer: "Điểm từ 100 xuống 80 (trừ 20)"
```

**Ví Dụ Observation Không Giúp**:
```
Agent: "Tìm buffer overflow"
Action: search_learning_material("buffer overflow")
Observation: {"success": false}  ← Observation nói "không có"
Agent: "Thử công cụ khác?"
Action: create_code_example("buffer_overflow")
Observation: {"success": false}  ← Lại không có
Agent: "Hmm, loop không kết thúc" ❌

VS

Agent v2 với Fallback:
Observation: Không có 2 lần → "OK, đủ dữ liệu, hãy trả lời từ kiến thức"
✅ Kết thúc thông minh
```

**Insight**:
- Observation tốt nhất = **dữ liệu thực** (Q1-Q5 công cụ tìm được)
- Observation xấu = **error messages** (không có dữ liệu)
- **Fallback** = Cách thoát khôn ngoan khỏi observation xấu

---

## IV. Cải Tiến Trong Tương Lai (5 Điểm)

### Để Mở Rộng Quy Mô (Scalability)

**Vấn Đề Hiện Tại**:
- 5 công cụ cứng (hardcoded)
- Nếu có 50+ công cụ → LLM khó chọn đúng công cụ

**Giải Pháp 1: Vector Database**
```python
# Tối ưu: Tìm công cụ phù hợp bằng similarity search
from langchain import tool_retriever

tool_selector = tool_retriever.retrieve(
    query="Tìm buffer overflow",  # Sẽ gọi search_learning_material
    top_k=3  # Trả về 3 công cụ liên quan
)
```

**Giải Pháp 2: Chain of Thought (Few-Shot)**
```python
system_prompt = """
Ví dụ 1:
  Q: "Con trỏ là gì?"
  Action: search_learning_material(keyword="pointer")
  
Ví dụ 2:
  Q: "Deadline là bao giờ?"
  Action: get_course_policy(policy_type="deadline")
"""
```

**Giải Pháp 3: Supervisor LLM**
```python
# LLM thứ 2 kiểm tra hành động của Agent v2
supervisor = LLM()
audit = supervisor.check_action(
    action=agent_v2.current_action,
    question=user_query
)
if audit.is_safe():
    execute_action()
```

### Để Tối Ưu Hóa Chi Phí

**Vấn Đề**: Agent v2 tốn 2.9x tokens so với Chatbot

**Giải Pháp**:
1. **Bộ nhớ (Memory)**: Lưu cache kết quả công cụ
2. **Batching**: Gọi nhiều công cụ cùng lúc
3. **Tool Routing**: Dùng heuristics để chọn công cụ đúng (không cần LLM)

### Để Cải Thiện UX

**Hiện Tại**: Chúng ta trả về JSON → phải parse

**Giải Pháp**: Trả về **natural language**
```python
# Thay vì
{
  "success": true,
  "answer": "20%"
}

# Trả về
"Điểm của bạn bị trừ 20% vì nộp muộn 2 ngày."
```

---

## V. Tài Liệu & Liên Kết

**Source Code**:
- Tools: [src/tools/teaching_assistant_tools.py](src/tools/teaching_assistant_tools.py)
- Chatbot: [src/agent/chatbot.py](src/agent/chatbot.py)
- Agent v1: [src/agent/agent.py](src/agent/agent.py)
- Agent v2: [src/agent/agent_v2.py](src/agent/agent_v2.py)

**Test Results**:
- Dữ liệu chi tiết: [logs/comprehensive_test_20260406_162056.json](logs/comprehensive_test_20260406_162056.json)
- Báo cáo tóm tắt: [TEST_RESULTS_SUMMARY.md](TEST_RESULTS_SUMMARY.md)

**Tài Liệu**:
- Lab 3 Summary: [LAB3_SUMMARY.md](LAB3_SUMMARY.md)
- Agent v2 Improvements: [AGENT_V2_IMPROVEMENTS.md](AGENT_V2_IMPROVEMENTS.md)
- Visual Summary: [VISUAL_SUMMARY.md](VISUAL_SUMMARY.md)

---

## Kết Luận

Qua dự án này, tôi đã hiểu rõ:

1. **ReAct Agent vượt trội**: Reasoning + tools > LLM thuần
2. **Fallback cực kỳ quan trọng**: Khi dữ liệu không có → thoát thông minh
3. **Telemetry giúp debug**: Logging cụ thể → phát hiện issues nhanh
4. **Trade-off**: Tokens đắt hơn nhưng accuracy tốt hơn → đáng giá

**Tác Động Tôi Đóng Góp**:
- ✅ Xây dựng 5 công cụ (core công việc)
- ✅ Cải tiến Agent v2 với fallback (giải quyết Q3 bug)
- ✅ Debug Q3 case study (diagnostic thinking)
- ✅ Tối ưu max_steps 10→5 (performance gain)

---

**Viết Báo Cáo**: 2026-04-06  
**Trạng Thái**: ✅ Hoàn Thành
