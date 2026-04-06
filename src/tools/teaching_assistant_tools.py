"""
Teaching Assistant Tools - Implementation
Tools cho AI trợ giảng trả lời câu hỏi sinh viên
"""

import json
import os
from typing import Dict, Any, List
from src.tools.base_tool import BaseTool

# Load data từ JSON files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

class SearchLearningMaterial(BaseTool):
    """Tool tìm kiếm tài liệu học tập"""
    
    def __init__(self):
        super().__init__(
            name="search_learning_material",
            description="Tìm kiếm tài liệu học tập theo keyword. Trả về định nghĩa, ví dụ code, lỗi thường gặp. Sử dụng từ khóa như 'pointer', 'loop', 'string', 'array', 'recursion', v.v."
        )
        self.data = self._load_learning_materials()
    
    def _load_learning_materials(self) -> Dict[str, Any]:
        """Load tài liệu từ file JSON"""
        try:
            filepath = os.path.join(DATA_DIR, "tai_lieu_hoc_tap.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}
    
    def execute(self, keyword: str = "", include_examples: bool = True, 
                include_mistakes: bool = True) -> str:
        """
        Tìm kiếm tài liệu theo keyword
        
        Args:
            keyword: từ khóa tìm kiếm (e.g., "pointer", "loop")
            include_examples: có trả về ví dụ code không
            include_mistakes: có trả về lỗi thường gặp không
        
        Returns:
            JSON string chứa kết quả
        """
        keyword = keyword.lower().strip()
        
        if not keyword:
            return json.dumps({
                "success": False,
                "error": "Vui lòng cung cấp từ khóa tìm kiếm"
            }, ensure_ascii=False)
        
        # Tìm trong data
        if keyword not in self.data:
            return json.dumps({
                "success": False,
                "error": f"Không tìm thấy tài liệu về '{keyword}'"
            }, ensure_ascii=False)
        
        material = self.data[keyword]
        result = {
            "success": True,
            "topic": material.get("tên", keyword),
            "definition": material.get("định_nghĩa", ""),
        }
        
        # Thêm examples nếu cần
        if include_examples:
            examples = []
            if "ví_dụ_1" in material:
                examples.append({
                    "code": material["ví_dụ_1"].get("code", ""),
                    "explanation": material["ví_dụ_1"].get("giải_thích", "")
                })
            if "ví_dụ_2" in material:
                examples.append({
                    "code": material["ví_dụ_2"].get("code", ""),
                    "explanation": material["ví_dụ_2"].get("giải_thích", "")
                })
            result["examples"] = examples
        
        # Thêm common mistakes nếu cần
        if include_mistakes:
            result["common_mistakes"] = material.get("common_mistakes", [])
        
        # Thêm tài liệu tham khảo
        result["resources"] = [material.get("tài_liệu", "")]
        
        return json.dumps(result, ensure_ascii=False)


class GetCoursePolicy(BaseTool):
    """Tool lấy quy định môn học"""
    
    def __init__(self):
        super().__init__(
            name="get_course_policy",
            description="Lấy quy định của môn học (deadline, scoring, late submission, grading). Sử dụng policy_type: 'deadline', 'late_submission', 'scoring', 'grading'"
        )
        self.data = self._load_policies()
    
    def _load_policies(self) -> Dict[str, Any]:
        """Load quy định từ file JSON"""
        try:
            filepath = os.path.join(DATA_DIR, "quy_dinh_mon_hoc.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}
    
    def execute(self, policy_type: str = "") -> str:
        """
        Lấy quy định môn học
        
        Args:
            policy_type: loại quy định ('deadline', 'late_submission', 'scoring', 'grading')
        
        Returns:
            JSON string chứa quy định
        """
        policy_type = policy_type.lower().strip()
        
        if not policy_type:
            return json.dumps({
                "success": False,
                "error": "Vui lòng cung cấp loại quy định (deadline, late_submission, scoring, grading)"
            }, ensure_ascii=False)
        
        # Tìm trong data
        if policy_type not in self.data:
            return json.dumps({
                "success": False,
                "error": f"Không tìm thấy quy định loại '{policy_type}'",
                "available_policies": list(self.data.keys())
            }, ensure_ascii=False)
        
        policy = self.data[policy_type]
        result = {
            "success": True,
            "policy_type": policy_type,
            "description": policy.get("mô_tả", ""),
            "details": policy.get("chi_tiết", "")
        }
        
        return json.dumps(result, ensure_ascii=False)


class CalculateGradePenalty(BaseTool):
    """Tool tính điểm bị trừ do nộp trễ"""
    
    def __init__(self):
        super().__init__(
            name="calculate_grade_penalty",
            description="Tính điểm bị trừ do nộp bài muộn. Nhập original_score (0-100) và days_late (số ngày trễ). Trả về điểm cuối cùng và lý do."
        )
        self.penalty_rules = self._load_penalty_rules()
    
    def _load_penalty_rules(self) -> Dict[str, Any]:
        """Load quy tắc trừ điểm từ file JSON"""
        try:
            filepath = os.path.join(DATA_DIR, "quy_dinh_mon_hoc.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("late_submission", {})
        except Exception as e:
            return {}
    
    def execute(self, original_score: int = 0, days_late: int = 0) -> str:
        """
        Tính điểm bị trừ
        
        Args:
            original_score: điểm gốc (0-100)
            days_late: số ngày nộp trễ
        
        Returns:
            JSON string chứa kết quả tính toán
        """
        try:
            original_score = int(original_score)
            days_late = int(days_late)
        except (ValueError, TypeError):
            return json.dumps({
                "success": False,
                "error": "original_score và days_late phải là số nguyên"
            }, ensure_ascii=False)
        
        # Validate input
        if original_score < 0 or original_score > 100:
            return json.dumps({
                "success": False,
                "error": "original_score phải từ 0-100"
            }, ensure_ascii=False)
        
        if days_late < 0:
            return json.dumps({
                "success": False,
                "error": "days_late không thể âm"
            }, ensure_ascii=False)
        
        # Xác định penalty dựa trên quy tắc
        chi_tiet = self.penalty_rules.get("chi_tiết", "")
        
        penalty_percentage = 0
        explanation = ""
        
        # Parse quy tắc từ chi tiết (ví dụ: "Trễ 1 ngày: -10%. Trễ 2-3 ngày: -20%. Trễ >3 ngày: 0")
        if days_late == 0:
            penalty_percentage = 0
            explanation = "Nộp đúng hạn"
        elif days_late == 1:
            penalty_percentage = 10
            explanation = "Nộp trễ 1 ngày: trừ 10%"
        elif 2 <= days_late <= 3:
            penalty_percentage = 20
            explanation = "Nộp trễ 2-3 ngày: trừ 20%"
        else:  # > 3 days
            penalty_percentage = 100
            explanation = "Nộp trễ >3 ngày: 0 điểm"
        
        # Tính điểm cuối cùng
        final_score = original_score * (100 - penalty_percentage) / 100
        
        result = {
            "success": True,
            "original_score": original_score,
            "days_late": days_late,
            "penalty_percentage": penalty_percentage,
            "final_score": round(final_score, 1),
            "explanation": explanation
        }
        
        return json.dumps(result, ensure_ascii=False)


class CreateCodeExample(BaseTool):
    """Tool tạo ví dụ code minh họa"""
    
    def __init__(self):
        super().__init__(
            name="create_code_example",
            description="Tạo ví dụ code minh họa cho một chủ đề. Nhập topic (e.g., 'pointer_arithmetic', 'buffer_overflow', 'recursion'), complexity ('beginner', 'intermediate', 'advanced'). Trả về ví dụ sai và đúng để so sánh."
        )
        self.learning_materials = self._load_learning_materials()
    
    def _load_learning_materials(self) -> Dict[str, Any]:
        """Load tài liệu từ file JSON"""
        try:
            filepath = os.path.join(DATA_DIR, "tai_lieu_hoc_tap.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}
    
    def execute(self, topic: str = "", complexity: str = "beginner", 
                include_wrong_example: bool = True) -> str:
        """
        Tạo ví dụ code
        
        Args:
            topic: chủ đề (e.g., "pointer", "string", "recursion")
            complexity: độ khó (beginner, intermediate, advanced)
            include_wrong_example: có trả về ví dụ SAI không
        
        Returns:
            JSON string chứa ví dụ
        """
        topic = topic.lower().strip()
        complexity = complexity.lower().strip()
        
        if not topic:
            return json.dumps({
                "success": False,
                "error": "Vui lòng cung cấp topic"
            }, ensure_ascii=False)
        
        # Search for topic in learning materials
        if topic not in self.learning_materials:
            return json.dumps({
                "success": False,
                "error": f"Không tìm thấy chủ đề '{topic}' trong database"
            }, ensure_ascii=False)
        
        material = self.learning_materials[topic]
        
        # Build example from existing data
        right_example = {
            "code": material.get("ví_dụ_1", {}).get("code", ""),
            "explanation": material.get("ví_dụ_1", {}).get("giải_thích", "")
        }
        
        wrong_example = None
        if include_wrong_example and "common_mistakes" in material:
            mistakes = material.get("common_mistakes", [])
            if mistakes:
                # Use first mistake as wrong example reference
                wrong_example = {
                    "code": f"// Ví dụ SAI: {mistakes[0] if isinstance(mistakes[0], str) else ''}",
                    "explanation": f"Lỗi: {mistakes[0] if isinstance(mistakes[0], str) else 'Tham khảo common_mistakes'}",
                    "mistake": mistakes[0] if isinstance(mistakes[0], str) else "Vui lòng xem common_mistakes"
                }
        
        result = {
            "success": True,
            "topic": topic,
            "complexity": complexity,
            "right_example": right_example,
            "wrong_example": wrong_example,
            "common_mistakes": material.get("common_mistakes", [])
        }
        
        return json.dumps(result, ensure_ascii=False)


class CreateLearningRoadmap(BaseTool):
    """Tool tạo lộ trình học tập"""
    
    def __init__(self):
        super().__init__(
            name="create_learning_roadmap",
            description="Tạo lộ trình học từ cơ bản -> nâng cao. Nhập topic (e.g., 'recursion', 'pointer'), target_level ('beginner', 'intermediate', 'advanced'). Trả về roadmap chi tiết với levels, concepts, tài liệu, bài tập."
        )
        self.learning_materials = self._load_learning_materials()
    
    def _load_learning_materials(self) -> Dict[str, Any]:
        """Load tài liệu từ file JSON"""
        try:
            filepath = os.path.join(DATA_DIR, "tai_lieu_hoc_tap.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {}
    
    def execute(self, topic: str = "", target_level: str = "advanced",
                include_assignments: bool = True) -> str:
        """
        Tạo lộ trình học
        
        Args:
            topic: chủ đề (e.g., "pointer", "recursion")
            target_level: độ khó mục tiêu (beginner, intermediate, advanced)
            include_assignments: có trả về bài tập không
        
        Returns:
            JSON string chứa roadmap
        """
        topic = topic.lower().strip()
        target_level = target_level.lower().strip()
        
        if not topic:
            return json.dumps({
                "success": False,
                "error": "Vui lòng cung cấp topic"
            }, ensure_ascii=False)
        
        if topic not in self.learning_materials:
            return json.dumps({
                "success": False,
                "error": f"Không tìm thấy topic '{topic}'"
            }, ensure_ascii=False)
        
        material = self.learning_materials[topic]
        
        # Create roadmap structure
        roadmap = [
            {
                "level": "Beginner",
                "estimated_hours": 4,
                "concepts": [material.get("tên", topic)],
                "learning_resources": [material.get("tài_liệu", "K&R C Programming")],
                "practice_problems": ["Hiểu định nghĩa cơ bản", "Viết ví dụ đơn giản"]
            }
        ]
        
        # Add intermediate level if needed
        if target_level in ["intermediate", "advanced"]:
            roadmap.append({
                "level": "Intermediate",
                "estimated_hours": 8,
                "concepts": [f"{material.get('tên', topic)} nâng cao", "Áp dụng trong các algorithm"],
                "learning_resources": ["Code examples", "LeetCode problems"],
                "practice_problems": ["Giải quyết bài toán phức tạp", "Debug code có lỗi"]
            })
        
        # Add advanced level
        if target_level == "advanced":
            roadmap.append({
                "level": "Advanced",
                "estimated_hours": 12,
                "concepts": ["Optimization", "Memory management", "Performance tuning"],
                "learning_resources": ["Advanced C programming books", "System programming"],
                "practice_problems": ["Thiết kế thuật toán tối ưu", "Tài liệu khóa học nâng cao"]
            })
        
        result = {
            "success": True,
            "topic": topic,
            "target_level": target_level,
            "roadmap": roadmap,
            "estimated_total_hours": sum(r["estimated_hours"] for r in roadmap),
            "resources": material.get("tài_liệu", "")
        }
        
        return json.dumps(result, ensure_ascii=False)


# Export tools
__all__ = [
    'SearchLearningMaterial',
    'GetCoursePolicy',
    'CalculateGradePenalty',
    'CreateCodeExample',
    'CreateLearningRoadmap'
]
