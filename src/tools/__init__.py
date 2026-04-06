"""
Tools module - registry and initialization
"""

from src.tools.teaching_assistant_tools import (
    SearchLearningMaterial,
    GetCoursePolicy,
    CalculateGradePenalty,
)

# Tool registry
_teaching_tools = {
    "search_learning_material": SearchLearningMaterial(),
    "get_course_policy": GetCoursePolicy(),
    "calculate_grade_penalty": CalculateGradePenalty(),
}


def init_tools():
    """Initialize all teaching tools"""
    return _teaching_tools


__all__ = [
    "SearchLearningMaterial",
    "GetCoursePolicy",
    "CalculateGradePenalty",
    "init_tools",
]
