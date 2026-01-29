# arm_application/planners/task_planner.py
"""
动作序列生成模块 由agent调用 返回原子序列 agent执行原子动作
"""
from typing import List, Tuple, Any

class TaskPlanner:
    def plan(self, task: dict) -> List[Tuple[str, ...]]:
        """
        输入: 结构化任务描述（由 Agent 构造）
        输出: 动作序列，每个动作是 (方法名, *参数) 的元组
        
        示例输入:
            task:{"action": "pick_and_place", "object": (0.3, 0.2, 0.1), "target": (0.5, -0.1, 0.05)}
            
        示例输出:
            [
                ("move_to", 0.3, 0.2, 0.2),
                ("move_to", 0.3, 0.2, 0.1),
                ("close_gripper",),
                ...
            ]
        """
        if task["action"] == "pick_and_place":
            return self._plan_pick_and_place(task["object"], task["target"])
        elif task["action"] == "pick":
            return self._plan_pick(task["object"])
        elif task["action"] == "place":
            return self._plan_place(task["target"])
        else:
            raise ValueError(f"Unsupported action: {task['action']}")

    def _plan_pick_and_place(self, obj: Tuple[float, float, float], tgt: Tuple[float, float, float]):
        return self._plan_pick(obj) + self._plan_place(tgt)

    def _plan_pick(self, pose):
        """
        抓取动作序列:
        1.前往目标上方
        2.夹爪对齐
        3.降下夹爪
        4.闭合夹爪
        5.抬起夹爪
        """
        SAFE_HEIGHT = 0.5   #夹爪初始高度
        DIV = 0.187         # 夹爪合适的下降位置

        x, y, z = pose
        above = z + DIV
        return [
            ("move_to", x, y, SAFE_HEIGHT),
            ("move_to", x, y, above),
            ("align_gripper_roll",),
            ("close_gripper",),
            ("move_to", x, y, SAFE_HEIGHT)
        ]

    def _plan_place(self, pose):
        """
        放置动作序列:
        1.前往目标上方
        2.降下夹爪
        3.打开夹爪
        4.抬起夹爪
        """
        SAFE_HEIGHT = 0.5   #夹爪初始高度
        DIV = 0.187         # 夹爪合适的下降位置

        x, y, z = pose
        above = z + DIV
        return [
            ("move_to", x, y, SAFE_HEIGHT),
            ("move_to", x, y, above),
            ("open_gripper",),
            ("move_to", x, y, SAFE_HEIGHT)
        ]