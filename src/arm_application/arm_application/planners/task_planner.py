# arm_application/planners/task_planner.py
"""
动作序列生成模块 由agent调用 返回原子序列 agent执行原子动作
"""
from typing import List, Tuple, Any

class TaskPlanner:
    def plan(self, task: dict) -> List[Tuple[str, ...]]:
        """
        输入: 结构化任务描述（由 Agent 构造）
        输出: 动作序列,每个动作是 (方法名, *参数) 的元组
        
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
        if task["action"] == "pick_place":
            return self._plan_pick_place(task["object"], task["target"])
        elif task["action"] == "pick":
            return self._plan_pick(task["object"])
        elif task["action"] == "place":
            return self._plan_place(task["target"])
        elif task["action"] == "reset":
            return self._plan_reset()
        elif task["action"] == "open_gripper":
            return self._plan_open_gripper()
        elif task["action"] == "close_gripper":
            return self._plan_close_gripper()
        else:
            raise ValueError(f"Unsupported action: {task['action']}")

    def _plan_pick_place(self, obj: Tuple[float, float, float], tgt: Tuple[float, float, float]):
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
        DIV = 0.188         # 夹爪合适的下降位置 原先假设目标z为0.05，现在为0.016
        # DIV = 0.188 + 0.05 - 0.016

        x, y, z = pose
        above = z + DIV
        return [
            ("move_to", x, y, SAFE_HEIGHT),
            ("align_gripper_roll",),
            ("move_to", x, y, above),
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
        DIV = 0.23          # 夹爪合适的下降位置
        # DIV = 0.23 + 0.05 - 0.016


        x, y, z = pose
        above = z + DIV
        return [
            ("move_to", x, y, SAFE_HEIGHT),
            # ("align_gripper_roll",),
            ("move_to", x, y, above),
            ("open_gripper",),
            ("move_to", x, y, SAFE_HEIGHT)
        ]

    def _plan_reset(self):
        """
        本身属于原子操作
        """
        return [
            ("reset",)
        ]
    
    def _plan_open_gripper(self):
        """
        本身属于原子操作
        """
        return [
            ("open_gripper",)
        ]
    def _plan_close_gripper(self):
        """
        本身属于原子操作
        """
        return [
            ("close_gripper",)
        ]