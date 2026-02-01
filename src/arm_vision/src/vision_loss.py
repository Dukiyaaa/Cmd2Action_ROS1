#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉网格测试脚本
在指定范围内以固定间隔生成方块,测试视觉检测精度
"""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
from agents.object_detector import ObjectDetector
from utils.gazebo_box_display import BoxSpawner
import numpy as np
import csv
from datetime import datetime

recommended_k_x = 1.025837
recommended_b_x = -0.009908
recommended_k_y = 1.025445
recommended_b_y = 0.001953
# 基于检测值的校正函数
def correct_detection_value(detected_x, detected_y, k_x=recommended_k_x, b_x=recommended_b_x, k_y=recommended_k_y, b_y=recommended_b_y):
    # x
    corrected_x = detected_x * k_x + b_x
    # y
    corrected_y = detected_y * k_y + b_y
    return corrected_x, corrected_y

def main():
    rospy.init_node('vision_grid_test', anonymous=True)
    object_detector = ObjectDetector()
    box_spawner = BoxSpawner()
    
    # 测试参数
    x_min, x_max = 0.3, 1.0
    y_min, y_max = -0.8, 0.8
    z = 0.05
    step = 0.04
    
    # 计算测试点数量
    x_points = int((x_max - x_min) / step) + 1
    y_points = int((y_max - y_min) / step) + 1
    total_points = x_points * y_points
    
    rospy.loginfo(f"开始视觉网格测试,总测试点: {total_points}")
    rospy.loginfo(f"测试范围: x[{x_min:.2f}, {x_max:.2f}], y[{y_min:.2f}, {y_max:.2f}], z={z:.2f}")
    rospy.loginfo(f"测试间隔: {step:.2f}")
    
    # 存储测试结果
    test_results = []
    
    # 遍历所有测试点
    point_count = 0
    for i in range(x_points):
        for j in range(y_points):
            # 计算当前测试点坐标
            x = x_min + i * step
            y = y_min + j * step
            pos = (x, y, z)
            
            # 生成测试方块（每次只生成一个）
            box_name = f'test_box_{i}_{j}'
            box_spawner.display_test_box(
                box_pos=pos,
                box_name=box_name
            )
            
            # 等待视觉系统检测
            rospy.sleep(0.5)
            
            # 获取检测结果
            detect_pos = object_detector.get_position(0)
            
            # 计算有效检测的平均值
            if detect_pos:
                # 计算误差（只使用校正后的Y值）
                corrected_x, corrected_y = correct_detection_value(detect_pos[0], detect_pos[1])
                opt_pos = (corrected_x, corrected_y, detect_pos[2])
                loss = np.linalg.norm(np.array(pos) - np.array(opt_pos))
                # 记录结果
                test_results.append({
                    'standard_pos': pos,
                    'detected_pos': opt_pos,
                    'loss': loss
                })
                rospy.loginfo(f"测试点 {point_count+1}/{total_points}: 标准坐标={pos}, 检测坐标={opt_pos}, 误差={loss:.4f}")
            else:
                # 检测失败
                rospy.logwarn(f"测试点 {point_count+1}/{total_points}: 标准坐标={pos}, 检测失败")
            
            # 删除测试方块（确保画面上只有一个方块）
            box_spawner.delete_entity(box_name)
            
            # 等待下一个测试
            rospy.sleep(0.5)
            point_count += 1
    
    # 打印测试结果
    rospy.loginfo("\n===== 测试结果汇总 =====")
    rospy.loginfo(f"总测试点: {total_points}")
    rospy.loginfo(f"成功检测点: {len(test_results)}")
    rospy.loginfo(f"检测成功率: {len(test_results)/total_points*100:.2f}%")
    
    # 计算平均误差
    if test_results:
        avg_loss = sum([r['loss'] for r in test_results]) / len(test_results)
        max_loss = max([r['loss'] for r in test_results])
        min_loss = min([r['loss'] for r in test_results])
        rospy.loginfo(f"平均误差: {avg_loss:.4f}")
        rospy.loginfo(f"最大误差: {max_loss:.4f}")
        rospy.loginfo(f"最小误差: {min_loss:.4f}")
    
    # 打印详细结果
    rospy.loginfo("\n===== 详细测试结果 =====")
    for i, result in enumerate(test_results):
        rospy.loginfo(f"测试点 {i+1}: 标准坐标={result['standard_pos']}, 检测坐标={result['detected_pos']}, 误差={result['loss']:.4f}")
    
    # 保存测试结果到CSV文件
    csv_filename = f"vision_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(script_dir, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['测试点编号', '标准X', '标准Y', '标准Z', '检测X', '检测Y', '检测Z', '误差']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, result in enumerate(test_results):
                writer.writerow({
                    '测试点编号': i + 1,
                    '标准X': result['standard_pos'][0],
                    '标准Y': result['standard_pos'][1],
                    '标准Z': result['standard_pos'][2],
                    '检测X': result['detected_pos'][0],
                    '检测Y': result['detected_pos'][1],
                    '检测Z': result['detected_pos'][2],
                    '误差': result['loss']
                })
        
        rospy.loginfo(f"\n===== 测试结果已保存 =====")
        rospy.loginfo(f"CSV文件路径: {csv_path}")
    except Exception as e:
        rospy.logerr(f"保存CSV文件失败: {e}")
    
    rospy.loginfo("视觉网格测试完成")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("视觉网格测试被中断")