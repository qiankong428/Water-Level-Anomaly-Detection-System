import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Visualizer:
    """
    可视化类，负责绘制各种图表
    """
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_history(self, history):
        """
        可视化训练过程
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_detection(self, results):
        """
        可视化异常检测结果
        """
        plt.figure(figsize=(15, 8))
        
        # 绘制水位变化
        plt.subplot(2, 1, 1)
        plt.plot(results['date'], results['waterlevels'], label='水位')
        plt.title('水位变化')
        plt.xlabel('时间')
        plt.ylabel('水位')
        plt.legend()
        
        # 绘制异常概率
        plt.subplot(2, 1, 2)
        plt.plot(results['date'], results['predicted_prob'], 'g-', label='预测异常概率')
        plt.scatter(
            results[results['actual_anomaly'] == 1]['date'],
            results[results['actual_anomaly'] == 1]['predicted_prob'],
            color='red', label='实际异常点'
        )
        plt.axhline(y=0.5, color='r', linestyle='--', label='阈值')
        plt.title('异常检测结果')
        plt.xlabel('时间')
        plt.ylabel('异常概率')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_slow_change_anomalies(self, results):
        """
        可视化缓变异常
        """
        plt.figure(figsize=(15, 10))
        
        # 绘制水位变化
        plt.subplot(3, 1, 1)
        plt.plot(results['date'], results['waterlevels'], label='水位')
        plt.title('水位变化')
        plt.xlabel('时间')
        plt.ylabel('水位')
        plt.legend()
        
        # 绘制水位变化率
        plt.subplot(3, 1, 2)
        plt.plot(results['date'], results['waterlevels_diff'], label='水位变化率')
        plt.plot(results['date'], results['rolling_diff'], label='24小时水位变化')
        plt.title('水位变化率')
        plt.xlabel('时间')
        plt.ylabel('变化率')
        plt.legend()
        
        # 绘制缓变异常
        plt.subplot(3, 1, 3)
        plt.scatter(
            results[results['slow_rise'] == 1]['date'],
            results[results['slow_rise'] == 1]['waterlevels'],
            color='red', label='缓变异常点'
        )
        plt.plot(results['date'], results['waterlevels'], 'b-', alpha=0.3)
        plt.title('缓变异常检测结果')
        plt.xlabel('时间')
        plt.ylabel('水位')
        plt.legend()
        
        plt.tight_layout()
        plt.show()