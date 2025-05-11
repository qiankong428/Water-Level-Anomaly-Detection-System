import os
import numpy as np
import pandas as pd

from data_processor import DataProcessor
from anomaly_detector import IsolationForestDetector, LSTMAnomalyDetector
from visualization import Visualizer
from alert_manager import AlertManager

def main():
    # 设置文件路径
    data_file = 'data.csv'
    results_file = 'water_level_anomaly_results.csv'
    
    print("=== 水位异常检测系统 ===")
    print("1. 数据加载与预处理")
    # 数据处理
    data_processor = DataProcessor()
    data = data_processor.load_data(data_file)
    data = data_processor.preprocess_data()
    
    print("\n2. 使用隔离森林进行异常检测")
    # 隔离森林异常检测
    if_detector = IsolationForestDetector()
    data = if_detector.fit_detect(data, column_name='waterlevels')
    
    print("\n3. 特征归一化")
    # 特征归一化
    scaled_features = data_processor.normalize_features()
    
    print("\n4. 准备训练数据")
    # 准备训练数据
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_data(time_steps=24, test_size=0.2)
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")
    
    try:
        print("\n5. 构建并训练LSTM模型")
        # LSTM模型训练
        lstm_detector = LSTMAnomalyDetector()
        model = lstm_detector.build_model((X_train.shape[1], X_train.shape[2]))
        history = lstm_detector.train(X_train, y_train, epochs=50, batch_size=32)
        
        print("\n6. 评估模型")
        # 模型评估
        y_pred_proba, y_pred, cm = lstm_detector.evaluate(X_test, y_test)
        
        print("\n7. 可视化训练过程")
        # 可视化训练过程
        visualizer = Visualizer()
        visualizer.plot_training_history(history)
        
        print("\n8. 检测缓变异常")
        # 对整个数据集进行预测
        all_sequences = []
        for i in range(len(scaled_features) - 24):
            all_sequences.append(scaled_features.iloc[i:i+24].values)
        all_sequences = np.array(all_sequences)
        
        # 预测异常概率
        anomaly_probs = lstm_detector.predict_anomalies(all_sequences)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'date': data.index[24:],
            'actual_anomaly': np.where(data['anomaly'].values[24:] == -1, 1, 0),
            'predicted_prob': anomaly_probs.flatten(),
            'predicted_anomaly': (anomaly_probs > 0.5).flatten().astype(int),
            'waterlevels': data['waterlevels'].values[24:]
        })
        
        # 计算水位变化率
        results['waterlevels_diff'] = results['waterlevels'].diff()
        results['rolling_diff'] = results['waterlevels'].diff(periods=24)  # 24小时变化率
        
        # 标记缓变异常
        results['slow_rise'] = 0
        for i in range(72, len(results)):  # 从第3天开始检查
            # 检查连续3天的水位是否都在上升
            if (results.iloc[i-72:i]['waterlevels_diff'] > 0).sum() >= 60 and results.iloc[i]['predicted_anomaly'] == 1:
                results.iloc[i, results.columns.get_loc('slow_rise')] = 1
        
        print("\n9. 可视化异常检测结果")
        # 可视化异常检测结果
        visualizer.plot_anomaly_detection(results)
        visualizer.plot_slow_change_anomalies(results)
        
        #print("\n10. 处理报警")
        # 报警处理
        #alert_manager = AlertManager()
        #alert_manager.process_results(results)
        
        # 保存结果
        results.to_csv(results_file, index=False)
        print(f"\n分析完成，结果已保存到{results_file}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("注意: 如果出现TensorFlow相关错误，请确保正确安装TensorFlow库")
        print("可以尝试使用以下命令安装: pip install tensorflow")

if __name__ == "__main__":
    main()