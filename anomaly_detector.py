import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

class IsolationForestDetector:
    """
    基于隔离森林的异常检测器
    """
    def __init__(self, contamination='auto', random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.predictions = None
        self.anomaly_scores = None
    
    def fit_detect(self, data, column_name='waterlevels'):
        """
        对指定列数据进行拟合并检测异常
        """
        # 对指定列数据进行拟合
        X = data[[column_name]].values
        self.model.fit(X)
        
        # 预测异常值 (1表示正常，-1表示异常)
        self.predictions = self.model.predict(X)
        
        # 计算异常分数 (越负表示越可能是异常)
        self.anomaly_scores = self.model.decision_function(X)
        
        # 将预测结果和分数添加到原始数据中
        data['anomaly'] = self.predictions
        data['anomaly_score'] = self.anomaly_scores
        data['is_anomaly'] = data['anomaly'].apply(lambda x: '异常' if x == -1 else '正常')
        
        # 输出异常值统计
        anomaly_count = (self.predictions == -1).sum()
        print(f"检测到 {anomaly_count} 个异常值，占比 {anomaly_count/len(data)*100:.2f}%")
        
        return data


class LSTMAnomalyDetector:
    """
    基于LSTM的异常检测器
    """
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """
        构建LSTM模型
        """
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # 编译模型
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, patience=10):
        """
        训练模型
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # 早停策略
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预测
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 打印分类报告
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 打印混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("混淆矩阵:")
        print(cm)
        
        return y_pred_proba, y_pred, cm
    
    def predict_anomalies(self, sequences):
        """
        预测异常概率
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        return self.model.predict(sequences)