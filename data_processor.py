import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataProcessor:
    """
    数据处理类，负责数据加载、预处理和特征工程
    """
    def __init__(self):
        self.data = None
        self.scaler = None
        self.scaled_features = None
        self.features = ['temperature', 'humidity', 'winddirection', 'windpower', 'rains', 'waterlevels']
    
    def load_data(self, file_path):
        """
        加载CSV数据文件
        """
        self.data = pd.read_csv(file_path)
        self.data = self.data.dropna()
        return self.data
    
    def preprocess_data(self):
        """
        数据预处理：标准化水位数据，转换时间列为索引
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 标准化水位数据
        scaler = StandardScaler()
        self.data['waterlevels'] = scaler.fit_transform(self.data[['waterlevels']])
        
        # 将时间列转换为datetime类型并设为索引
        if 'index' in self.data.columns:
            self.data['index'] = pd.to_datetime(self.data['index'])
            self.data = self.data.set_index('index')
        
        return self.data
    
    def normalize_features(self):
        """
        对特征进行归一化处理
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 使用MinMaxScaler对特征进行归一化
        self.scaler = MinMaxScaler()
        self.scaled_features = pd.DataFrame(
            self.scaler.fit_transform(self.data[self.features]),
            columns=self.features,
            index=self.data.index
        )
        
        return self.scaled_features
    
    def create_sequences(self, X, y, time_steps=24):
        """
        创建时间序列窗口数据
        X: 特征数据
        y: 标签数据
        time_steps: 窗口大小，默认为24小时
        """
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X.iloc[i:(i + time_steps)].values)
            # 使用窗口最后一个时间点的标签
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    def prepare_train_test_data(self, time_steps=24, test_size=0.2):
        """
        准备训练集和测试集
        """
        if self.scaled_features is None:
            self.normalize_features()
        
        # 创建序列数据
        X_seq, y_seq = self.create_sequences(self.scaled_features, self.data['anomaly'], time_steps=time_steps)
        
        # 将-1转换为1（异常），1转换为0（正常）以便于训练
        y_seq = np.where(y_seq == -1, 1, 0)
        
        # 划分训练集和测试集
        train_size = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        return X_train, X_test, y_train, y_test