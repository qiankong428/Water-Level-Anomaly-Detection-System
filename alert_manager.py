import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import datetime
import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

class AlertManager:
    """
    报警管理类，负责异常报警的触发和记录
    """
    def __init__(self, threshold: float = 0.8, log_file: str = 'alerts.log', email_to: str = '2879295415@qq.com'):
        """
        初始化报警管理器
        threshold: 报警阈值，默认为0.8
        log_file: 报警日志文件路径
        email_to: 接收报警邮件的邮箱地址
        """
        self.threshold = threshold
        self.log_file = log_file
        self.email_to = email_to
        
        # 邮件发送配置
        self.smtp_server = 'smtp.qq.com'  # QQ邮箱SMTP服务器
        self.smtp_port = 587  # SMTP端口
        self.email_from = '替换为邮箱'  # 发送邮件的邮箱
        self.email_password = '替换为邮箱授权码'  # 邮箱授权码（非登录密码）
        
        # 创建图表保存目录
        self.charts_dir = os.path.join(os.path.dirname(os.path.abspath(log_file)), 'alert_charts')
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def check_anomaly(self, scores: np.ndarray) -> List[bool]:
        """
        根据阈值判断是否触发报警
        scores: 异常分数数组
        返回: 布尔数组，表示每个点是否触发报警
        """
        return scores > self.threshold
    
    def log_alert(self, sensor_id: str, score: float, timestamp=None, data: Optional[pd.Series] = None) -> None:
        """
        报警记录保存
        sensor_id: 传感器ID
        score: 异常分数
        timestamp: 时间戳，默认为当前时间
        data: 异常数据点及其上下文数据
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        alert_info = {
            'sensor_id': sensor_id,
            'score': float(score),
            'timestamp': timestamp.isoformat(),
            'alert_level': self._get_alert_level(score)
        }
        
        # 将报警信息写入日志文件
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(alert_info) + '\n')
        
        alert_level = alert_info['alert_level']
        print(f"报警触发: 传感器 {sensor_id}, 异常分数 {score:.4f}, 报警级别 {alert_level}")
        
        # 如果提供了数据，则生成可视化图表并发送邮件报警
        if data is not None:
            chart_path = self._generate_alert_chart(sensor_id, data, timestamp, score)
            self._send_alert_email(sensor_id, score, timestamp, alert_level, data, chart_path)
    
    def _get_alert_level(self, score: float) -> str:
        """
        根据异常分数确定报警级别
        score: 异常分数
        返回: 报警级别 (低、中、高)
        """
        if score < 0.85:
            return "低"
        elif score < 0.95:
            return "中"
        else:
            return "高"
    
    def _get_context_data(self, results: pd.DataFrame, idx) -> pd.Series:
        """
        获取异常点的上下文数据（前后24小时的数据）
        results: 完整的数据集
        idx: 异常点的索引（可能是时间索引或整数索引）
        返回: 包含异常点及其上下文的数据片段
        """
        try:
            # 检查索引类型
            if isinstance(idx, datetime.datetime):
                # 如果是日期时间对象，则获取前后12小时的数据
                start_time = idx - datetime.timedelta(hours=12)
                end_time = idx + datetime.timedelta(hours=12)
                
                # 尝试获取时间范围内的数据
                context_data = results.loc[start_time:end_time]
            elif 'date' in results.columns:
                # 如果索引是整数但存在date列，则使用date列进行筛选
                current_date = results.loc[idx, 'date']
                if isinstance(current_date, datetime.datetime):
                    start_time = current_date - datetime.timedelta(hours=12)
                    end_time = current_date + datetime.timedelta(hours=12)
                    context_data = results[(results['date'] >= start_time) & (results['date'] <= end_time)]
                else:
                    # 如果date列不是日期时间对象，则获取前后12个数据点
                    start_idx = max(0, idx - 12)
                    end_idx = min(len(results) - 1, idx + 12)
                    context_data = results.iloc[start_idx:end_idx+1]
            else:
                # 如果索引是整数且没有date列，则获取前后12个数据点
                start_idx = max(0, idx - 12)
                end_idx = min(len(results) - 1, idx + 12)
                context_data = results.iloc[start_idx:end_idx+1]
            
            # 如果数据不足，则返回异常点本身的数据
            if context_data.empty:
                if isinstance(idx, datetime.datetime):
                    return results.loc[idx:idx]
                else:
                    return results.iloc[[idx]]
            
            return context_data
        except Exception as e:
            print(f"获取上下文数据失败: {e}")
            # 如果出错，至少返回异常点本身的数据
            try:
                if isinstance(idx, datetime.datetime):
                    return results.loc[idx:idx]
                else:
                    return results.iloc[[idx]]
            except:
                # 如果还是失败，返回一个空的DataFrame
                return pd.DataFrame()
    
    def _generate_alert_chart(self, sensor_id: str, data: pd.Series, timestamp: datetime.datetime, score: float) -> str:
        """
        生成异常报警的可视化图表
        sensor_id: 传感器ID
        data: 异常数据及上下文
        timestamp: 异常发生时间
        score: 异常分数
        返回: 图表文件路径
        """
        try:
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制水位数据
            if 'value' in data.columns:
                plt.plot(data.index, data['value'], 'b-', label='水位值')
            
            # 标记异常点
            if timestamp in data.index:
                anomaly_value = data.loc[timestamp, 'value'] if 'value' in data.columns else 0
                plt.scatter([timestamp], [anomaly_value], color='red', s=100, zorder=5, label='异常点')
            
            # 添加预测概率
            if 'predicted_prob' in data.columns:
                ax2 = plt.twinx()
                ax2.plot(data.index, data['predicted_prob'], 'g--', label='异常概率')
                ax2.set_ylabel('异常概率')
                ax2.set_ylim(0, 1)
            
            # 设置图表标题和标签
            plt.title(f'传感器 {sensor_id} 异常报警 (分数: {score:.4f})')
            plt.xlabel('时间')
            plt.ylabel('水位值')
            plt.grid(True)
            
            # 自动调整日期格式
            plt.gcf().autofmt_xdate()
            
            # 添加图例
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            if 'predicted_prob' in data.columns:
                lines2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
            else:
                plt.legend(loc='best')
            
            # 保存图表
            timestamp_str = timestamp.strftime('%Y%m%d%H%M%S')
            chart_filename = f"{sensor_id}_{timestamp_str}.png"
            chart_path = os.path.join(self.charts_dir, chart_filename)
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return chart_path
        except Exception as e:
            print(f"生成图表失败: {e}")
            return ""
    
    def _send_alert_email(self, sensor_id: str, score: float, timestamp: datetime.datetime, 
                          alert_level: str, data: pd.Series, chart_path: str) -> None:
        """
        发送报警邮件
        sensor_id: 传感器ID
        score: 异常分数
        timestamp: 异常发生时间
        alert_level: 报警级别
        data: 异常数据
        chart_path: 图表文件路径
        """
        try:
            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f'【{alert_level}级报警】传感器 {sensor_id} 异常报警'
            
            # 邮件正文
            email_body = f"""
            <html>
            <body>
                <h2>水利监测系统异常报警</h2>
                <p><strong>报警时间:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>传感器ID:</strong> {sensor_id}</p>
                <p><strong>异常分数:</strong> {score:.4f}</p>
                <p><strong>报警级别:</strong> {alert_level}</p>
                <p><strong>异常数据:</strong></p>
                <pre>{data.to_string() if hasattr(data, 'to_string') else str(data)}</pre>
                <p><strong>异常图表:</strong></p>
                <img src="cid:alert_chart" width="800">
                <p>请及时处理此异常情况!</p>
            </body>
            </html>
            """
            
            # 添加HTML内容
            msg.attach(MIMEText(email_body, 'html'))
            
            # 添加图表附件
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-ID', '<alert_chart>')
                    image.add_header('Content-Disposition', 'inline', filename=os.path.basename(chart_path))
                    msg.attach(image)
            
            # 连接SMTP服务器并发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)
            
            print(f"报警邮件已发送至 {self.email_to}")
        except Exception as e:
            print(f"发送报警邮件失败: {e}")
    
    def process_results(self, results: Dict[str, Any], sensor_prefix: str = 'WL') -> None:
        """
        处理异常检测结果并触发报警
        results: 异常检测结果DataFrame
        sensor_prefix: 传感器ID前缀
        """
        # 获取超过阈值的异常点
        alerts = results[results['predicted_prob'] > self.threshold]
        
        # 对每个异常点触发报警
        for idx, row in alerts.iterrows():
            # 使用row['date']获取日期时间对象，而不是使用索引
            date_obj = row['date'] if 'date' in row else idx
            if isinstance(date_obj, (int, float)):
                # 如果date_obj是数字，则使用当前时间
                date_obj = datetime.datetime.now()
            sensor_id = f"{sensor_prefix}_{date_obj.strftime('%Y%m%d%H')}"
            # 获取异常点前后的数据作为上下文（如果可用）
            context_data = self._get_context_data(results, idx)
            self.log_alert(sensor_id, row['predicted_prob'], date_obj, context_data)
        
        print(f"共触发 {len(alerts)} 个报警")
        
        # 检测缓变异常
        slow_rise_alerts = results[results['slow_rise'] == 1]
        if not slow_rise_alerts.empty:
            print(f"检测到 {len(slow_rise_alerts)} 个缓变异常")
            for idx, row in slow_rise_alerts.iterrows():
                # 使用row['date']获取日期时间对象，而不是使用索引
                date_obj = row['date'] if 'date' in row else idx
                if isinstance(date_obj, (int, float)):
                    # 如果date_obj是数字，则使用当前时间
                    date_obj = datetime.datetime.now()
                sensor_id = f"{sensor_prefix}_SLOW_{date_obj.strftime('%Y%m%d%H')}"
                context_data = self._get_context_data(results, idx)
                self.log_alert(sensor_id, 0.99, date_obj, context_data)  # 缓变异常给予高优先级
