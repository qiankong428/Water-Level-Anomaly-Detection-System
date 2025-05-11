#报警模块
class AlertManager:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def check_anomaly(self, scores: np.ndarray) -> List[bool]:
        """根据阈值判断是否触发报警"""
        return scores > self.threshold

    def log_alert(self, sensor_id: str, score: float) -> None:
        """报警记录保存"""
    def process_results():
        return result