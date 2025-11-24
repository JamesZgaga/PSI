# import numpy as np
# import yaml

# def read_config():
#     with open('config/config.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# class PSICalculator:
#     def __init__(self):
#         """初始化心理状态指数计算器"""
#         # 读取配置
#         config = read_config()
#         psi_config = config['psi']
        
#         # 设置权重
#         self.negative_emotion_weight = psi_config['negative_emotion_weight']
#         self.pain_weight = psi_config['pain_weight']
        
#         # 获取负面情绪列表
#         self.negative_emotions = psi_config['negative_emotions']
    
#     def calculate_negative_emotion_score(self, emotion_probs):
#         """计算负面情绪分数"""
#         # 提取负面情绪的概率总和
#         negative_score = sum(emotion_probs.get(emotion, 0) for emotion in self.negative_emotions)
#         return negative_score
    
#     def calculate_psi(self, emotion_probs, pain_result):
#         """
#         计算综合心理状态指数(PSI)
        
#         参数:
#             emotion_probs: 情绪概率字典
#             pain_result: 痛苦评估结果
            
#         返回:
#             psi: 综合心理状态指数 (0-1，值越高表示心理状态越糟糕)
#         """
#         # 计算负面情绪分数
#         negative_emotion_score = self.calculate_negative_emotion_score(emotion_probs)
        
#         # 获取痛苦分数
#         pain_score = pain_result['pain_score'] if pain_result else 0
        
#         # 计算加权PSI
#         psi = (self.negative_emotion_weight * negative_emotion_score + 
#                self.pain_weight * pain_score)
        
#         # 确保PSI在0-1范围内
#         psi = max(0, min(1, psi))
        
#         return psi*100

import numpy as np
import yaml
import time
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass

# 定义帧数据结构（存储时间窗口内的有效识别结果）
@dataclass
class EmotionFrame:
    timestamp: float  # 帧时间戳（秒）
    emotion: str      # 识别出的情绪类别（无效识别标记为"invalid"）

class PSICalculator:
    def __init__(self):
        """初始化时间相关的PSI计算器（全面处理极端情况）"""
        # 1. 初始化警告缓存（确保方法调用前属性已存在）
        self.warnings: List[str] = []
        # 2. 加载配置并合并默认值
        self.config = self._load_config()
        # 3. 校验并修正配置参数（处理极端配置）
        self.config = self._validate_and_correct_config(self.config)
        # 4. 初始化核心参数（从校验后的配置中读取）
        self._init_core_params()
        # 5. 状态缓存
        self.emotion_history: List[EmotionFrame] = []  # 时间窗口内的所有帧数据
        self.current_continuous_negative_frames = 0    # 当前连续负性情绪帧数

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件，读取失败时返回内置默认配置"""
        default_config = {
            "w1": 0.4,
            "w2": 0.3,
            "w3": 0.3,
            "positive_emotions": ["Happiness", "Surprise"],
            "negative_emotions": ["Sadness", "Anger", "Fear"],
            "valid_emotions": ["Happiness", "Neutral", "Surprise", "Disgust", "Fear", "Anger", "Sadness"],
            "time_window_seconds": 10.0,
            "frame_interval_seconds": 0.1,
            "max_negative_persistence_seconds": 5.0,
            "level_good_threshold": 30,
            "level_medium_threshold": 60,
            "no_valid_data_psi": 40,
            "no_valid_data_level": "中等",
            "min_time_window": 0.1,
            "min_frame_interval": 0.01,
            "min_max_persistence": 0.1
        }
        
        try:
            with open("config/config.yaml", "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            user_config = full_config.get("psi_time_based", {})
            merged_config = {**default_config, **user_config}
            print("✅ 配置文件加载成功")
            return merged_config
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            warning = f"⚠️  配置文件读取失败：{e}，将使用内置默认配置"
            print(warning)
            self.warnings.append(warning)
            return default_config

    def _validate_and_correct_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """校验并修正配置参数，处理极端配置（如负数、空列表、零值）"""
        corrected_config = config.copy()
        warnings = []

        # --------------------------
        # 1. 权重校验（非负，提示总和建议为1）
        # --------------------------
        weights = [config["w1"], config["w2"], config["w3"]]
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or w < 0:
                corrected_config[f"w{i+1}"] = 0.0
                warnings.append(f"⚠️  权重w{i+1}配置无效（必须非负），已修正为0.0")
        weight_sum = sum([corrected_config["w1"], corrected_config["w2"], corrected_config["w3"]])
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            warnings.append(f"⚠️  权重总和为{weight_sum:.3f}（建议为1.0），可能导致PSI范围异常")

        # --------------------------
        # 2. 情绪类别校验（非空，否则使用默认值）
        # --------------------------
        if not isinstance(config["positive_emotions"], list) or len(config["positive_emotions"]) == 0:
            corrected_config["positive_emotions"] = ["Happiness", "Surprise"]
            warnings.append("⚠️  正性情绪类别配置为空/无效，已修正为默认值：['Happiness', 'Surprise']")
        
        if not isinstance(config["negative_emotions"], list) or len(config["negative_emotions"]) == 0:
            corrected_config["negative_emotions"] = ["Sadness", "Anger", "Fear"]
            warnings.append("⚠️  负性情绪类别配置为空/无效，已修正为默认值：['Sadness', 'Anger', 'Fear']")
        
        if not isinstance(config["valid_emotions"], list) or len(config["valid_emotions"]) == 0:
            corrected_config["valid_emotions"] = ["Happiness", "Neutral", "Surprise", "Disgust", "Fear", "Anger", "Sadness"]
            warnings.append("⚠️  有效情绪类别配置为空/无效，已修正为默认7类情绪")

        # --------------------------
        # 3. 时间参数校验（正数，不小于最小阈值）
        # --------------------------
        time_window = config["time_window_seconds"]
        if not isinstance(time_window, (int, float)) or time_window < config["min_time_window"]:
            corrected_config["time_window_seconds"] = config["min_time_window"]
            warnings.append(f"⚠️  时间窗口配置无效（必须≥{config['min_time_window']}秒），已修正为{config['min_time_window']}秒")
        
        frame_interval = config["frame_interval_seconds"]
        if not isinstance(frame_interval, (int, float)) or frame_interval < config["min_frame_interval"]:
            corrected_config["frame_interval_seconds"] = config["min_frame_interval"]
            warnings.append(f"⚠️  帧间隔配置无效（必须≥{config['min_frame_interval']}秒），已修正为{config['min_frame_interval']}秒")
        
        max_persistence = config["max_negative_persistence_seconds"]
        if not isinstance(max_persistence, (int, float)) or max_persistence < config["min_max_persistence"]:
            corrected_config["max_negative_persistence_seconds"] = config["min_max_persistence"]
            warnings.append(f"⚠️  最大负性持续时间配置无效（必须≥{config['min_max_persistence']}秒），已修正为{config['min_max_persistence']}秒")

        # --------------------------
        # 4. 等级阈值校验（良好≤中等，否则交换）
        # --------------------------
        good_thr = config["level_good_threshold"]
        medium_thr = config["level_medium_threshold"]
        if not isinstance(good_thr, int) or good_thr < 0 or good_thr > 100:
            corrected_config["level_good_threshold"] = 30
            warnings.append("⚠️  良好等级阈值配置无效（必须0-100），已修正为30")
        if not isinstance(medium_thr, int) or medium_thr < 0 or medium_thr > 100:
            corrected_config["level_medium_threshold"] = 60
            warnings.append("⚠️  中等等级阈值配置无效（必须0-100），已修正为60")
        if corrected_config["level_good_threshold"] > corrected_config["level_medium_threshold"]:
            corrected_config["level_good_threshold"], corrected_config["level_medium_threshold"] = corrected_config["level_medium_threshold"], corrected_config["level_good_threshold"]
            warnings.append("⚠️  等级阈值配置不合理（良好>中等），已自动交换")

        # --------------------------
        # 5. 无有效数据默认值校验（PSI在0-100）
        # --------------------------
        no_data_psi = config["no_valid_data_psi"]
        if not isinstance(no_data_psi, (int, float)) or no_data_psi < 0 or no_data_psi > 100:
            corrected_config["no_valid_data_psi"] = 40
            warnings.append("⚠️  无有效数据默认PSI配置无效（必须0-100），已修正为40")

        # 保存警告信息（此时self.warnings已初始化，可安全扩展）
        self.warnings.extend(warnings)
        for warn in warnings:
            print(warn)
        return corrected_config

    def _init_core_params(self) -> None:
        """从校验后的配置中初始化核心参数（无硬编码）"""
        # 权重
        self.w1 = self.config["w1"]
        self.w2 = self.config["w2"]
        self.w3 = self.config["w3"]
        # 情绪类别
        self.positive_emotions = self.config["positive_emotions"]
        self.negative_emotions = self.config["negative_emotions"]
        self.valid_emotions = self.config["valid_emotions"]
        # 时间参数
        self.time_window = self.config["time_window_seconds"]
        self.frame_interval = self.config["frame_interval_seconds"]
        self.max_negative_persistence = self.config["max_negative_persistence_seconds"]
        # 等级阈值
        self.level_good_thr = self.config["level_good_threshold"]
        self.level_medium_thr = self.config["level_medium_threshold"]
        # 无有效数据默认值
        self.no_valid_data_psi = round(self.config["no_valid_data_psi"])
        self.no_valid_data_level = self.config["no_valid_data_level"]

    def _clean_expired_history(self) -> None:
        """清理时间窗口外的历史数据（极端情况：窗口为空时不报错）"""
        if not self.emotion_history:
            return
        current_time = time.time()
        self.emotion_history = [
            frame for frame in self.emotion_history
            if (current_time - frame.timestamp) <= self.time_window
        ]

    def _add_frame(self, current_emotion: Optional[str]) -> None:
        """添加当前帧到历史缓存（处理current_emotion为None/空字符串的极端情况）"""
        # 处理输入为None/空字符串的情况
        if current_emotion is None or not isinstance(current_emotion, str) or current_emotion.strip() == "":
            processed_emotion = "invalid"
            self.warnings.append("⚠️  输入情绪为None/空字符串，视为无效识别")
        else:
            processed_emotion = current_emotion.strip() if current_emotion.strip() in self.valid_emotions else "invalid"
        
        # 添加到历史并清理过期数据
        self.emotion_history.append(
            EmotionFrame(timestamp=time.time(), emotion=processed_emotion)
        )
        self._clean_expired_history()

    def _count_valid_emotion_occurrences(self) -> Dict[str, int]:
        """统计情绪出现次数（极端情况：历史为空时返回全0）"""
        count = {emotion: 0 for emotion in self.valid_emotions}
        count["invalid"] = 0
        if not self.emotion_history:
            return count
        for frame in self.emotion_history:
            if frame.emotion == "invalid":
                count["invalid"] += 1
            else:
                count[frame.emotion] += 1
        return count

    # --------------------------
    # 核心指标计算（处理所有极端输入）
    # --------------------------
    def calculate_positive_ratio(self) -> float:
        """正性情绪比例（极端情况：无有效帧/正性情绪类别为空 → 返回0.0）"""
        emotion_count = self._count_valid_emotion_occurrences()
        total_positive = sum(emotion_count[emo] for emo in self.positive_emotions)
        total_valid = sum(emotion_count[emo] for emo in self.valid_emotions)
        
        if total_valid == 0:
            return 0.0
        return min(1.0, max(0.0, total_positive / total_valid))  # 双重限制范围

    def calculate_emotion_fluctuation_entropy(self) -> float:
        """情绪波动熵（极端情况：无有效帧/仅1类有效情绪 → 返回0.0）"""
        emotion_count = self._count_valid_emotion_occurrences()
        total_valid = sum(emotion_count[emo] for emo in self.valid_emotions)
        
        if total_valid == 0:
            return 0.0
        
        # 计算概率（排除0概率，避免log(0)）
        probabilities = [
            emotion_count[emo] / total_valid 
            for emo in self.valid_emotions 
            if emotion_count[emo] > 0
        ]
        
        # 极端情况：仅1类情绪（熵为0）
        if len(probabilities) <= 1:
            return 0.0
        
        # 计算熵并归一化
        shannon_entropy = -sum(p * np.log(p) for p in probabilities)
        max_entropy = np.log(len(self.valid_emotions))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        return min(1.0, max(0.0, normalized_entropy))

    def calculate_negative_persistence(self, current_emotion: Optional[str]) -> float:
        """负性持续时间（极端情况：当前情绪无效/负性类别为空 → 返回0.0）"""
        # 处理当前情绪为None/空的情况
        processed_emotion = current_emotion.strip() if (current_emotion and isinstance(current_emotion, str)) else "invalid"
        
        # 负性类别为空时，持续时间恒为0
        if not self.negative_emotions:
            return 0.0
        
        # 更新连续负性帧数
        if processed_emotion in self.negative_emotions:
            self.current_continuous_negative_frames += 1
        else:
            self.current_continuous_negative_frames = 0
        
        # 计算持续时间（避免帧间隔为0的极端情况，已在配置校验中处理）
        continuous_time = self.current_continuous_negative_frames * self.frame_interval
        # 归一化（避免最大阈值为0，已在配置校验中处理）
        normalized_persistence = continuous_time / self.max_negative_persistence
        return min(1.0, max(0.0, normalized_persistence))

    # --------------------------
    # 核心PSI计算（处理所有极端情况）
    # --------------------------
    def calculate_psi(self, 
                     current_emotion: Optional[str] = None, 
                     pain_result: Optional[Dict[str, float]] = None) -> Dict[str, Union[int, str, Dict, List]]:
        """
        计算时间相关的PSI指数（全面处理极端情况）
        
        参数:
            current_emotion: 当前帧识别的情绪类别（支持None/空字符串）
            pain_result: 痛苦评估结果（预留扩展接口）
        
        返回:
            包含PSI分数、等级、统计信息、中间指标和警告的完整字典
        """
        # 重置警告缓存（每次计算重新收集）
        self.warnings = []
        
        # 1. 添加当前帧到历史（处理None/空输入）
        self._add_frame(current_emotion)
        
        # 2. 统计有效帧数
        emotion_count = self._count_valid_emotion_occurrences()
        total_valid = sum(emotion_count[emo] for emo in self.valid_emotions)
        total_frames = len(self.emotion_history)
        actual_negative_persistence = self.current_continuous_negative_frames * self.frame_interval

        # 3. 极端情况：无有效数据（返回默认值+警告）
        if total_valid == 0:
            return {
                "psi": self.no_valid_data_psi,
                "psi_level": self.no_valid_data_level,
                "configuration": self._get_config_summary(),
                "time_window_stats": {
                    "窗口内总帧数": total_frames,
                    "有效识别帧数": 0,
                    "无效识别帧数": emotion_count["invalid"],
                    "当前连续负性情绪时间(秒)": 0.0
                },
                "intermediate_metrics": {
                    "正性情绪比例": 0.0,
                    "情绪波动熵": 0.0,
                    "归一化负性情绪持续时间": 0.0,
                    "PSI原始值(未标准化)": round(self.no_valid_data_psi / 100, 4)
                },
                "warnings": self.warnings + ["⚠️  时间窗口内无有效情绪识别数据，返回默认PSI值"]
            }

        # 4. 计算核心指标（正常情况）
        positive_ratio = self.calculate_positive_ratio()
        fluctuation_entropy = self.calculate_emotion_fluctuation_entropy()
        negative_persistence = self.calculate_negative_persistence(current_emotion)
        
        # 5. 计算PSI（处理权重总和异常导致的溢出）
        psi_raw = (
            self.w1 * (1 - positive_ratio)
            + self.w2 * fluctuation_entropy
            + self.w3 * negative_persistence
        )
        # 强制限制原始值在[0,1]（应对权重总和≠1的极端情况）
        psi_raw_clipped = max(0.0, min(1.0, psi_raw))
        psi = round(psi_raw_clipped * 100)

        # 6. 等级划分
        if psi <= self.level_good_thr:
            psi_level = "良好"
        elif self.level_good_thr < psi <= self.level_medium_thr:
            psi_level = "中等"
        else:
            psi_level = "较差"

        # 7. 返回完整结果
        return {
            "psi": psi,
            "psi_level": psi_level,
            "configuration": self._get_config_summary(),
            "time_window_stats": {
                "窗口内总帧数": total_frames,
                "有效识别帧数": total_valid,
                "无效识别帧数": emotion_count["invalid"],
                "当前连续负性情绪时间(秒)": round(actual_negative_persistence, 2)
            },
            "intermediate_metrics": {
                "正性情绪比例": round(positive_ratio, 4),
                "情绪波动熵": round(fluctuation_entropy, 4),
                "归一化负性情绪持续时间": round(negative_persistence, 4),
                "PSI原始值(未标准化)": round(psi_raw, 4)
            },
            "warnings": self.warnings
        }

    def _get_config_summary(self) -> Dict[str, Union[float, tuple, int]]:
        """获取配置摘要（用于返回结果）"""
        return {
            "时间窗口(秒)": self.time_window,
            "帧间隔(秒)": self.frame_interval,
            "权重(w1,w2,w3)": (round(self.w1, 3), round(self.w2, 3), round(self.w3, 3)),
            "最大负性持续时间(秒)": self.max_negative_persistence,
            "等级阈值(良好/中等)": (self.level_good_thr, self.level_medium_thr)
        }

    def reset(self) -> None:
        """重置计算器状态（用于新测试会话）"""
        self.emotion_history.clear()
        self.current_continuous_negative_frames = 0
        self.warnings.clear()
        print("✅ PSI计算器状态已重置")

    def update_weights(self, new_w1: float, new_w2: float, new_w3: float) -> None:
        """更新PSI权重（支持后期机器学习优化，含校验）"""
        # 校验新权重
        new_weights = [new_w1, new_w2, new_w3]
        for i, w in enumerate(new_weights):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"权重w{i+1}必须为非负数")
        # 更新权重
        self.w1 = new_w1
        self.w2 = new_w2
        self.w3 = new_w3
        # 提示权重总和
        weight_sum = sum(new_weights)
        print(f"✅ 权重已更新为：w1={new_w1:.3f}, w2={new_w2:.3f}, w3={new_w3:.3f}（总和：{weight_sum:.3f}）")
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            print(f"⚠️  权重总和非1.0，可能导致PSI范围异常")


# ------------------------------
# 极端情况测试示例
# ------------------------------
if __name__ == "__main__":
    print("="*60)
    print("📊 极端情况测试")
    print("="*60 + "\n")

    # 测试1：初始化后未输入任何帧（无有效数据）
    print("🔍 测试1：未输入任何情绪帧")
    calculator = PSICalculator()
    result1 = calculator.calculate_psi()
    print(f"PSI分数：{result1['psi']} | 等级：{result1['psi_level']}")
    print(f"警告：{result1['warnings']}\n")

    # 测试2：输入空字符串/None（无效识别）
    print("🔍 测试2：输入空字符串和None")
    calculator.reset()
    calculator.calculate_psi(current_emotion="")  # 空字符串
    calculator.calculate_psi(current_emotion=None)  # None
    result2 = calculator.calculate_psi(current_emotion="InvalidEmo")  # 无效情绪
    print(f"PSI分数：{result2['psi']} | 等级：{result2['psi_level']}")
    print(f"有效帧数：{result2['time_window_stats']['有效识别帧数']}")
    print(f"警告：{result2['warnings']}\n")

    # 测试3：权重配置为负数（会自动修正）
    print("🔍 测试3：权重配置为负数（模拟错误配置）")
    # 临时修改配置文件逻辑（模拟错误配置）
    calculator.config["w1"] = -0.2
    calculator.config["w2"] = 1.5
    calculator.config = calculator._validate_and_correct_config(calculator.config)
    calculator._init_core_params()
    result3 = calculator.calculate_psi(current_emotion="Happiness")
    print(f"修正后权重：{result3['configuration']['权重(w1,w2,w3)']}")
    print(f"PSI分数：{result3['psi']} | 等级：{result3['psi_level']}")
    print(f"警告：{result3['warnings']}\n")

    # 测试4：连续负性情绪远超最大阈值
    print("🔍 测试4：连续负性情绪远超最大阈值（5秒）")
    calculator.reset()
    for _ in range(100):  # 100帧 × 0.1秒 = 10秒（远超5秒）
        calculator.calculate_psi(current_emotion="Anger")
    result4 = calculator.calculate_psi(current_emotion="Anger")
    print(f"连续负性时间：{result4['time_window_stats']['当前连续负性情绪时间(秒)']}秒")
    print(f"归一化负性持续时间：{result4['intermediate_metrics']['归一化负性情绪持续时间']}")
    print(f"PSI分数：{result4['psi']} | 等级：{result4['psi_level']}\n")

    # 测试5：权重总和为2.0（模拟错误配置）
    print("🔍 测试5：权重总和为2.0（模拟错误配置）")
    calculator.reset()
    calculator.update_weights(new_w1=1.0, new_w2=0.5, new_w3=0.5)  # 总和2.0
    result5 = calculator.calculate_psi(current_emotion="Sadness")
    print(f"PSI原始值（未裁剪）：{result5['intermediate_metrics']['PSI原始值(未标准化)']}")
    print(f"PSI分数（裁剪后）：{result5['psi']} | 等级：{result5['psi_level']}")
    print(f"警告：{result5['warnings']}")
