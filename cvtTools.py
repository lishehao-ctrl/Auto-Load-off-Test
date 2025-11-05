import re
import math
import numpy as np  

class CvtTools:

    @staticmethod
    def parse_general_val(input: str, default_unit: str=None) -> float|int:
        """
        功能：解析数值+前缀，返回实际数值
        参数: input 输入字符串, default_unit 默认单位前缀
        边界: 空值或非法返回0, 未知前缀倍率=1
        """
        input = input.replace(" ", "")
        input_match = re.search(r"([+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?)([A-Za-zµ]?)", input)
        input_val = input_match.group(1)
        input_unit = input_match.group(2)

        try:
            input_val = int(input_val) if input_val.isdigit() else float(input_val)
        except:
            return 0.0
        
        if not input_unit: 
            val = input_val * CvtTools.convert_general_unit(default_unit)
            # 若结果是整数，返回 int，避免后续将其作为“数量”使用时报 float 不能当作 int
            try:
                return int(val) if float(val).is_integer() else val
            except Exception:
                return val
        prefix = input_unit[0]

        if prefix in ('G', 'g'):       scale = 1e9
        elif prefix == 'M':            scale = 1e6
        elif prefix in ('k', 'K'):     scale = 1e3
        elif prefix == 'm':            scale = 1e-3
        elif prefix in ('u', 'µ', 'μ'):scale = 1e-6
        elif prefix in ('n', 'N'):     scale = 1e-9
        elif prefix in ('p', 'P'):     scale = 1e-12
        else:                          scale = 1
        
        val = input_val * scale
        try:
            return int(val) if float(val).is_integer() else val
        except Exception:
            return val
    
    @staticmethod
    def convert_general_unit(unit: str) -> float|int:
        """
        功能: 仅解析前缀倍率, 空值或未知前缀返回1
        """
        if not unit: return 1
        input_match = re.search(r"([+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?)([A-Za-zµ]?)", unit)
        input_unit = input_match.group(2)

        if not input_unit: return 1
        prefix = input_unit[0]

        if prefix in ('G', 'g'):       scale = 1e9
        elif prefix == 'M':            scale = 1e6
        elif prefix in ('k', 'K'):     scale = 1e3
        elif prefix == 'm':            scale = 1e-3
        elif prefix in ('u', 'µ', 'μ'):scale = 1e-6
        elif prefix in ('n', 'N'):     scale = 1e-9
        elif prefix in ('p', 'P'):     scale = 1e-12
        else:                          scale = 1

        return scale
    
    @staticmethod
    def parse_to_hz(freq: str, default_unit: str = "") -> float:
        """
        功能: 解析频率，支持默认单位
        """
        new_freq = CvtTools.parse_general_val(input=freq, default_unit=default_unit)
        
        return new_freq if new_freq else 0
    
    @staticmethod
    def parse_to_Vpp(vpp: str) -> float:
        """
        功能: 解析电压, 统一转为Vpp
        支持: Vpp=1倍, Vpk=2倍,  Vrms=√8倍, m前缀 * 0.001
        """
        vpp = vpp.replace(" ", "")
        vpp_macth = re.search(r"([+-]?\d*(?:\.\d+)?(?:[eE][+-]?\d+)?)([A-Za-zµ]*)", vpp)
        vpp_val = vpp_macth.group(1)
        vpp_unit = vpp_macth.group(2)

        if not vpp_val: return ""
        vpp_val = float(vpp_val)

        if not vpp_unit:              scale = 1
        elif "Vpp".lower() in vpp_unit.lower():   scale = 1
        elif "Vpk".lower() in vpp_unit.lower():   scale = 2 
        elif "Vrms".lower() in vpp_unit.lower():  scale = math.sqrt(8) 
        else:                                     scale = 1 

        if vpp_unit and vpp_unit[0] == "m": scale *= 0.001
        return vpp_val * scale
    
    @staticmethod
    def parse_to_V(volts: str):
        """
        功能: 通用电压解析
        """
        return CvtTools.parse_general_val(input=volts)
    
    @staticmethod
    def _parabolic_interp_delta(m1, m0, p1):
        """
        功能: log|X|三点抛物线插值，估计峰值偏移
        """
        eps = 1e-30
        m1 = np.log(max(m1, eps))
        m0 = np.log(max(m0, eps))
        p1 = np.log(max(p1, eps))
        denom = (m1 - 2*m0 + p1)
        if abs(denom) < 1e-12: return 0.0
        return 0.5 * (m1 - p1) / denom

    @staticmethod
    def _complex_tone_at(times, volts_ac, f_hz, window=None):
        """
        功能: 在f_hz处计算单点DFT, 返回复数
        """
        if window is None:
            return np.sum(volts_ac * np.exp(-1j * 2*np.pi * f_hz * times))
        return np.sum(window * volts_ac * np.exp(-1j * 2*np.pi * f_hz * times))