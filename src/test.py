from typing import Callable
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import queue
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import time
from channel import AWG_Channel, OSC_Channel
from cvtTools import CvtTools
from mapping import Mapping

class testBase:
    """测试基类：提供数据队列、停止事件和结果存储"""

    device_num: int  

    def __init__(self):
        """初始化测试基类: 创建数据队列、停止事件和空的结果字典"""
        self.data_queue = queue.Queue()      
        self.stop_event = threading.Event()   
        self.results = {}                     

    def connection_check(self):
        """检查设备连接, 子类需重写"""
        pass
  
class TestLoadOff(testBase):
    """离线加载测试类: 管理AWG与OSC设备, 存储频率与幅相数据"""

    # 设备数量
    device_num = 2

    def __init__(self, freq_unit: tk.StringVar):
        """初始化: 创建AWG与OSC通道, 设置控制变量与初始数据结构"""
        super().__init__()

        # 通道对象
        self.awg: AWG_Channel
        self.osc_test: OSC_Channel
        self.osc_ref: OSC_Channel
        self.osc_trig: OSC_Channel

        # 频率单位
        self.freq_unit = freq_unit

        # 自动量程控制
        self.is_auto_osc_range = tk.BooleanVar()

        # 校准模式与开关, 带回调触发
        self.var_correct_mode = tk.StringVar(value="")
        self.var_correct_mode.trace_add("write", self.trace_trig_chan_index)

        self.is_correct_enabled = tk.BooleanVar()
        self.is_correct_enabled.trace_add("write", self.refresh_plot)
        self.is_correct_enabled.set(False)

        # 参考文件路径与插值句柄
        self.ref_file_save_path: str = None
        self.href_at: Callable = None

        # 触发模式
        self.trig_mode = tk.StringVar(value="")

        # 图表模式与显示选项, 带回调刷新
        self.var_mag_or_phase = tk.StringVar(value="")
        self.var_mag_or_phase.trace_add("write", self.refresh_plot)
        self.var_mag_or_phase.set(Mapping.label_for_mag)

        self.figure_mode = tk.StringVar(value="")
        self.figure_mode.trace_add("write", self.show_plot)
        self.figure_mode.set(Mapping.label_for_figure_gain_freq)

        # 自动复位开关
        self.auto_reset = tk.BooleanVar()      

        # 数据结果存储
        self.results = {
            Mapping.mapping_freq            : np.array([]),
            Mapping.mapping_gain_raw        : np.array([]),
            Mapping.mapping_gain_db_raw     : np.array([]),
            Mapping.mapping_phase_deg       : np.array([]),
            Mapping.mapping_gain_corr       : np.array([]),
            Mapping.mapping_gain_db_corr    : np.array([]),
            Mapping.mapping_phase_deg_corr  : np.array([]),
            Mapping.mapping_gain_complex    : np.array([]),
        }


    def trace_trig_chan_index(self, *args):
        """进入“双通道校准”模式时，让 osc_trig.chan_index 与 osc_ref.chan_index 双向联动"""
        if self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct:

            def trig_on_ref(*args):
                # ref → trig：当 ref 的通道号变化时，把 trig 同步为 ref 的值
                # 如果两边已经相同，直接返回，避免无意义 set 与回环触发
                if self.osc_ref.chan_index.get() == self.osc_trig.chan_index.get():
                    return
                try:
                    self.osc_trig.chan_index.set(self.osc_ref.chan_index.get())
                except:
                    pass

            def ref_on_trig(*args):
                # trig → ref：当 trig 的通道号变化时，把 ref 同步为 trig 的值
                # 同样先比较，避免来回 set 造成循环触发
                if self.osc_trig.chan_index.get() == self.osc_ref.chan_index.get():
                    return
                try:
                    self.osc_ref.chan_index.set(self.osc_trig.chan_index.get())
                except:
                    pass

            # 初次进入模式时，先做一次单向对齐：以 ref 的通道号为准，设置到 trig
            self.osc_trig.chan_index.set(self.osc_ref.chan_index.get())

            # 绑定写入监听。注意：trace_add 返回的回调名（cbname）需要在解除时原样传回
            self.ref_on_trig_id = self.osc_trig.chan_index.trace_add("write", ref_on_trig)
            self.trig_on_ref_id = self.osc_ref.chan_index.trace_add("write", trig_on_ref)

        else:
            # 非“双通道校准”模式：移除上面绑定的两个写入监听，解除联动
            try:
                self.osc_trig.chan_index.trace_remove("write", self.ref_on_trig_id)
                self.osc_ref.chan_index.trace_remove("write", self.trig_on_ref_id)
            except:
                pass

               
    def start_swep_test(self):

        def auto_osc_range_modifier(osc: OSC_Channel, volts: np.ndarray, force_auto: bool = False) -> bool:
            """
            说明：根据实时波形电压电平，自动微调示波器的量程 (range) 与偏置 (yoffset)。
            返回 True 表示已经做过一次自动调整 (调用方可选择重测一次); False 表示无需调整。
            """
            if not (self.is_auto_osc_range.get() or force_auto):
                return False
            if volts is None or len(volts) == 0:
                return False

            # 基本统计量：峰峰值、中心电平
            vmax = float(np.max(volts))
            vmin = float(np.min(volts))            
            vpp  = vmax - vmin
            mid  = (vmax + vmin) / 2.0

            # 当前量程（纵轴全幅）
            rng_cur, yofs_cur = osc.get_y()

            # 阈值参数（经验系数）
            HI_s, LO_s, TARGET_s = 0.8, 0.6, 0.7  # 期望把 vpp/量程 控制在 ~0.7 左右

            ratio = vpp / rng_cur
            # 偏置相对量程一半的比例（离目标偏置 TARGET_o 的偏差）
            yofs_ratio = abs(mid - yofs_cur) / (rng_cur / 2.0)

            # 容差：对 set_y() 与 get_y() 的读回进行容差判断，避免反复抖动
            RTOL_RANGE = 1e-2
            ATOL_RANGE = 1e-3
            RTOL_OFS   = 1e-2
            ATOL_OFS   = 1e-3  

            upper = yofs_cur + rng_cur * 0.95 / 2.0
            lower = yofs_cur - rng_cur * 0.95 / 2.0

            wave_find  = (lower < vmax < upper) or (upper > vmin > lower)   # 部分在量程内

            # 优先做偏置居中：DC 耦合且未触边但偏离较大时，尝试把波形中心移到目标偏置附近
            if wave_find and yofs_ratio > 0.2 and osc.coupling.get() == Mapping.mapping_coup_dc:
                yofs_needed =  mid
                osc.yoffset.set(str(yofs_needed))
                osc.set_y()

                # 读回确认偏置是否达标，达标则锁定读回数值
                _, yofs_read = osc.get_y()
                if np.isclose(float(yofs_read), float(yofs_needed), rtol=RTOL_OFS, atol=ATOL_OFS):
                    osc.yoffset.set(str(yofs_read))
                    self.try_re_center = False
                    return True
                else:
                    # 未达标：仍把读回值写回 UI 变量，并允许再尝试一次（防止设备分辨率/限位导致的反复）
                    osc.yoffset.set(str(yofs_read))
                    if not self.try_re_center:
                        self.try_re_center = True
                        return True
                    else:
                        if not self.warning_reach_ofs_lim_shown:
                            messagebox.showwarning(
                                Mapping.title_alert,
                                f"{osc.chan_index.get()}号示波器通道超出偏移限制\n自动量程已设置: {yofs_read} V"
                            )
                            self.warning_reach_ofs_lim_shown = True

            if not wave_find:
                rng_needed = rng_cur * 3.0
                osc.range.set(str(rng_needed))
                osc.set_y()

                rng_read, _ = osc.get_y()

                # 达标则锁定读回值；否则允许再试一次，失败才告警
                if np.isclose(float(rng_read), float(rng_needed), rtol=RTOL_RANGE, atol=ATOL_RANGE):
                    osc.range.set(str(rng_read))
                    self.try_get_target = False
                    return True
                else:
                    osc.range.set(str(rng_read))
                    if not self.try_get_target:
                        self.try_get_target = True
                        return True
                    else:
                        if not self.warning_lost_target_shown:
                            messagebox.showwarning(
                                Mapping.title_alert,
                                f"{osc.chan_index.get()}号示波器通道无法找到波形\n自动量程已设置: {rng_read} V"
                            )
                            self.warning_lost_target_shown = True

            # 没有触边时，按 vpp/量程 的比例做“精调”，目标 ~0.7
            if (ratio > HI_s) or (ratio < LO_s):
                rng_needed = vpp / TARGET_s
                osc.range.set(str(rng_needed))
                osc.set_y()

                rng_read, _ = osc.get_y()
                if np.isclose(float(rng_read), float(rng_needed), rtol=RTOL_RANGE, atol=ATOL_RANGE):
                    osc.range.set(str(rng_read))
                    self.try_set_res = False
                    return True
                else:
                    osc.range.set(str(rng_read))
                    if not self.try_set_res:
                        self.try_set_res = True
                        return True
                    else:
                        if not self.warning_lack_res_shown:
                            messagebox.showwarning(
                                Mapping.title_alert,
                                f"{osc.chan_index.get()}号示波器通道超出量程限制\n自动量程已设置: {rng_read} V"
                            )
                            self.warning_lack_res_shown = True

            return False
        
        def calc_vin_peak(vpp_panel, awg_imp, osc_imp):
            """说明：根据 AWG 面板幅度 (Vpp) 与阻抗匹配, 估算待测端实际 Vpeak。"""
            RS = 50.0
            RL = 50.0 if osc_imp == Mapping.mapping_imp_r50 else 1e6

            # 若 AWG 输出端设置为 50Ω，面板 Vpp 是“端接 50Ω 时”的数值，等效开路电压加倍；
            # 若 AWG 为“高阻/Hi-Z”标定，面板值就是开路电压。
            if awg_imp == Mapping.mapping_imp_r50: 
                voc = 2 * vpp_panel
            else:                             
                voc = vpp_panel

            # 分压到负载的 Vpp
            vload = voc * RL / (RS + RL)

            # Vpp -> Vpeak
            vpeak = 0.5 * vload
            return vpeak
            
        def append_result():
            """说明：把本次频点的测量结果追加到结果数组。"""
            self.results[Mapping.mapping_freq]        = np.append(self.results[Mapping.mapping_freq], freq)
            self.results[Mapping.mapping_gain_raw]    = np.append(self.results[Mapping.mapping_gain_raw], gain_raw) 
            self.results[Mapping.mapping_gain_db_raw] = np.append(self.results[Mapping.mapping_gain_db_raw], gain_db_raw)
            if self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct or self.trig_mode.get() == Mapping.label_for_triggered: 
                self.results[Mapping.mapping_gain_complex] = np.append(self.results[Mapping.mapping_gain_complex], gain_c)
                self.results[Mapping.mapping_phase_deg] = np.append(self.results[Mapping.mapping_phase_deg], phase)

        def initialize_devices():
            """说明：按当前模式初始化各设备的 on/imp/coup/trigger 与 y 轴量程"""
            if self.auto_reset.get(): 
                awg.rst()
                osc_test.rst()

            awg.set_on()
            awg.set_imp()

            osc_test.set_on()
            if self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct:
                osc_ref.set_on()
                scale, offs = osc_ref.get_y()
                osc_ref.range = tk.StringVar(value=str(scale))
                osc_ref.yoffset = tk.StringVar(value=str(offs))

            if self.trig_mode.get() == Mapping.label_for_triggered: 
                osc_trig.set_on()
                scale, offs = osc_trig.get_y()
                osc_trig.range = tk.StringVar(value=str(scale))
                osc_trig.yoffset = tk.StringVar(value=str(offs))
                osc_trig.set_trig_rise()

            # 先设置耦合，再设置阻抗，避免 UI trace 导致的状态不同步
            osc_test.set_coup()
            osc_test.set_imp()
            if self.trig_mode.get() == Mapping.label_for_free_run: 
                osc_test.set_free_run()

        def check_user_input():
            # 说明：校验用户手动输入的量程/偏置/幅度是否在设备允许范围内，
            # 若设备读回与期望不一致，则以读回为准并弹出提示。
            rng_needed = CvtTools.parse_to_V(osc_test.range.get())
            yofs_needed = CvtTools.parse_to_V(osc_test.yoffset.get())
            osc_test.set_y()
            rng_read, yofs_read = osc_test.get_y()

            # 量程校验（达标则锁定读回；否则写回读回值并告警）
            if np.isclose(rng_read, rng_needed, atol=1e-2, rtol=1e-2):
                osc_test.range.set(str(rng_read))
            else:
                osc_test.range.set(str(rng_read))
                messagebox.showwarning(
                    Mapping.title_alert, 
                    f"{osc_test.chan_index.get()}号示波器通道超出量程限制\n{Mapping.label_for_range}: {rng_needed} V\n已改成: {rng_read} V"
                )
                self.warning_lack_res_shown = True

            # 偏置校验（达标则锁定读回；否则写回读回值并告警）
            if np.isclose(yofs_read, yofs_needed, rtol=1e-2, atol=1e-1):
                osc_test.yoffset.set(str(yofs_read))
            else:
                osc_test.yoffset.set(str(yofs_read))
                messagebox.showwarning(
                    Mapping.title_alert,
                    f"{osc_test.chan_index.get()}号示波器通道超出偏移限制\n{Mapping.label_for_yoffset}: {yofs_needed} V\n已改成: {yofs_read} V"
                )
                self.warning_reach_ofs_lim_shown = True

            # AWG 幅度校验（达标则锁定读回；否则写回读回值并告警）
            amp_needed = CvtTools.parse_to_Vpp(awg.amp.get())
            awg.set_amp()
            amp_read = awg.get_amp()

            if np.isclose(amp_read, amp_needed, rtol=1e-3, atol=1e-2):
                awg.amp.set(str(amp_read))
            else:
                awg.amp.set(str(amp_read))
                messagebox.showwarning(
                    Mapping.title_alert,
                    f"{awg.chan_index.get()}号信号发生器通道超出幅度限制\n{Mapping.label_for_set_amp}: {amp_needed} Vpp\n已改成: {amp_read}Vpp"
                )


        # --- 状态标志初始化 ---
        self.warning_lack_res_shown = False
        self.warning_lost_target_shown = False
        self.warning_reach_ofs_lim_shown = False
        self.warning_freq_out_of_range_shown = False
        self.warning_amp_out_of_range_shown = False
        self.try_set_res = False
        self.try_get_target = False
        self.try_re_center = False

        awg: AWG_Channel = self.awg
        osc_test: OSC_Channel = self.osc_test
        osc_trig: OSC_Channel = self.osc_trig
        osc_ref: OSC_Channel = self.osc_ref

        # 频点列表与索引
        freq_points = awg.get_sweep_freq_points()
        freq_index = 0
        # 预设初始频点
        awg.set_freq(freq=freq_points[0])

        initialize_devices()
        check_user_input()

        # 等待设备稳定
        time.sleep(0.5)

        self.results = {
                        Mapping.mapping_freq : np.array([]),
                        Mapping.mapping_gain_raw : np.array([]),
                        Mapping.mapping_gain_db_raw : np.array([]),
                        Mapping.mapping_phase_deg : np.array([]),
                        Mapping.mapping_gain_corr : np.array([]),
                        Mapping.mapping_gain_db_corr : np.array([]),
                        Mapping.mapping_phase_deg_corr : np.array([]),
                        Mapping.mapping_gain_complex : np.array([]),
        }

        self.refresh_plot()

        while freq_index < len(freq_points):
            freq = freq_points[freq_index]

            if self.stop_event.is_set(): 
                break

            if self.warning_freq_out_of_range_shown:
                break

            awg.set_freq(freq=freq)
            freq = awg.get_freq() 

            if not np.isclose(freq, freq_points[freq_index], atol=1e-3, rtol=5e-6):
                self.warning_freq_out_of_range_shown = True
                messagebox.showwarning(
                    Mapping.title_alert,
                    f"频率设置不成功\n目标频率: {freq_points[freq_index]} Hz\n实际频率: {freq} Hz\n程序已停止运行"
                )

                if freq in self.results[Mapping.mapping_freq]:
                    freq_index += 1
                    continue

            new_amp = awg.get_amp()
            if not np.isclose(new_amp, CvtTools.parse_to_Vpp(awg.amp.get()), atol=1e-2, rtol=1e-3):
                messagebox.showwarning(
                    Mapping.title_alert,
                    f"{awg.chan_index.get()}号信号发生器通道超出幅度限制\n{Mapping.label_for_set_amp}: {CvtTools.parse_to_Vpp(awg.amp.get())} Vpp\n已改成: {new_amp} Vpp"
                )
                awg.amp.set(str(awg.get_amp()))
                self.warning_amp_out_of_range_shown = True

            device_sampling_rate = osc_test.get_sample_rate()
            sampling_time = self.cal_sampling_time(
                freq=freq, 
                device_sr=device_sampling_rate,
                points=CvtTools.parse_general_val(osc_test.points.get())
            )

            # ------------------ 无校准分支 ------------------
            if self.var_correct_mode.get() == Mapping.label_for_no_correct:
                # 设置 osc x/y 轴
                osc_test.set_x(xscale=sampling_time)
                osc_test.set_y()

                # 根据触发模式采集波形
                if self.trig_mode.get() == Mapping.label_for_triggered: 
                    osc_test.trig_measure()
                elif self.trig_mode.get() == Mapping.label_for_free_run:
                    osc_test.quick_measure()

                # 读回波形： 如果自动量程调整后，重新测量
                times, volts = osc_test.read_raw_waveform()
                if auto_osc_range_modifier(osc=osc_test, volts=volts): 
                    continue

                # 去直流分量
                volts_ac = volts - np.mean(volts)

                # FFT 计算
                window = np.hanning(len(volts_ac))
                Vfft   = np.fft.rfft(window * volts_ac)
                freqs  = np.fft.rfftfreq(len(volts_ac), times[1]-times[0])
                k0     = np.argmin(np.abs(freqs - freq))
                lo     = max(0, k0 - 2)
                hi     = min(len(Vfft), k0 + 3)

                # 计算输出输入峰值
                Vout_peak = (2/(np.sqrt(np.sum(window**2) * len(volts_ac)))) * np.sqrt(np.sum(abs(Vfft[lo:hi] ** 2)))
                Vin_peak = calc_vin_peak(
                    vpp_panel=CvtTools.parse_to_Vpp(awg.amp.get()), 
                    awg_imp=awg.imp.get(), 
                    osc_imp=osc_test.imp.get()
                ) 

                # 计算增益（线性与对数）
                gain_raw    = Vout_peak / Vin_peak         
                gain_db_raw = 20.0 * np.log10(np.maximum(gain_raw, 1e-12)) 

                # 计算相位（仅在触发模式下计算）
                if self.trig_mode.get() == Mapping.label_for_triggered:
                    mags = np.abs(Vfft)
                    if 1 <= k0 <= mags.size-2:
                        delta = CvtTools._parabolic_interp_delta(mags[k0-1], mags[k0], mags[k0+1])
                    else:
                        delta = 0.0
                    df = freqs[1] - freqs[0]
                    f_hat = freqs[k0] + delta * df

                    # 计算相位时，优先用插值后的频率 f_hat 来计算复数谐波分量
                    X = CvtTools._complex_tone_at(times, volts_ac, f_hat, window)
                    if np.abs(X) < 1e-15:  
                        ang = np.angle(np.sum(Vfft[lo:hi]))
                    else:
                        ang = np.angle(X)

                    # 计算复数增益与相位角
                    Vout_phasor = Vout_peak * np.exp(1j * ang)
                    gain_c = Vout_phasor / Vin_peak
                    phase = np.degrees(np.angle(gain_c))

            # ------------------ 单通道校准分支 ------------------
            elif self.var_correct_mode.get() == Mapping.label_for_single_chan_correct:
                # 设置 osc x/y 轴
                osc_test.set_x(xscale=sampling_time)
                osc_test.set_y()

                # 根据触发模式采集波形
                if self.trig_mode.get() == Mapping.label_for_triggered: 
                    osc_test.trig_measure()
                elif self.trig_mode.get() == Mapping.label_for_free_run:
                    osc_test.quick_measure()

                # 读回波形： 如果自动量程调整后，重新测量
                times, volts = osc_test.read_raw_waveform()
                if auto_osc_range_modifier(osc=osc_test, volts=volts): 
                    continue  

                # 去直流分量
                volts_ac = volts - np.mean(volts)

                # FFT 计算
                window = np.hanning(len(volts_ac))
                Vfft   = np.fft.rfft(window * volts_ac)
                freqs  = np.fft.rfftfreq(len(volts_ac), times[1]-times[0])
                k0     = np.argmin(np.abs(freqs - freq))
                lo     = max(0, k0 - 2)
                hi     = min(len(Vfft), k0 + 3)

                # 计算输出输入峰值
                Vout_peak = (2/(np.sqrt(np.sum(window**2) * len(volts_ac)))) * np.sqrt(np.sum(abs(Vfft[lo:hi] ** 2)))
                Vin_peak = calc_vin_peak(
                    vpp_panel=CvtTools.parse_to_Vpp(awg.amp.get()), 
                    awg_imp=awg.imp.get(), 
                    osc_imp=osc_test.imp.get()
                ) 

                # 计算增益（线性与对数）
                gain_raw    = Vout_peak / Vin_peak         
                gain_db_raw = 20.0 * np.log10(np.maximum(gain_raw, 1e-12))

                # 计算相位（仅在触发模式下计算）
                if self.trig_mode.get() == Mapping.label_for_triggered:
                    mags = np.abs(Vfft)
                    if 1 <= k0 <= mags.size-2:
                        delta = CvtTools._parabolic_interp_delta(mags[k0-1], mags[k0], mags[k0+1])
                    else:
                        delta = 0.0
                    df = freqs[1] - freqs[0]
                    f_hat = freqs[k0] + delta * df

                    # 计算相位时，优先用插值后的频率 f_hat 来计算复数谐波分量
                    X = CvtTools._complex_tone_at(times, volts_ac, f_hat, window)
                    if np.abs(X) < 1e-15:
                        ang = np.angle(np.sum(Vfft[lo:hi]))
                    else:
                        ang = np.angle(X)

                    # 计算复数增益与相位角
                    Vout_phasor = Vout_peak * np.exp(1j * ang)
                    gain_c = Vout_phasor / Vin_peak
                    phase = np.degrees(np.angle(gain_c))

            # ------------------ 双通道校准分支 ------------------
            elif self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct:
                # 设置 osc x/y 轴
                osc_test.set_x(xscale=sampling_time)
                osc_test.set_y()
                osc_ref.set_x(xscale=sampling_time)
                osc_ref.set_y()

                # 根据触发模式采集波形
                if self.trig_mode.get() == Mapping.label_for_triggered: 
                    osc_test.trig_measure() 
                elif self.trig_mode.get() == Mapping.label_for_free_run:
                    osc_test.quick_measure()

                # 读回波形：如果自动量程调整后，重新测量
                times_t, volts_t = osc_test.read_raw_waveform()
                if auto_osc_range_modifier(osc=osc_test, volts=volts_t): 
                    continue
                times_r, volts_r = osc_ref.read_raw_waveform()
                if auto_osc_range_modifier(osc=osc_ref, volts=volts_r, force_auto=True): 
                    continue

                # 去直流分量
                volts_ac_t = volts_t - np.mean(volts_t)
                volts_ac_r = volts_r - np.mean(volts_r)

                # FFT 计算
                window_t = np.hanning(len(volts_ac_t))
                window_r = np.hanning(len(volts_ac_r))

                Vfft_t = np.fft.rfft(window_t * volts_ac_t)
                Vfft_r = np.fft.rfft(window_r * volts_ac_r)

                freqs_t = np.fft.rfftfreq(len(volts_ac_t), times_t[1] - times_t[0])
                freqs_r = np.fft.rfftfreq(len(volts_ac_r), times_r[1] - times_r[0])

                k0_t = np.argmin(np.abs(freqs_t - freq))
                k0_r = np.argmin(np.abs(freqs_r - freq))

                lo_t = max(0, k0_t - 2)
                hi_t = min(len(Vfft_t), k0_t + 3)
                lo_r = max(0, k0_r - 2)
                hi_r = min(len(Vfft_r), k0_r + 3)

                mag_t = (2 / (np.sqrt(np.sum(window_t**2) * len(volts_ac_t)))) * np.sqrt(np.sum(abs(Vfft_t[lo_t:hi_t] ** 2)))
                mag_r = (2 / (np.sqrt(np.sum(window_r**2) * len(volts_ac_r)))) * np.sqrt(np.sum(abs(Vfft_r[lo_r:hi_r] ** 2)))

                Ph_t = np.angle(np.sum(Vfft_t[lo_t:hi_t]))
                Ph_r = np.angle(np.sum(Vfft_r[lo_r:hi_r]))

                mag_ratio = max(mag_t / max(mag_r, 1e-15), 1e-15)  
                dphi = Ph_t - Ph_r

                # 计算复数增益与相位角
                gain_c = mag_ratio * (np.cos(dphi) + 1j*np.sin(dphi))
                gain_raw    = np.abs(gain_c)
                gain_db_raw = 20.0 * np.log10(np.maximum(gain_raw, 1e-12))  

                mags_r = np.abs(Vfft_r)
                if 1 <= k0_r <= mags_r.size-2:
                    delta_r = CvtTools._parabolic_interp_delta(mags_r[k0_r-1], mags_r[k0_r], mags_r[k0_r+1])
                else:
                    delta_r = 0.0
                df_r = freqs_r[1] - freqs_r[0]
                f_hat = freqs_r[k0_r] + delta_r * df_r

                # 计算相位时，优先用插值后的频率 f_hat 来计算复数谐波分量
                Xt = CvtTools._complex_tone_at(times_t, volts_ac_t, f_hat, window_t)
                Xr = CvtTools._complex_tone_at(times_r, volts_ac_r, f_hat, window_r)
                if (np.abs(Xt) < 1e-15) or (np.abs(Xr) < 1e-15):
                    # 回退方法
                    Ph_t = np.angle(np.sum(Vfft_t[lo_t:hi_t]))
                    Ph_r = np.angle(np.sum(Vfft_r[lo_r:hi_r]))
                    S_t = mag_t * np.exp(1j*Ph_t)
                    S_r = mag_r * np.exp(1j*Ph_r)
                    phase = np.degrees(np.angle(S_t / S_r))
                else:
                    Sxy = Xt * np.conj(Xr)
                    phase = np.degrees(np.angle(Sxy))  

            # 追加结果
            append_result()

            self.data_queue.put(freq)
            freq_index += 1
        
        self.data_queue.put(None)


    def connection_check(self):
        """检查设备连接"""
        # 如果选择了“无校准”模式，只检查 AWG 与被测通道
        if self.var_correct_mode.get() == Mapping.label_for_no_correct:
            self.awg.inst_open()
            self.awg.check_open()
            self.osc_test.inst_open()
            self.osc_test.check_open()
        # 如果选择了“单通道校准”模式，只检查 AWG 与被测通道
        elif self.var_correct_mode.get() == Mapping.label_for_single_chan_correct:
            self.awg.inst_open()
            self.awg.check_open()
            self.osc_test.inst_open()
            self.osc_test.check_open()
        # 如果选择了“双通道校准”模式，检查 AWG、被测通道、参考通道
        elif self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct:
            self.awg.inst_open()
            self.awg.check_open()
            self.osc_test.inst_open()
            self.osc_test.check_open()
            self.osc_ref.inst_open()
            self.osc_ref.check_open()

        # 如果选择了“触发模式”，检查触发通道
        if self.trig_mode.get() == Mapping.label_for_triggered:
            self.osc_trig.inst_open()
            self.osc_trig.check_open()
        
    def cal_sampling_time(self, freq, device_sr, points, *, T_MIN=1e-6, N_CYC=10, PTS_MAX=1e7):
        """说明：计算示波器采样时间基准。返回值单位为秒(s)。"""
        f = max(float(freq), 1e-12)  # 防止除零
        # 三个候选值取最大：保证足够长
        T = max(points/device_sr, N_CYC/f, T_MIN)
        # 如果设备有点数上限，做一个夹紧
        if PTS_MAX:
            T = min(T, PTS_MAX/device_sr)

        return T

    def setup_plots(self, frame: tk.Frame):
        """说明：创建并初始化两个图表的 Figure、Axes、Line2D、Canvas 等对象，并放置在指定的 frame 里。"""

        self.frame_plot = tk.Frame(frame)
        self.frame_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        freq_unit = self.freq_unit.get()

        # === 线性增益图 ===
        self.fig_gain = Figure(figsize=(8, 4))
        try:
            self.fig_gain.set_tight_layout(True)
        except Exception:
            pass
        self.ax_gain = self.fig_gain.add_subplot(111)
        self.ax_gain.set_xlabel(f"{Mapping.label_for_freq}({freq_unit})")
        self.ax_gain.set_ylabel(Mapping.label_for_figure_gain)
        (self.line_gain,) = self.ax_gain.plot([], [], linestyle='-')

        # 线性增益图右轴：相位
        self.ax_gain_right = self.ax_gain.twinx() 
        self.ax_gain_right.set_ylabel(Mapping.label_for_figure_phase) 
        (self.line_phase,) = self.ax_gain_right.plot([], [], linestyle=':', color=Mapping.mapping_color_for_phase_line)

        self.canvas_gain = FigureCanvasTkAgg(self.fig_gain, master=self.frame_plot)
        w = self.canvas_gain.get_tk_widget()
        w.bind("<FocusIn>", lambda e:'break') # 禁止 matplotlib 图表获取焦点，避免与 Tkinter 的键盘快捷键冲突

        # === 对数增益图 ===
        self.fig_db = Figure(figsize=(8, 4))
        try:
            self.fig_db.set_tight_layout(True)
        except Exception:
            pass
        self.ax_db = self.fig_db.add_subplot(111)
        self.ax_db.set_xlabel(f"{Mapping.label_for_freq}({freq_unit})")
        self.ax_db.set_ylabel(Mapping.label_for_figure_gain_db)
        (self.line_db,) = self.ax_db.plot([], [], linestyle='-')

        # 对数增益图右轴：相位
        self.ax_db_right = self.ax_db.twinx() 
        self.ax_db_right.set_ylabel(Mapping.label_for_figure_phase) 
        (self.line_phase_db,) = self.ax_db_right.plot([], [], linestyle=':', color=Mapping.mapping_color_for_phase_line)

        self.canvas_db = FigureCanvasTkAgg(self.fig_db, master=self.frame_plot)
        w = self.canvas_db.get_tk_widget()
        w.bind("<FocusIn>", lambda e:'break') # 禁止 matplotlib 图表获取焦点，避免与 Tkinter 的键盘快捷键冲突

        self.canvas_gain.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 游标交互
        mplcursors.cursor([self.line_gain, self.line_phase, self.line_db, self.line_phase_db], hover=True)

        # 初始显示
        self.refresh_plot()

    def show_plot(self, *args):
        """说明：根据当前 figure_mode 变量，显示对应的图表。"""
        try:
            # 切换显示
            if self.figure_mode.get() == Mapping.label_for_figure_gain_freq:
                self.canvas_db.get_tk_widget().pack_forget()
                self.canvas_gain.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            elif self.figure_mode.get() == Mapping.label_for_figure_gaindb_freq:
                self.canvas_gain.get_tk_widget().pack_forget()
                self.canvas_db.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 刷新图表
            self.refresh_plot()
        except:
            pass

    def refresh_plot(self, *args):
        """说明：根据当前数据与设置，刷新图表内容。"""

        def get_gain_corr():
            """说明：计算校准后的增益与相位数据，存入 results 的对应字段。"""
            # 无校准时，校准数据直接等于原始数据
            if not (self.is_correct_enabled.get() and hasattr(self, "href_at") and self.href_at):
                self.results[Mapping.mapping_gain_corr] = self.results[Mapping.mapping_gain_raw]
                self.results[Mapping.mapping_gain_db_corr] = self.results[Mapping.mapping_gain_db_raw]
                self.results[Mapping.mapping_phase_deg_corr] = self.results[Mapping.mapping_phase_deg]
                return

            href_vals = self.href_at(self.results[Mapping.mapping_freq])
            # 如果是“双通道校准”或“触发模式”，则用复数除法计算校准后的增益与相位；
            # 否则用幅值除法计算校准后的增益。
            if self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct or self.trig_mode.get() == Mapping.label_for_triggered:
                href_vals = np.array(href_vals, dtype=np.complex128)
                H_corr = self.results[Mapping.mapping_gain_complex] / href_vals
                self.results[Mapping.mapping_gain_corr] = np.abs(H_corr)
                self.results[Mapping.mapping_gain_db_corr] = np.round(20 * np.log10(np.abs(H_corr)), 6)
                self.results[Mapping.mapping_phase_deg_corr] = np.round(np.degrees(np.angle(H_corr)), 6)
            else:
                eps = 1e-12
                href_abs = np.maximum(np.abs(href_vals), eps)
                self.results[Mapping.mapping_gain_corr]    = np.round(self.results[Mapping.mapping_gain_raw]/href_abs, 6)
                self.results[Mapping.mapping_gain_db_corr] = np.round(20*np.log10(np.maximum(self.results[Mapping.mapping_gain_corr], eps)), 6)

        def switch_mag_phase():
            """说明：根据当前 figure_mag_or_phase 变量，切换各条曲线的显隐状态。"""
            # 如果是“增益 + 相位”模式，四条曲线都显示；
            if self.var_mag_or_phase.get() == Mapping.label_for_mag_and_phase:
                self.line_gain.set_visible(True)
                self.line_phase.set_visible(True)
                self.line_db.set_visible(True)
                self.line_phase_db.set_visible(True)
            # 如果是“增益”模式，只显示增益曲线；
            elif self.var_mag_or_phase.get() == Mapping.label_for_mag:
                self.line_gain.set_visible(True)
                self.line_phase.set_visible(False)
                self.line_db.set_visible(True)
                self.line_phase_db.set_visible(False)
            # 如果是“相位”模式，只显示相位曲线；
            elif self.var_mag_or_phase.get() == Mapping.label_for_phase:
                self.line_gain.set_visible(False)
                self.line_phase.set_visible(True)
                self.line_db.set_visible(False)
                self.line_phase_db.set_visible(True)

        # 更新各条曲线的数据
        try:
            freq_unit = self.freq_unit.get()
            get_gain_corr()
            switch_mag_phase()

            # 更新线性增益曲线数据，并自动调整坐标轴
            self.line_gain.set_data(
                self.results[Mapping.mapping_freq]/CvtTools.convert_general_unit(freq_unit),
                self.results[Mapping.mapping_gain_corr] if self.is_correct_enabled.get() else self.results[Mapping.mapping_gain_raw]
            )
            self.ax_gain.set_xlabel(f"{Mapping.label_for_freq}({freq_unit})")
            

            # 更新对数增益曲线数据，并自动调整坐标轴
            self.line_db.set_data(
                self.results[Mapping.mapping_freq]/CvtTools.convert_general_unit(freq_unit),
                self.results[Mapping.mapping_gain_db_corr] if self.is_correct_enabled.get() else self.results[Mapping.mapping_gain_db_raw]
            )
            self.ax_db.set_xlabel(f"{Mapping.label_for_freq}({freq_unit})")

            # 如果是“双通道校准”或“触发模式”，并且有相位数据，则更新相位曲线数据，并自动调整坐标轴
            if self.results[Mapping.mapping_phase_deg].size and (self.var_correct_mode.get() == Mapping.label_for_duo_chan_correct or self.trig_mode.get() == Mapping.label_for_triggered):
                self.line_phase.set_data(
                    self.results[Mapping.mapping_freq]/CvtTools.convert_general_unit(freq_unit),
                    self.results[Mapping.mapping_phase_deg_corr] if self.is_correct_enabled.get() else self.results[Mapping.mapping_phase_deg]
                )

                self.line_phase_db.set_data(
                    self.results[Mapping.mapping_freq]/CvtTools.convert_general_unit(freq_unit),
                    self.results[Mapping.mapping_phase_deg_corr] if self.is_correct_enabled.get() else self.results[Mapping.mapping_phase_deg]
                )
            else:
                self.line_phase.set_data([], [])
                self.line_phase_db.set_data([], [])

            self.ax_gain.relim(); self.ax_gain.autoscale_view()
            self.ax_db.relim(); self.ax_db.autoscale_view()
            self.ax_gain_right.relim(); self.ax_gain_right.autoscale_view()
            self.ax_db_right.relim(); self.ax_db_right.autoscale_view()

            # 切换显示
            if self.figure_mode.get() == Mapping.label_for_figure_gain_freq:
                self.canvas_gain.draw_idle()
            elif self.figure_mode.get() == Mapping.label_for_figure_gaindb_freq:
                self.canvas_db.draw_idle()
        except:
            pass
