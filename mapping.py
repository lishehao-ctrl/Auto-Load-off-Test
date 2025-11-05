from __future__ import annotations

class Mapping:
    # ========================= messages =========================
    label_for_input_ui                 = "输入控制界面"
    label_for_file_menu                = "文件"
    label_for_config_menu              = "设备管理"
    label_for_device_configure_window  = "高级设置"
    label_for_exit                     = "退出"

    # ========================= connection =========================
    label_for_auto_lan                 = "连接模式选择"
    label_for_visa_address             = "请输入设备visa地址"
    label_for_ip_address               = "请输入设备ip地址"
    label_for_auto                     = "Auto"
    label_for_lan                      = "LAN"

    # ========================= channel =========================
    label_for_chan_index               = "号通道"
    label_for_test_chan                = "号被测通道"
    label_for_ref_chan                 = "号参考通道"
    label_for_trig_chan                = "号触发通道"

    # ========================= awg/osc setting =========================
    label_for_set_start_frequency      = "起始频率"
    label_for_set_stop_frequency       = "终止频率"
    label_for_set_step_freq            = "步长频率"
    label_for_set_step_num             = "步数数量"
    label_for_set_center_frequency     = "中心频率"
    label_for_set_interval_frequency   = "扫描宽度"
    label_for_log                      = "对数"
    label_for_freq_unit                = "单位"
    label_for_points                   = "最大采样数量"
    label_for_freq                     = "Frequency"
    label_for_set_amp                  = "信号幅度"
    label_for_set_imp                  = "设置输出阻抗"
    label_for_imp_r50                  = "R50"
    label_for_imp_inf                  = "高阻态"
    label_for_coup                     = "耦合方式"
    label_for_yoffset                  = "中心显示电压"
    label_for_range                    = "满幅电压范围"
    label_for_auto_range               = "自动"

    # ========================= correct/reference =========================
    label_for_single_chan_correct      = "单通道"
    label_for_duo_chan_correct         = "双通道"
    label_for_no_correct               = "不校准"
    label_for_set_ref                  = "设为参考"
    label_for_load_ref                 = "读取参考"
    label_for_enable_ref               = "校准使能"

    # ========================= figure =========================
    label_for_figure_gain              = "Gain"
    label_for_figure_gain_db           = "dB"
    label_for_figure_phase             = "Phase (deg)"
    label_for_figure_gain_freq         = f"{label_for_figure_gain}_vs_{label_for_freq}"
    label_for_figure_gaindb_freq       = f"{label_for_figure_gain_db}_vs_{label_for_freq}"

    # ========================= load/save data =========================
    label_for_load_file_to_show        = "读数据文件用作显示"
    label_for_load_file_to_ref         = "读数据文件用作参考"
    label_for_load_config              = "读取配置文件"
    label_for_save_file                = "存文件"
    label_for_save_config              = "存配置"
    label_for_file_is_saved            = "文件已保存"
    label_for_sub_folder_data          = "__data__"

    # ========================= errors & titles =========================
    error_file_not_save                = "数据保存失败！！！"
    error_fail_auto_save               = "自动保存失败！！！"
    title_alert                        = "警告"

    # ========================= mappings / options =========================
    mapping_auto_detect                = "自动识别"
    label_for_device_type_awg          = "任意波形发生器"
    label_for_device_type_osc          = "示波器"

    mapping_DSG_4102                   = "DSG4102"
    mapping_DSG_836                    = "DSG836"
    mapping_MDO_34                     = "MDO34"
    mapping_MDO_3024                   = "MDO3024"
    mapping_DHO_1202                   = "DHO1202"
    mapping_DHO_1204                   = "DHO1204"

    mapping_hz                         = "Hz"
    mapping_khz                        = "KHz"
    mapping_mhz                        = "MHz"
    mapping_ghz                        = "GHz"

    mapping_imp_r50                    = "50"
    mapping_imp_high_z                 = "INF"

    mapping_vpp                        = "Vpp"
    mapping_vpk                        = "Vpk"
    mapping_vrms                       = "Vrms"

    mapping_file_ext_mat               = ".mat"
    mapping_file_ext_csv               = ".csv"
    mapping_file_ext_txt               = ".txt"
    mapping_file_ext_png               = ".png"

    mapping_coup_ac                    = "AC"
    mapping_coup_dc                    = "DC"

    mapping_state_on                   = "ON"
    mapping_state_off                  = "OFF"

    # ========================= combobox values =========================
    values_awg = [
        mapping_DSG_4102,
        mapping_DSG_836,
    ]
    values_osc = [
        mapping_MDO_34,
        mapping_MDO_3024,
        mapping_DHO_1202,
        mapping_DHO_1204,
    ]
    values_device_type = [
        label_for_device_type_awg,
        label_for_device_type_osc,
    ]
    values_freq_unit = [
        mapping_hz,
        mapping_khz,
        mapping_mhz,
        mapping_ghz,
    ]
    values_device_num_list             = [1, 2, 3, 4]
    values_test_load_off_figure = [
        label_for_figure_gain_freq,
        label_for_figure_gaindb_freq,
    ]
    values_correct_modes = [
        label_for_no_correct,
        label_for_single_chan_correct,
        label_for_duo_chan_correct,
    ]
    values_coup = [
        mapping_coup_ac,
        mapping_coup_dc,
    ]

    # ========================= data keys =========================
    mapping_freq                       = "freq"
    mapping_gain_raw                   = "gain_raw"
    mapping_gain_db_raw                = "gain_db_raw"
    mapping_phase_deg                  = "phase"
    mapping_gain_corr                  = "gain_corr"
    mapping_gain_db_corr               = "gain_db_corr"
    mapping_phase_deg_corr             = "phase_corr"
    mapping_gain_complex               = "gain_complex"

    # ========================= trigger modes =========================
    label_for_free_run                 = "free run"
    label_for_triggered                = "triggered"
    values_trig_mode                   = [label_for_free_run, label_for_triggered]

    # ========================= colors =========================
    mapping_color_for_phase_line       = "tab:red"

    # ========================= plot options =========================
    label_for_mag                      = "幅度"
    label_for_phase                    = "相位"
    label_for_mag_and_phase            = "幅度 + 相位"
    values_mag_or_phase                = [label_for_mag, label_for_phase, label_for_mag_and_phase]

    # ========================= defaults (UI/fonts/theme) =========================
    default_data_fn                    = "Test_File"
    default_show_selection_font        = ("Microsoft YaHei", 18)
    default_text_font                  = ("Microsoft YaHei", 10)
    default_terminal_bg                = "black"
    default_terminal_fg                = "white"

    # ========================= defaults (general sweep) =========================
    default_start_freq                 = "1.0"
    default_stop_freq                  = "100.0"
    default_step_freq                  = "1.0"
    default_step_num                   = "100"
    default_is_log_freq_enabled        = mapping_state_off
    default_freq_unit                  = mapping_mhz
    default_samp_pts                   = "10000"

    # ========================= defaults (AWG/OSC settings) =========================
    default_awg_amp                    = "1.0"
    default_awg_imp                    = "50"
    default_yoffset                    = "0.0"
    default_range                      = "1.0"
    default_osc_imp                    = "50"
    default_osc_coup                   = mapping_coup_dc

    # ========================= defaults (modes/switches) =========================
    default_is_auto_range              = mapping_state_on
    default_correct_mode               = label_for_no_correct
    default_is_correct_enabled         = mapping_state_off
    default_trig_mode                  = label_for_free_run
    default_is_auto_save               = mapping_state_on
    default_is_auto_reset              = mapping_state_on

    # ========================= defaults (device names) =========================
    default_awg_name                   = mapping_DSG_4102
    default_osc_name                   = mapping_MDO_34

    # ========================= defaults (channel indices) =========================
    default_awg_chan_index             = "1"
    default_osc_test_chan_index        = "1"
    default_osc_trig_chan_index        = "2"
    default_osc_ref_chan_index         = "2"

    # ========================= defaults (connection) =========================
    default_awg_connect_mode           = label_for_auto
    default_osc_connect_mode           = label_for_auto
    default_awg_visa                   = ""
    default_osc_visa                   = ""
    default_awg_ip                     = "0.0.0.0"
    default_osc_ip                     = "0.0.0.0"
