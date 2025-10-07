# 实时血氧SpO2监测脚本
## 简介
ppg_spo2_lsl.py通过LSL协议实时接收Xmuse S的PPG传感器信号，经过处理后计算并显示血氧饱和度(SpO2)的平均值，exp1.csv为样例数据

## 运行指南
1. **安装依赖：**
    打开终端，运行命令：pip install pylsl numpy scipy
2. **连接设备：**
    确保Xmuse S开机并且已经蓝牙连接Xmuse Direct软件，若连接成功，运行脚本后可在终端中看到“已连接到PPG数据流”
3. **运行脚本：**
    在终端启动脚本：python ppg_spo2_lsl.py
    运行后，程序会自动查找PPG数据流并开始输出64个数据点（设备采样率为64Hz，即为1s）的平均SpO2值
4. **参数调整：**
    可以根据实际需求修改脚本代码中的关键参数：SPO2_WINDOW_SIZE（控制输出结果的平滑程度）；不同设备传感器硬件差异可能会导致血氧值存在±1~2%的波动，是正常现象


# 离线PPG数据血氧(SpO2)计算脚本
## 简介
ppg_spo2_csv.py用于处理离线的PPG数据文件（.csv），通过信号滤波和AC/DC分量分析，计算并输出整个数据文件的平均血氧饱和度（SpO2）

## 运行指南
1.  **文件准备：**
    PPG数据CSV文件放置在与脚本相同的目录下 *注意：csv文件必须包含以下列：`timestamps`, `ppg_1`, `ppg_2`, `ppg_3`*
2. **安装依赖：**
    确保已在终端安装所需库：pip install pandas numpy scipy
3. **运行脚本：**
    在终端中运行脚本：python your_script_name.py，脚本将自动读取并处理数据，最后输出计算出的`R`值和`SpO2`值。
4. **修改参数：**
    如果需要，可以修改 `csv_filename = 'exp1.csv'` 来处理不同的文件