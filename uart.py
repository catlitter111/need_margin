import serial
import time
# 定义一个字典来存储所有的命令
commands = {

    "a":"#000P0500T1500!",
    "b":"#000P1500T1500!",
    "c1":"#000PRAD!",



}
# 串口配置参数
UART_DEVICE = "/dev/ttyS9"  # UART0 设备节点（可能是 /dev/ttyS2）
BAUD_RATE = 115200  # 波特率
TIMEOUT = 0.1  # 读取超时（秒)


def init():
    try:
        # 创建串口对象
        ser = serial.Serial(
            port=UART_DEVICE,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,  # 8位数据位
            parity=serial.PARITY_NONE,  # 无校验
            stopbits=serial.STOPBITS_ONE,  # 1位停止位
            timeout=TIMEOUT,  # 读取超时
            xonxoff=False,  # 禁用软件流控
            rtscts=False  # 禁用硬件流控
        )
        if not ser.is_open:
            ser.open()
        return ser
    except Exception as e:
        print(f"无法打开串口: {e}")
        return None


def send(ser, data):
    try:
        if isinstance(data, str):
            data = data.encode('utf-8')
        written = ser.write(data)
        print(f"已发送 {written} 字节: {data}")
        return written
    except Exception as e:
        print(f"发送失败: {e}")
        return 0


def receive(ser):
    try:
        while True:
            data = ser.read(256)
            if data:
                # 假设数据是字符串格式类似 "#000P1000!\r\n"
                data_str = data.decode('utf-8').strip()
                # print(f"接收到数据: {data_str}")
                
                # 提取舵机编号和角度
                if data_str.startswith("#") and data_str.endswith("!"):
                    data_content = data_str.strip("#!").strip().strip("\r\n")
                    parts = data_content.split('P')
                    if len(parts) >= 2:
                        servo_id = parts[0]
                        angle = int(parts[1].split('!')[0])  # 以防有其他字符
                        # 打印结果
                        print(f"舵机编号: {servo_id}, 角度: {angle}")
                else:
                    print("数据格式不正确")
                break
            else:
                print("等待接收数据...")
                time.sleep(1)
    except Exception as e:
        print(f"接收失败: {e}")
        return None


def main():
    # 初始化串口
    uart = init()
    if not uart:
        print("串口初始化失败")
        return
    else:
        print("串口初始化成功")

    try:
        while True:
            # 等待用户输入
            user_input = input("请输入指令 (s发送数据, q退出): ")
            if user_input.lower() == 'z':
                # 发送测试数据
                send(uart, commands["a"])
                #receive(uart)
            elif user_input.lower() == 'x':
                send(uart, commands["b"])
                # receive(uart)
            elif user_input.lower() == 'c':
                send(uart, commands["c1"])
                receive(uart)
            elif user_input.lower() == 'v':
                send(uart, commands["rt_catch3"])
            elif user_input.lower() == 'b':
                send(uart, commands["rt_catch4"])
            elif user_input.lower() == 'r':
                send(uart, commands["read"])
                receive(uart)
            elif user_input.lower() == 'g':
                send(uart, commands["DST"])
            elif user_input.lower() == 'q':
                print("退出程序")
                break
            else:
                print("无效指令，请重新输入")

            # 接收数据
            # receive(uart)

    except KeyboardInterrupt:
        print("\n用户中断操作")
    finally:
        if uart and uart.is_open:
            uart.close()
            print("串口已关闭")


if __name__ == "__main__":
    main()