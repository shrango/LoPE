import os
import time
import subprocess
import signal
import logging
from datetime import datetime

# ================= 配置区 =================
CHECK_INTERVAL = 30           # 每次检查的间隔时间（秒）
STUCK_TIMEOUT = 600           # 持续卡住多久后触发重启（秒），10分钟 = 600秒
POWER_THRESHOLD = 150         # 功耗阈值（瓦）
VRAM_THRESHOLD = 1000         # 显存占用阈值（MB）
# RESTART_CMD = "bash qwen3-1.7b-base_openr1_data_baseline.sh" 
# RESTART_CMD = "bash qwen3-1.7b-base_openr1_data_ici_wokl.sh"
# RESTART_CMD = "bash qwen2.5_math_7b_openr1_data_baseline.sh"
# RESTART_CMD = "bash qwen2.5_math_7b_openr1_data_ici_wokl.sh"
# RESTART_CMD = "bash qwen3-1.7b-base_openr1_data_fake_sent_24_wokl.sh"
# RESTART_CMD = "bash qwen2.5_math_7b_openr1_data_lorem.sh"
RESTART_CMD = "bash qwen3-4b-base_openr1_data_lorem_wokl_luffy_fixadv.sh"
TARGET_GPU_IDS = [0,1,2,3,4, 5, 6, 7]  # 指定要监控的GPU编号列表，设为 None 则监控所有GPU
# ==========================================

# 配置日志格式，自带当前时间
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def get_gpu_metrics(gpu_ids=None):
    """获取指定GPU的当前状态 (显存, 利用率, 功耗)，gpu_ids 为 None 时获取所有GPU"""
    cmd = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits"
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')
        metrics = []
        for line in output:
            if not line.strip(): continue
            idx, mem, util, power = [float(x.strip()) for x in line.split(',')]
            if gpu_ids is not None and int(idx) not in gpu_ids:
                continue
            metrics.append({'idx': int(idx), 'mem': mem, 'util': util, 'power': power})
        return metrics
    except Exception as e:
        logging.error(f"获取GPU状态失败: {e}")
        return []

def is_gpu_stuck(metric):
    """判断单张GPU是否处于僵死状态"""
    mem_occupied = metric['mem'] > VRAM_THRESHOLD
    low_usage = (metric['util'] == 0) or (metric['power'] < POWER_THRESHOLD)
    return mem_occupied and low_usage

def kill_gpu_processes(gpu_ids=None):
    """杀掉指定GPU上的计算进程，gpu_ids 为 None 时清理所有GPU上的进程"""
    if gpu_ids is not None:
        logging.warning(f"正在清理 GPU {gpu_ids} 上的僵尸进程...")
    else:
        logging.warning("正在清理所有 GPU 上的僵尸进程...")
    cmd = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"
    try:
        gpu_uuid_map = {}
        if gpu_ids is not None:
            uuid_cmd = "nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader"
            uuid_output = subprocess.check_output(uuid_cmd, shell=True).decode('utf-8').strip().split('\n')
            for line in uuid_output:
                if not line.strip(): continue
                idx_str, uuid = [x.strip() for x in line.split(',', 1)]
                if int(idx_str) in gpu_ids:
                    gpu_uuid_map[uuid] = int(idx_str)

        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')
        killed_pids = set()
        for line in output:
            line = line.strip()
            if not line: continue
            uuid, pid = [x.strip() for x in line.split(',', 1)]
            if gpu_ids is not None and uuid not in gpu_uuid_map:
                continue
            if pid and pid not in killed_pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed_pids.add(pid)
                    logging.info(f"已强制结束 PID: {pid}")
                except ProcessLookupError:
                    pass
    except Exception as e:
        logging.error(f"清理进程时发生异常: {e}")

def main():
    stuck_start_time = None
    gpu_ids = TARGET_GPU_IDS
    target_count = len(gpu_ids) if gpu_ids is not None else None
    
    if gpu_ids is not None:
        logging.info(f"GPU 监控服务已启动，监控指定 GPU: {gpu_ids}")
    else:
        logging.info("GPU 监控服务已启动，监控所有 GPU")
    
    while True:
        metrics = get_gpu_metrics(gpu_ids)
        
        if target_count is not None and len(metrics) != target_count:
            logging.warning(f"检测到的 GPU 数量 ({len(metrics)}) 与预期 ({target_count}) 不符，跳过此次检查。")
            time.sleep(CHECK_INTERVAL)
            continue
            
        stuck_status = [is_gpu_stuck(m) for m in metrics]
        all_stuck = all(stuck_status)
        stuck_count = sum(stuck_status)
        
        stuck_gpus_idx = [str(m['idx']) for m, stuck in zip(metrics, stuck_status) if stuck]
        if stuck_count > 0:
            status_msg = f"检测到 {stuck_count}/{len(metrics)} 张卡疑似僵死 (GPU: {', '.join(stuck_gpus_idx)})。"
        else:
            status_msg = f"所有监控的 GPU 均在正常工作。"

        if all_stuck:
            if stuck_start_time is None:
                stuck_start_time = time.time()
                logging.warning(f"{status_msg} 触发全部卡死条件！开始计时...")
            else:
                elapsed = time.time() - stuck_start_time
                logging.info(f"{status_msg} 持续卡死时间: {int(elapsed)} 秒 / 阈值: {STUCK_TIMEOUT} 秒")
                
                if elapsed >= STUCK_TIMEOUT:
                    logging.error("条件满足！准备Kill进程并重启训练！")
                    kill_gpu_processes(gpu_ids)
                    time.sleep(5)
                    
                    logging.info(f"正在执行重启命令: {RESTART_CMD}")
                    subprocess.Popen(RESTART_CMD, shell=True, executable='/bin/bash')
                    
                    stuck_start_time = None
                    logging.info("重启命令已下发，休眠 5 分钟等待环境初始化...")
                    time.sleep(300) 
        else:
            if stuck_start_time is not None:
                logging.info(f"{status_msg} 状态恢复正常，累计的卡死计时器已重置。")
                stuck_start_time = None
            else:
                logging.info(f"状态汇报: {status_msg}")
                
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
