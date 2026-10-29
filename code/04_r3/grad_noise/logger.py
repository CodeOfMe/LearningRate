"""实验日志与断点恢复框架。

严格遵循「全局要求」：
  1. 每一步操作完整保存；
  2. 中间数据与结果统一 JSON 存储；
  3. 每条记录含精确时间戳 YYYY-MM-DD HH:MM:SS；
  4. 记录含：步骤编号、步骤描述、输入参数、输出结果、执行状态、错误信息；
  5. 任意步骤失败后可精确定位并恢复，从失败步骤重跑，不重复已完成部分。

用法示例见 stage1_diagnose/detect_dead_values.py。
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime


def now_str():
    """精确时间戳，格式 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class ExperimentLogger:
    """写 JSONL 日志 + 每步骤 JSON 详情，支持断点恢复扫描。

    run_log.jsonl: 每行一条记录（append 友好），用于断点恢复与进度追踪。
    results/<step_id>.json: 每个步骤的完整详情（输入/输出/错误）。
    """

    def __init__(self, log_dir, run_name='run', results_dir=None):
        self.log_dir = log_dir
        # results_dir 默认取实验根目录下的 results/（与 logs/ 平级）
        self.results_dir = results_dir or os.path.join(os.path.dirname(log_dir), 'results')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f'{run_name}_log.jsonl')
        self.done_steps = self._scan_done()

    def _scan_done(self):
        """扫描已成功完成的步骤，用于断点恢复（跳过已完成项）。"""
        done = set()
        if not os.path.exists(self.log_path):
            return done
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('status') == 'success':
                    done.add(rec.get('step_id'))
        return done

    def log(self, step_id, description, input_params=None, output=None,
            status='success', error=None):
        """记录一条步骤，返回记录 dict。"""
        rec = {
            'step_id': step_id,
            'description': description,
            'input': input_params if input_params is not None else {},
            'output': output if output is not None else {},
            'status': status,  # success / failed
            'error': error if error is not None else None,
            'timestamp': now_str(),
        }
        # 1) 追加到 JSONL 日志
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # 2) 每个步骤单独存一份 JSON 详情
        detail_path = os.path.join(self.results_dir, f'{step_id}.json')
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec

    def is_done(self, step_id):
        return step_id in self.done_steps


class ResumableRunner:
    """通用断点恢复执行器。

    - 对每个任务调用 logger.is_done(task_id) 跳过已完成项；
    - 执行成功记录 status=success，失败记录 status=failed + 错误堆栈；
    - 失败不中断整个流程，而是继续后续任务（可事后重跑失败项）。
    """

    def __init__(self, logger):
        self.logger = logger

    def run_tasks(self, tasks, fn):
        """tasks: list[dict]，每项必须含 'task_id'。
        fn(task) -> output dict，抛异常则记 failed。
        返回 (成功数, 失败数, 跳过数)。"""
        succ = fail = skip = 0
        for task in tasks:
            tid = task['task_id']
            if self.logger.is_done(tid):
                skip += 1
                print(f'  [skip] {tid}  (已完成)')
                continue
            try:
                output = fn(task)
                self.logger.log(tid, task.get('description', ''), task, output, 'success')
                succ += 1
                print(f'  [ok]   {tid}')
            except Exception as e:
                err = traceback.format_exc()
                self.logger.log(tid, task.get('description', ''), task, None, 'failed', err)
                fail += 1
                print(f'  [FAIL] {tid}: {e}')
        return {'success': succ, 'failed': fail, 'skipped': skip}
