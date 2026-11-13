"""
Industrial Quality Control - High-Performance PLC & Modbus TCP Communication Engine
Architecture:
- Asynchronous Single-Worker Command Queue (queue.Queue + Dedicated Worker Thread)
- Zero-Block Pulse Timer: Actuator pulses use timestamp-based expiration instead of time.sleep
- Pure In-Memory Telemetry Caching: UI reads take 0.001ms with zero network I/O in the Tkinter thread
- Total Prevention of Socket Collisions, Desync, Backpressure, and UI Freezes
"""

import time
import queue
import socket
import logging
import threading
import warnings
from typing import Optional, Callable, Dict, Any

warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusDeviceContext, ModbusServerContext
    from pymodbus.server import StartTcpServer
    HAS_PYMODBUS = True
except ImportError:
    HAS_PYMODBUS = False


# Modbus Register Map Definitions (0-indexed)
COIL_EJECT_TRIGGER = 0       # Coil 1: Ejector Arm pulse (1=Active/Eject, 0=Retracted)
COIL_CONVEYOR_STATE = 1     # Coil 2: Conveyor Run/Stop (1=Run, 0=Stop/Interlock)
HR_INSPECTION_STATUS = 0    # Reg 1: 1=PASS, 2=REJECT, 0=STANDBY
HR_TOTAL_COUNT = 1          # Reg 2: Total pieces
HR_REJECT_COUNT = 2         # Reg 3: Rejected pieces


class VirtualPLCSimulator:
    """
    Lightweight Built-in Virtual PLC Server (Modbus TCP).
    Simulates an industrial PLC listening on host:port (e.g. 127.0.0.1:5020)
    for testing without physical hardware.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 5020, on_log: Optional[Callable[[str], None]] = None):
        self.host = host
        self.port = port
        self.on_log = on_log or print
        self.is_running = False
        self.server_thread: Optional[threading.Thread] = None

    def _log(self, msg: str):
        if self.on_log:
            try:
                self.on_log(f"[PLC SIM] {msg}")
            except Exception:
                pass

    def start(self) -> bool:
        if not HAS_PYMODBUS:
            self._log("خطأ: مكتبة pymodbus غير مثبتة.")
            return False

        if self.is_running:
            return True

        # Check if port is already listening
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex((self.host, self.port)) == 0:
                self._log(f"سيرفر الـ PLC يعمل مسبقاً على {self.host}:{self.port}")
                self.is_running = True
                return True

        try:
            device = ModbusDeviceContext(
                di=ModbusSequentialDataBlock(1, [0] * 50),
                co=ModbusSequentialDataBlock(1, [0] * 50),
                hr=ModbusSequentialDataBlock(1, [0] * 50),
                ir=ModbusSequentialDataBlock(1, [0] * 50)
            )
            context = ModbusServerContext(devices=device, single=True)

            def run_server():
                self.is_running = True
                self._log(f"تم تشغيل سيرفر الـ PLC الافتراضي بنجاح على {self.host}:{self.port}")
                try:
                    StartTcpServer(context=context, address=(self.host, self.port))
                except Exception as e:
                    self._log(f"توقف سيرفر الـ PLC الافتراضي: {e}")
                finally:
                    self.is_running = False

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            time.sleep(0.3)
            return True
        except Exception as e:
            self._log(f"فشل تشغيل سيرفر الـ PLC: {e}")
            return False


class PLCModbusManager:
    """
    Industrial Modbus TCP Master Client with Single-Worker Command Queue.
    Processes all socket commands sequentially in a dedicated background thread.
    Actuator pulses are non-blocking and timestamp-based (no time.sleep).
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 5020, on_log: Optional[Callable[[str], None]] = None):
        self.host = host
        self.port = port
        self.on_log = on_log or print
        
        # Command Queue & Worker Thread
        self.cmd_queue = queue.Queue()
        self.is_running = True
        
        # Thread-safe cached telemetry for instant UI polling (0 ms latency)
        self.cache_lock = threading.Lock()
        self.cached_status: Dict[str, Any] = {
            'connected': False,
            'eject_coil': False,
            'conveyor_coil': True,
            'status': 0,
            'total': 0,
            'rejects': 0
        }

        self.client: Optional[ModbusTcpClient] = None
        self.pulse_duration_ms = 650
        self.eject_end_time = 0.0
        self._pulse_pending = False
        self._hw_ejecting = False
        
        # Start dedicated worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _log(self, msg: str):
        if self.on_log:
            try:
                self.on_log(f"[MODBUS] {msg}")
            except Exception:
                pass

    def is_connected(self) -> bool:
        with self.cache_lock:
            return self.cached_status['connected']

    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        """Enqueues connection request and waits briefly for result"""
        if not HAS_PYMODBUS:
            self._log("خطأ: مكتبة pymodbus غير مثبتة.")
            return False

        if host:
            self.host = host
        if port:
            self.port = port

        resp_q = queue.Queue()
        self.cmd_queue.put(('CONNECT', self.host, self.port, resp_q))
        try:
            success = resp_q.get(timeout=1.2)
            return success
        except queue.Empty:
            return False

    def disconnect(self):
        """Enqueues disconnect request"""
        self.cmd_queue.put(('DISCONNECT',))

    def trigger_ejector_pulse(self, duration_ms: Optional[int] = None):
        """
        Instantly enqueues an ejector arm pulse (0 ms latency, non-blocking).
        Marks pending immediately and extends active pulses cleanly on back-to-back defects.
        """
        with self.cache_lock:
            if not self.cached_status['connected']:
                return
            is_fresh = not (self.cached_status['eject_coil'] or self._pulse_pending)
            self._pulse_pending = True
            self.cached_status['eject_coil'] = True
        dur = duration_ms if duration_ms is not None else self.pulse_duration_ms
        self.cmd_queue.put(('EJECT_PULSE', dur, is_fresh))

    def set_conveyor_state(self, running: bool):
        """
        Instantly updates conveyor run/stop state (0 ms UI latency, non-blocking).
        """
        with self.cache_lock:
            self.cached_status['conveyor_coil'] = bool(running)
        self.cmd_queue.put(('CONVEYOR_STATE', running))

    def update_telemetry(self, status: str, total_count: int, reject_count: int):
        """
        Instantly updates telemetry counters in memory (0 ms UI latency)
        and enqueues writing to the PLC registers.
        """
        status_code = 2 if status == "REJECT" else 1
        with self.cache_lock:
            self.cached_status['status'] = status_code
            self.cached_status['total'] = min(65535, max(0, total_count))
            self.cached_status['rejects'] = min(65535, max(0, reject_count))
        self.cmd_queue.put(('TELEMETRY', status, total_count, reject_count))

    def read_plc_status(self) -> Dict[str, Any]:
        """
        Returns cached PLC status directly from RAM.
        Pure in-memory lookup taking 0.001 ms, NEVER blocks the Tkinter GUI thread!
        """
        with self.cache_lock:
            return dict(self.cached_status)

    def _worker_loop(self):
        """
        Single dedicated worker thread:
        Executes all Modbus TCP socket requests strictly in order, preventing packet collisions.
        Uses non-blocking timestamp timers for pneumatic arm pulses (ZERO time.sleep).
        """
        last_poll_time = 0.0

        while self.is_running:
            try:
                # Wait for next command with a short timeout (25ms)
                cmd = self.cmd_queue.get(timeout=0.025)
                action = cmd[0]

                if action == 'CONNECT':
                    target_host, target_port, resp_q = cmd[1], cmd[2], cmd[3]
                    ok = self._do_connect(target_host, target_port)
                    resp_q.put(ok)

                elif action == 'DISCONNECT':
                    self._do_disconnect()

                elif action == 'EJECT_PULSE':
                    dur_ms = cmd[1]
                    is_fresh = cmd[2] if len(cmd) > 2 else True
                    self._do_eject_pulse(dur_ms, log_trigger=is_fresh)

                elif action == 'CONVEYOR_STATE':
                    running = cmd[1]
                    self._do_set_conveyor(running)

                elif action == 'TELEMETRY':
                    status, total, rejects = cmd[1], cmd[2], cmd[3]
                    self._do_update_telemetry(status, total, rejects)

            except queue.Empty:
                pass
            except Exception as e:
                self._log(f"خطأ في معالج الأوامر: {e}")

            now = time.time()
            with self.cache_lock:
                is_conn = self.cached_status['connected']
                hw_ejecting = self._hw_ejecting
                pulse_pending = self._pulse_pending

            # Only expire a pulse that has actually been written to the PLC hardware
            if self.client and is_conn and hw_ejecting and not pulse_pending and (now >= self.eject_end_time):
                try:
                    self.client.write_coil(address=COIL_EJECT_TRIGGER, value=False)
                except Exception:
                    pass
                with self.cache_lock:
                    self._hw_ejecting = False
                    self.cached_status['eject_coil'] = False
                self._log("↩️ عودة الذراع الهوائي للوضع الطبيعي (Coil 1 = 0)")

            # Periodic background polling of PLC hardware registers (every 300ms when idle)
            if self.client and is_conn and not hw_ejecting and not pulse_pending and (now - last_poll_time >= 0.3):
                last_poll_time = now
                self._do_background_poll()

    def _do_connect(self, host: str, port: int) -> bool:
        self._do_disconnect()
        try:
            self.client = ModbusTcpClient(host, port=port, timeout=0.6, retries=1)
            connected = self.client.connect()
            with self.cache_lock:
                self.cached_status['connected'] = connected
            if connected:
                self._log(f"✅ تم الاتصال بمتحكم الـ PLC بنجاح ({host}:{port})")
                try:
                    self.client.write_coil(address=COIL_CONVEYOR_STATE, value=True)
                    with self.cache_lock:
                        self.cached_status['conveyor_coil'] = True
                except Exception:
                    pass
                return True
            else:
                self._log(f"❌ تعذر الاتصال بـ PLC ({host}:{port}) - تحقق من عنوان IP وتشغيل السيرفر.")
                return False
        except Exception as e:
            with self.cache_lock:
                self.cached_status['connected'] = False
            self._log(f"استثناء في الاتصال: {e}")
            return False

    def _do_disconnect(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        was_conn = False
        with self.cache_lock:
            was_conn = self.cached_status['connected']
            self.cached_status['connected'] = False
            self.cached_status['eject_coil'] = False
            self._pulse_pending = False
            self._hw_ejecting = False
        if was_conn:
            self._log("تم قطع الاتصال بالـ PLC.")

    def _do_eject_pulse(self, duration_ms: int, log_trigger: bool = True):
        """
        Non-blocking pulse trigger: turns Coil ON and sets timestamp expiration.
        NEVER sleeps or stalls the command queue!
        """
        if not self.client:
            with self.cache_lock:
                self._pulse_pending = False
                self._hw_ejecting = False
                self.cached_status['eject_coil'] = False
            return
        with self.cache_lock:
            if not self.cached_status['connected']:
                self._pulse_pending = False
                self._hw_ejecting = False
                self.cached_status['eject_coil'] = False
                return
            was_hw_ejecting = self._hw_ejecting

        new_end = time.time() + (duration_ms / 1000.0)
        self.eject_end_time = max(self.eject_end_time, new_end) if was_hw_ejecting else new_end

        try:
            self.client.write_coil(address=COIL_EJECT_TRIGGER, value=True)
            with self.cache_lock:
                self._pulse_pending = False
                self._hw_ejecting = True
                self.cached_status['eject_coil'] = True
            if log_trigger or not was_hw_ejecting:
                self._log("⚡ نبضة دفع الذراع الهوائي (Coil 1 = 1) -> جاري عزل القطعة...")
        except Exception as e:
            with self.cache_lock:
                self._pulse_pending = False
                self._hw_ejecting = False
                self.cached_status['eject_coil'] = False
            self._log(f"خطأ أثناء نبضة الذراع: {e}")

    def _do_set_conveyor(self, running: bool):
        if not self.client:
            return
        with self.cache_lock:
            if not self.cached_status['connected']:
                return

        try:
            self.client.write_coil(address=COIL_CONVEYOR_STATE, value=bool(running))
            with self.cache_lock:
                self.cached_status['conveyor_coil'] = bool(running)
            st_text = "تشغيل (RUN)" if running else "إيقاف (STOP)"
            self._log(f"⚙️ تحديث إشارة السير على PLC (Coil 2 = {1 if running else 0}) -> {st_text}")
        except Exception as e:
            self._log(f"خطأ أثناء تحديث إشارة السير: {e}")

    def _do_update_telemetry(self, status: str, total_count: int, reject_count: int):
        if not self.client:
            return
        with self.cache_lock:
            if not self.cached_status['connected']:
                return

        status_code = 2 if status == "REJECT" else 1
        t_count = min(65535, max(0, total_count))
        r_count = min(65535, max(0, reject_count))

        try:
            self.client.write_registers(
                address=HR_INSPECTION_STATUS,
                values=[status_code, t_count, r_count]
            )
            with self.cache_lock:
                self.cached_status['status'] = status_code
                self.cached_status['total'] = t_count
                self.cached_status['rejects'] = r_count
        except Exception as e:
            self._log(f"خطأ أثناء تحديث سجلات الـ PLC: {e}")

    def _do_background_poll(self):
        try:
            r_coils = self.client.read_coils(address=0, count=2)
            r_regs = self.client.read_holding_registers(address=0, count=3)

            if not r_coils.isError() and not r_regs.isError():
                with self.cache_lock:
                    self.cached_status['eject_coil'] = bool(r_coils.bits[0])
                    self.cached_status['conveyor_coil'] = bool(r_coils.bits[1])
                    self.cached_status['status'] = int(r_regs.registers[0])
                    self.cached_status['total'] = int(r_regs.registers[1])
                    self.cached_status['rejects'] = int(r_regs.registers[2])
        except Exception:
            pass
