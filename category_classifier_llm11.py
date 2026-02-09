"""
Product Category CLASSIFIER V11 - ACCURACY OPTIMIZED
- Enhanced prompting with few-shot examples
- Strict category validation
- Explicit valid category constraints
- Optimized for Gemini 2.5 Flash Lite accuracy
"""
from __future__ import annotations
import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import Counter
from datetime import datetime
import sys
import random
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import google.generativeai as genai

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ============================ CONFIGURATION ============================
INPUT_FILE = "tf_menu.csv"
OUTPUT_FILE = "tf_menu_labeled_v11.csv"
TAXONOMY_FILE = "category_definitions.json"

CONFIG: Dict[str, Any] = {
    # --- Gemini ---
    "GEMINI_API_KEY": os.getenv("GENAI_API_KEY", ""), 
    "MODEL_NAME": "gemini-2.5-flash-lite",
    "TEMPERATURE": 0.1,  # Slightly higher than 0 for better reasoning
    "TOP_P": 0.95,
    "TOP_K": 40,
    
    # --- Processing (REDUCED for better accuracy) ---
    "MAX_WORKERS": 3,  # Reduced from 5
    "RATE_LIMIT_PER_SEC": 3,  # Reduced from 5
    "BATCH_SIZE": 10,  # Reduced from 30 for better accuracy
    
    # --- Retries ---
    "MAX_RETRIES": 3,  # Increased from 2
    "RETRY_DELAY_SEC": 3.0,  # Increased from 2.0
    
    # --- Verification (NEW) ---
    "ENABLE_TWO_PASS_VERIFICATION": False,  # Set to True for critical accuracy
    "VERIFICATION_THRESHOLD": 0.3,  # Verify if >30% of batch has issues
    
    # --- Validation (NEW) ---
    "REJECT_INVALID_CATEGORIES": True,  # Mark invalid categories as ERROR
    "MAX_INVALID_RATE": 0.2,  # Trigger warning if >20% invalid
    
    "MAX_IN_FLIGHT_MULTIPLIER": 1,
    
    # --- Telegram ---
    "TELEGRAM_ENABLED": True,
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
    "TELEGRAM_TIMEOUT_SEC": 15,
    "TELEGRAM_MAX_MSG_CHARS": 4000,
    
    # --- Reporting ---
    "REPORT_EVERY_SECONDS": 60,
    
    # --- Prompt truncation ---
    "TRUNC_TITLE_CHARS": 180,
    "TRUNC_DESC_CHARS": 400,
    "TRUNC_CTX_CHARS": 80,
    
    # --- Rate Watchdog ---
    "WATCHDOG_ENABLED": False,
    "MIN_ITEMS_PER_MINUTE": 0.000000001,
}

if not CONFIG["GEMINI_API_KEY"]:
    raise RuntimeError("GENAI_API_KEY is not set. Please set env var GENAI_API_KEY and re-run.")

genai.configure(api_key=CONFIG["GEMINI_API_KEY"])

# ============================ GLOBAL SHUTDOWN FLAG ============================
class ShutdownFlag:
    """Global flag to signal immediate shutdown on critical errors."""
    def __init__(self):
        self._should_shutdown = False
        self._reason = ""
        self._lock = threading.Lock()
    
    def set(self, reason: str):
        with self._lock:
            if not self._should_shutdown:
                self._should_shutdown = True
                self._reason = reason
    
    def is_set(self) -> bool:
        with self._lock:
            return self._should_shutdown
    
    def reason(self) -> str:
        with self._lock:
            return self._reason

shutdown_flag = ShutdownFlag()

# ============================ RATE LIMITER ============================
class RateLimiter:
    """Thread-safe rate limiter"""
    def __init__(self, max_calls_per_second: float):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self.last_call = 0.0
        self.lock = threading.Lock()
    
    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter(CONFIG["RATE_LIMIT_PER_SEC"])

# ============================ TELEGRAM REPORTER ============================
class TelegramReporter:
    def __init__(self, enabled: bool, bot_token: str, chat_id: str, timeout_sec: int, max_chars: int):
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_sec = timeout_sec
        self.max_chars = max_chars
    
    def _chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            nl = text.rfind("\n", start, end)
            if nl != -1 and nl > start + 200:
                end = nl
            part = text[start:end].strip()
            if part:
                chunks.append(part)
            start = end
        return chunks
    
    def send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload_base = {"chat_id": self.chat_id, "disable_web_page_preview": True}
        for part in self._chunk_text(text):
            payload = dict(payload_base)
            payload["text"] = part
            for attempt in range(3):
                try:
                    r = requests.post(url, json=payload, timeout=self.timeout_sec)
                    if r.status_code == 200:
                        break
                    time.sleep(1.5 * (attempt + 1))
                except Exception:
                    time.sleep(1.5 * (attempt + 1))

telegram = TelegramReporter(
    enabled=CONFIG["TELEGRAM_ENABLED"],
    bot_token=CONFIG["TELEGRAM_BOT_TOKEN"],
    chat_id=CONFIG["TELEGRAM_CHAT_ID"],
    timeout_sec=CONFIG["TELEGRAM_TIMEOUT_SEC"],
    max_chars=CONFIG["TELEGRAM_MAX_MSG_CHARS"],
)

# ============================ COST TRACKING ============================
@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    input_cost_per_1m: float = 0.50
    output_cost_per_1m: float = 3.00
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, in_tokens: int, out_tokens: int):
        with self.lock:
            self.input_tokens += int(in_tokens)
            self.output_tokens += int(out_tokens)
            self.calls += 1
    
    def summary_str(self) -> str:
        with self.lock:
            in_cost = (self.input_tokens / 1_000_000) * self.input_cost_per_1m
            out_cost = (self.output_tokens / 1_000_000) * self.output_cost_per_1m
            total = in_cost + out_cost
            return (
                f"Calls: {self.calls} | "
                f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
                f"Cost: ${total:.4f} (in ${in_cost:.4f} + out ${out_cost:.4f})"
            )

cost_tracker = CostTracker()

# ============================ STATS TRACKING ============================
@dataclass
class StatsTracker:
    processed: int = 0
    success: int = 0
    unknown: int = 0
    error: int = 0
    invalid_category: int = 0  # NEW: Track invalid categories
    level1_counts: Counter = field(default_factory=Counter)
    level12_counts: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_from_rows(self, rows: List[Dict[str, Any]]):
        with self.lock:
            for r in rows:
                self.processed += 1
                l1 = str(r.get("level_1", "ERROR"))
                l2 = str(r.get("level_2", "ERROR"))
                
                if "INVALID_CATEGORY" in r.get("reason", ""):
                    self.invalid_category += 1
                
                if l1 == "ERROR" or l2 == "ERROR":
                    self.error += 1
                elif l1 == "UNKNOWN" or l2 == "UNKNOWN":
                    self.unknown += 1
                else:
                    self.success += 1
                self.level1_counts[l1] += 1
                self.level12_counts[(l1, l2)] += 1
    
    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "processed": self.processed,
                "success": self.success,
                "unknown": self.unknown,
                "error": self.error,
                "invalid_category": self.invalid_category,
                "level1_counts": self.level1_counts.copy(),
                "level12_counts": self.level12_counts.copy(),
            }

stats = StatsTracker()

# ============================ ERROR NOTIFIER ============================
class ErrorNotifier:
    def __init__(self, telegram_reporter: TelegramReporter):
        self.telegram = telegram_reporter
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.last_notification_time: Dict[str, float] = {}
        self.min_notification_interval = 60
    
    def notify_error(self, error_type: str, error_msg: str, batch_info: str = ""):
        with self.lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            count = self.error_counts[error_type]
            now = time.time()
            last_time = self.last_notification_time.get(error_type, 0)
            if now - last_time < self.min_notification_interval:
                return
            self.last_notification_time[error_type] = now
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"🚨 ERROR ALERT @ {ts}\n"
                f"Type: {error_type}\n"
                f"Occurrence: #{count}\n"
            )
            if batch_info:
                msg += f"Batch: {batch_info}\n"
            msg += f"Message: {error_msg[:500]}\n"
            msg += f"\nStats: {stats.processed:,} items processed so far"
            self.telegram.send(msg)

error_notifier = ErrorNotifier(telegram)

# ============================ RATE WATCHDOG ============================
class RateWatchdog:
    def __init__(self, min_items_per_minute: float):
        self.min_items_per_minute = min_items_per_minute
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.last_check_processed = 0
        self.lock = threading.Lock()
    
    def check_rate(self, current_processed: int) -> tuple[bool, str]:
        with self.lock:
            now = time.time()
            if now - self.last_check_time < 120:
                return (False, "")
            if now - self.start_time < 300:
                return (False, "")
            time_elapsed_min = (now - self.last_check_time) / 60.0
            items_processed = current_processed - self.last_check_processed
            if time_elapsed_min > 0:
                rate = items_processed / time_elapsed_min
                self.last_check_time = now
                self.last_check_processed = current_processed
                if rate < self.min_items_per_minute:
                    reason = (
                        f"Processing rate too slow: {rate:.1f} items/min "
                        f"(minimum: {self.min_items_per_minute} items/min). "
                        f"Likely hitting quota limits."
                    )
                    return (True, reason)
            return (False, "")

rate_watchdog = RateWatchdog(CONFIG["MIN_ITEMS_PER_MINUTE"])

# ============================ HELPERS ============================
def normalize_item_id(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip().replace(",", "")

def _clean_text(x: Any, max_len: int) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).replace('"', "").replace("\n", " ").strip()
    if max_len > 0 and len(s) > max_len:
        s = s[:max_len].rstrip()
    return s

def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * part / total):.2f}%"

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def get_processed_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_csv(filepath, usecols=["item_id"], dtype={"item_id": "string"})
        return set(df["item_id"].astype(str))
    except Exception:
        return set()

def batch_iter_from_df(df: pd.DataFrame, batch_size: int):
    batch: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        batch.append(row._asdict())
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def save_final_reports(level1_counts: Counter, level12_counts: Counter):
    pd.DataFrame([{"level_1": k, "count": v} for k, v in level1_counts.most_common()]).to_csv(
        "final_level1_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"level_1": k[0], "level_2": k[1], "count": v} for k, v in level12_counts.most_common()]).to_csv(
        "final_level2_counts.csv", index=False, encoding="utf-8-sig"
    )

def format_progress_message(snap: Dict[str, Any]) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]
    invalid = snap["invalid_category"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[tf_menu] Minute report @ {ts}\n"
        f"Processed: {processed:,}\n"
        f"✅ Success: {success:,} ({_pct(success, processed)})\n"
        f"❓ Unknown: {unknown:,} ({_pct(unknown, processed)})\n"
        f"❌ Error:   {error:,} ({_pct(error, processed)})\n"
        f"⚠️ Invalid Categories: {invalid:,}\n\n"
        f"[COST]\n{cost_tracker.summary_str()}\n"
    )

def format_shutdown_message(reason: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snap = stats.snapshot()
    return (
        f"[tf_menu] 🚨 SHUTDOWN TRIGGERED @ {ts}\n"
        f"Reason: {reason}\n\n"
        f"Progress so far:\n"
        f"Processed: {snap['processed']:,}\n"
        f"✅ Success: {snap['success']:,}\n"
        f"❓ Unknown: {snap['unknown']:,}\n"
        f"❌ Error:   {snap['error']:,}\n"
        f"⚠️ Invalid: {snap['invalid_category']:,}\n\n"
        f"[COST]\n{cost_tracker.summary_str()}\n"
        f"\nThe app has stopped to prevent further API calls."
    )

def format_final_message(snap: Dict[str, Any], top_n_l1: int = 40, top_n_l12: int = 60) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]
    invalid = snap["invalid_category"]
    level1_counts: Counter = snap["level1_counts"]
    level12_counts: Counter = snap["level12_counts"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"[tf_menu] FINAL report @ {ts}")
    lines.append(f"Processed: {processed:,}")
    lines.append(f"✅ Success: {success:,}")
    lines.append(f"❓ Unknown: {unknown:,}")
    lines.append(f"❌ Error:   {error:,}")
    lines.append(f"⚠️ Invalid: {invalid:,}")
    lines.append("")
    lines.append("[COST]")
    lines.append(cost_tracker.summary_str())
    lines.append("")
    lines.append(f"Top Level 1 categories (top {top_n_l1}):")
    for k, v in level1_counts.most_common(top_n_l1):
        lines.append(f"- {k}: {v:,}")
    lines.append("")
    lines.append(f"Top Level 1 → Level 2 pairs (top {top_n_l12}):")
    for (l1, l2), v in level12_counts.most_common(top_n_l12):
        lines.append(f"- {l1} → {l2}: {v:,}")
    lines.append("")
    lines.append("Saved CSVs: final_level1_counts.csv, final_level2_counts.csv")
    return "\n".join(lines)

# ============================ ERROR DETECTION ============================
def is_critical_gemini_error(err: str) -> bool:
    e = (err or "").lower()
    if "status: 403" in e or "received http2 header with status: 403" in e:
        return True
    if "goaway received" in e and "client_misbehavior" in e:
        return True
    if "grpc_status:14" in e and "http2_error:11" in e:
        return True
    auth_phrases = ["permission denied", "authentication", "invalid api key", "unauthorized", "forbidden"]
    if any(p in e for p in auth_phrases):
        return True
    return False

def is_transient_gemini_error(err: str) -> bool:
    e = (err or "").lower()
    if "429" in e:
        return True
    if "504" in e:
        return True
    if "deadline expired" in e or "deadline_exceeded" in e:
        return True
    if "stream cancelled" in e or "stream canceled" in e:
        return True
    if "rpc::cancelled" in e or "status = cancelled" in e:
        return True
    if "unavailable" in e:
        return True
    for code in ["500", "502", "503"]:
        if code in e:
            return True
    transient_phrases = [
        "connection reset", "connection aborted", "timed out", "timeout", "tls",
        "socket", "temporarily unavailable", "server closed", "broken pipe",
    ]
    if any(p in e for p in transient_phrases):
        return True
    return False

# ============================ JSON PARSING ============================
def parse_json_strict(text: str) -> Any:
    raw = (text or "").strip()
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        lower = raw.lower()
        if "```json" in lower:
            start = lower.find("```json") + 7
            end = raw.find("```", start)
            if end != -1:
                candidate = raw[start:end].strip()
                return json.loads(candidate)
        if "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            if end != -1:
                candidate = raw[start:end].strip()
                return json.loads(candidate)
    except Exception:
        pass
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")
    candidates: List[str] = []
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw[first_brace:last_brace + 1])
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(raw[first_bracket:last_bracket + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    raise ValueError("Failed to parse JSON from model response")

# ============================ TAXONOMY VALIDATOR ============================
class TaxonomyValidator:
    """Validates classifications against taxonomy"""
    
    def __init__(self, taxonomy: Dict[str, Any]):
        self.taxonomy = taxonomy
        self.valid_level1: Set[str] = set()
        self.level1_to_level2: Dict[str, Set[str]] = {}
        
        # Build validation structures
        for level_2, data in taxonomy.items():
            level_1 = data.get("level_1", "")
            self.valid_level1.add(level_1)
            if level_1 not in self.level1_to_level2:
                self.level1_to_level2[level_1] = set()
            self.level1_to_level2[level_1].add(level_2)
    
    def validate(self, level_1: str, level_2: str) -> Tuple[bool, str]:
        """
        Returns (is_valid, error_message)
        """
        # Allow ERROR and UNKNOWN
        if level_1 in ["ERROR", "UNKNOWN"] or level_2 in ["ERROR", "UNKNOWN"]:
            return (True, "")
        
        # Check level_1 exists
        if level_1 not in self.valid_level1:
            return (False, f"Invalid level_1: '{level_1}' not in taxonomy")
        
        # Check level_2 belongs to level_1
        if level_1 in self.level1_to_level2:
            if level_2 not in self.level1_to_level2[level_1]:
                valid_l2s = ", ".join(sorted(list(self.level1_to_level2[level_1]))[:5])
                return (False, f"Invalid level_2: '{level_2}' not valid for level_1 '{level_1}'. Valid options: {valid_l2s}...")
        
        return (True, "")
    
    def get_valid_categories_str(self) -> str:
        """Generate string showing all valid categories"""
        lines = ["VALID CATEGORIES:\n"]
        lines.append("Level 1 Categories:")
        for l1 in sorted(self.valid_level1):
            lines.append(f"  - {l1}")
        
        lines.append("\nLevel 1 → Level 2 Mappings:")
        for l1 in sorted(self.valid_level1):
            if l1 in self.level1_to_level2:
                l2s = sorted(list(self.level1_to_level2[l1]))
                for l2 in l2s:
                    lines.append(f"  - {l1} → {l2}")
        
        return "\n".join(lines)

# ============================ CLASSIFIER ENGINE ============================
class ClassifierEngine:
    """
    Enhanced classifier with:
    - Few-shot examples
    - Explicit valid categories
    - Strict validation
    - Optional two-pass verification
    """
    
    def __init__(self, taxonomy_path: str):
        # Load taxonomy
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        
        self.taxonomy = taxonomy
        self.taxonomy_str = json.dumps(taxonomy, ensure_ascii=False, separators=(",", ":"))
        
        # Initialize validator
        self.validator = TaxonomyValidator(taxonomy)
        
        # Get valid categories string
        self.valid_categories_str = self.validator.get_valid_categories_str()
        
        # Initialize model
        self.model = genai.GenerativeModel(
            CONFIG["MODEL_NAME"],
            generation_config={
                "temperature": CONFIG["TEMPERATURE"],
                "top_p": CONFIG.get("TOP_P", 0.95),
                "top_k": CONFIG.get("TOP_K", 40),
            },
        )
        
        logging.info(f"✅ Classifier initialized with {len(self.validator.valid_level1)} level_1 categories")
    
    def _get_few_shot_examples(self) -> str:
        """Generate few-shot examples for the prompt"""
        return """
EXAMPLES OF CORRECT CLASSIFICATION:

Example 1 - Iranian Kebab:
Input: {"id": "001", "title": "چلو کباب کوبیده", "desc": "کباب کوبیده با برنج سفید", "context_category": "غذا"}
Output: {"id": "001", "level_1": "غذای ایرانی", "level_2": "کباب", "reason": "کباب کوبیده با چلو - مشخصاً کباب ایرانی سیخی"}

Example 2 - Pizza:
Input: {"id": "002", "title": "پیتزا پپرونی مخصوص", "desc": "پیتزا با گوشت پپرونی و پنیر موزارلا", "context_category": "فست فود"}
Output: {"id": "002", "level_1": "فست‌فود", "level_2": "پیتزا", "reason": "پیتزا پپرونی - واضحاً پیتزا"}

Example 3 - Side Dish:
Input: {"id": "003", "title": "ماست و خیار", "desc": "ماست با خیار خلال شده", "context_category": "پیش غذا"}
Output: {"id": "003", "level_1": "مشترک (غذای ایرانی/فست‌فود)", "level_2": "مخلفات", "reason": "آیتم کنار غذا - ماست و خیار"}

Example 4 - Coffee:
Input: {"id": "004", "title": "کافه لاته بزرگ", "desc": "اسپرسو با شیر بخار داده شده", "context_category": "نوشیدنی گرم"}
Output: {"id": "004", "level_1": "کافه", "level_2": "نوشیدنی گرم بر پایه قهوه", "reason": "لاته - نوشیدنی گرم قهوه‌ای"}

Example 5 - Burger:
Input: {"id": "005", "title": "دابل چیزبرگر", "desc": "برگر با دو لایه گوشت و پنیر چدار", "context_category": "فست فود"}
Output: {"id": "005", "level_1": "فست‌فود", "level_2": "برگر", "reason": "دابل چیزبرگر - برگر با دو لایه"}

Example 6 - Persian Stew:
Input: {"id": "006", "title": "قورمه سبزی", "desc": "خورش قورمه سبزی با لوبیا قرمز", "context_category": "خورش"}
Output: {"id": "006", "level_1": "غذای ایرانی", "level_2": "خورش", "reason": "قورمه سبزی - خورش ایرانی کلاسیک"}

Example 7 - Fried Chicken:
Input: {"id": "007", "title": "مرغ سوخاری کنتاکی", "desc": "تکه‌های مرغ سوخاری کرانچی", "context_category": "فست فود"}
Output: {"id": "007", "level_1": "فست‌فود", "level_2": "سوخاری", "reason": "مرغ سوخاری تکه‌ای"}

Example 8 - Salad:
Input: {"id": "008", "title": "سالاد سزار", "desc": "سالاد با مرغ گریل، پارمزان و سس سزار", "context_category": "سالاد"}
Output: {"id": "008", "level_1": "غذای سالم", "level_2": "سالاد", "reason": "سالاد سزار کلاسیک"}

Example 9 - Dessert:
Input: {"id": "009", "title": "چیزکیک نیویورکی", "desc": "چیزکیک کلاسیک با بستر بیسکوییت", "context_category": "دسر"}
Output: {"id": "009", "level_1": "دسر", "level_2": "دسر", "reason": "چیزکیک - دسر مدرن"}

Example 10 - Unknown Case:
Input: {"id": "010", "title": "محصول ویژه", "desc": "توضیحات کامل ندارد", "context_category": "نامشخص"}
Output: {"id": "010", "level_1": "UNKNOWN", "level_2": "UNKNOWN", "reason": "اطلاعات ناکافی برای دسته‌بندی"}
"""
    
    def _build_prompt(self, products_for_prompt: List[Dict[str, Any]]) -> str:
        """Build enhanced prompt with examples and constraints"""
        products_json = json.dumps(products_for_prompt, ensure_ascii=False, separators=(",", ":"))
        
        few_shot = self._get_few_shot_examples()
        
        prompt = f"""You are an expert AI Food Data Classifier for Persian/Iranian restaurant menus.

{few_shot}

{self.valid_categories_str}

TAXONOMY (COMPLETE REFERENCE):
{self.taxonomy_str}

CRITICAL CLASSIFICATION RULES:
1. ⚠️ You MUST ONLY use categories that exist in the TAXONOMY above
2. ⚠️ DO NOT INVENT or CREATE new categories - they will be marked as ERROR
3. ⚠️ Every level_2 MUST be a valid child of the level_1 you choose
4. If NO category fits perfectly, use 'UNKNOWN' - NEVER make up categories
5. Analyze 'title' and 'desc' first; use 'context_category' only as hint
6. Check 'explanation' (what INCLUDES) and 'exclusions' (what MUST NOT) carefully
7. Choose the MOST SPECIFIC category that fits
8. When in doubt between similar categories, check exclusions to decide

STEP-BY-STEP PROCESS FOR EACH ITEM:
Step 1: Read title and description carefully
Step 2: Identify main ingredients and preparation method
Step 3: Find the best matching Level 1 category from valid list
Step 4: Within that Level 1, find the most specific Level 2
Step 5: Double-check against exclusions
Step 6: Verify your choice exists in the taxonomy
Step 7: Output the classification

INPUT PRODUCTS (JSON array):
{products_json}

OUTPUT FORMAT:
Return ONLY a valid JSON array (no markdown, no extra text).
Same length and same order as input.
Each item must have this exact structure:
[
  {{"id": "...", "level_1": "...", "level_2": "...", "reason": "brief explanation"}},
  ...
]

FINAL REMINDER:
- Only use categories from the TAXONOMY
- If uncertain → use UNKNOWN
- Never invent new categories
- Match level_2 to correct level_1
"""
        return prompt
    
    def _fallback_results(self, products_for_prompt: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        """Generate fallback ERROR results"""
        return [{"id": p["id"], "level_1": "ERROR", "level_2": "ERROR", "reason": reason} for p in products_for_prompt]
    
    def _validate_and_normalize(self, products_for_prompt: List[Dict[str, Any]], parsed: Any) -> List[Dict[str, Any]]:
        """Validate and normalize API response with strict category checking"""
        id_order = [str(p["id"]) for p in products_for_prompt]
        
        if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
            parsed = parsed["results"]
        
        if not isinstance(parsed, list):
            return self._fallback_results(products_for_prompt, "JSON_SHAPE_INVALID")
        
        result_map: Dict[str, Dict[str, Any]] = {}
        for r in parsed:
            if isinstance(r, dict) and "id" in r:
                result_map[str(r["id"])] = r
        
        out: List[Dict[str, Any]] = []
        invalid_count = 0
        
        for pid in id_order:
            r = result_map.get(pid)
            if not isinstance(r, dict):
                out.append({"id": pid, "level_1": "ERROR", "level_2": "ERROR", "reason": "MISSING_ID"})
                continue
            
            level_1 = str(r.get("level_1", "ERROR"))
            level_2 = str(r.get("level_2", "ERROR"))
            reason = str(r.get("reason", ""))
            
            # Validate categories
            if CONFIG.get("REJECT_INVALID_CATEGORIES", True):
                is_valid, error_msg = self.validator.validate(level_1, level_2)
                if not is_valid:
                    logging.warning(f"⚠️ {error_msg} for item {pid}")
                    level_1 = "ERROR"
                    level_2 = "ERROR"
                    reason = f"INVALID_CATEGORY: {error_msg}"
                    invalid_count += 1
            
            out.append({
                "id": pid,
                "level_1": level_1,
                "level_2": level_2,
                "reason": reason,
            })
        
        # Check invalid rate
        if invalid_count > 0:
            invalid_rate = invalid_count / len(id_order)
            if invalid_rate > CONFIG.get("MAX_INVALID_RATE", 0.2):
                logging.error(f"❌ HIGH INVALID RATE: {invalid_count}/{len(id_order)} ({invalid_rate*100:.1f}%)")
                error_notifier.notify_error(
                    "HIGH_INVALID_CATEGORY_RATE",
                    f"{invalid_count}/{len(id_order)} items had invalid categories",
                    f"IDs: {id_order[:3]}"
                )
        
        return out
    
    def _verify_batch(self, products_for_prompt: List[Dict[str, Any]], classifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optional two-pass verification for critical accuracy
        Only verifies items that seem questionable
        """
        if not CONFIG.get("ENABLE_TWO_PASS_VERIFICATION", False):
            return classifications
        
        # Find items to verify (those with ERROR or questionable classifications)
        items_to_verify = []
        for i, (prod, classif) in enumerate(zip(products_for_prompt, classifications)):
            if classif["level_1"] == "ERROR" or classif["level_2"] == "ERROR":
                items_to_verify.append((i, prod, classif))
        
        if not items_to_verify:
            return classifications
        
        logging.info(f"🔍 Verifying {len(items_to_verify)} questionable items...")
        
        # Verify each questionable item
        for idx, prod, classif in items_to_verify:
            verification_prompt = f"""
Verify this classification is correct:

Product: "{prod['title']}" - {prod['desc']}
Classified as: Level 1 = {classif['level_1']}, Level 2 = {classif['level_2']}

Is this classification correct according to the taxonomy?

Respond with JSON only:
{{"is_correct": true/false, "corrected_level_1": "...", "corrected_level_2": "...", "reason": "..."}}
"""
            
            try:
                rate_limiter.wait()
                response = self.model.generate_content(verification_prompt)
                txt = getattr(response, "text", "") or ""
                verified = parse_json_strict(txt)
                
                if not verified.get("is_correct", True):
                    logging.info(f"✏️ Corrected classification for {prod['id']}: {classif['level_1']}/{classif['level_2']} → {verified.get('corrected_level_1')}/{verified.get('corrected_level_2')}")
                    classifications[idx] = {
                        "id": classif["id"],
                        "level_1": verified.get("corrected_level_1", classif["level_1"]),
                        "level_2": verified.get("corrected_level_2", classif["level_2"]),
                        "reason": f"VERIFIED: {verified.get('reason', classif['reason'])}"
                    }
            except Exception as e:
                logging.warning(f"⚠️ Verification failed for {prod['id']}: {e}")
        
        return classifications
    
    def classify_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main classification method with enhanced error handling"""
        if shutdown_flag.is_set():
            raise RuntimeError(f"Shutdown already triggered: {shutdown_flag.reason()}")
        
        # Prepare products for prompt
        products_for_prompt: List[Dict[str, Any]] = []
        for item in batch_data:
            products_for_prompt.append({
                "id": normalize_item_id(item.get("item_id", "")),
                "title": _clean_text(item.get("item_title"), int(CONFIG["TRUNC_TITLE_CHARS"])),
                "desc": _clean_text(item.get("item_description"), int(CONFIG["TRUNC_DESC_CHARS"])),
                "context_category": _clean_text(item.get("category_name"), int(CONFIG["TRUNC_CTX_CHARS"])),
            })
        
        prompt = self._build_prompt(products_for_prompt)
        last_err: Optional[str] = None
        batch_ids = [p["id"] for p in products_for_prompt]
        
        logging.info(f"📤 Processing batch: {len(batch_ids)} items, IDs: {batch_ids[:3]}...")
        
        for attempt in range(int(CONFIG["MAX_RETRIES"])):
            if shutdown_flag.is_set():
                raise RuntimeError(f"Shutdown triggered: {shutdown_flag.reason()}")
            
            try:
                logging.info(f"🔄 API call attempt {attempt+1}/{CONFIG['MAX_RETRIES']} for batch {batch_ids[:3]}...")
                
                # Rate limiting
                rate_limiter.wait()
                
                # API call
                response = self.model.generate_content(prompt)
                txt = getattr(response, "text", "") or ""
                
                if not txt.strip():
                    raise RuntimeError("Empty response from model")
                
                # Track tokens
                in_tokens = 0
                out_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                else:
                    in_tokens = max(1, len(prompt) // 3)
                    out_tokens = max(30, len(txt) // 4)
                cost_tracker.update(in_tokens, out_tokens)
                
                # Parse and validate
                parsed = parse_json_strict(txt)
                normalized = self._validate_and_normalize(products_for_prompt, parsed)
                
                # Optional verification
                if CONFIG.get("ENABLE_TWO_PASS_VERIFICATION", False):
                    normalized = self._verify_batch(products_for_prompt, normalized)
                
                logging.info(f"✅ Batch completed: {len(batch_ids)} items processed successfully")
                return normalized
                
            except Exception as e:
                last_err = str(e)
                logging.error(f"❌ API call failed: {last_err[:200]}")
                
                # Categorize error
                le = last_err.lower()
                error_type = "UNKNOWN_ERROR"
                if "429" in le:
                    error_type = "429_QUOTA_EXCEEDED"
                elif "504" in le or "deadline" in le:
                    error_type = "504_TIMEOUT"
                elif "403" in le:
                    error_type = "403_FORBIDDEN"
                elif "500" in le or "502" in le or "503" in le:
                    error_type = "5XX_SERVER_ERROR"
                elif "failed to parse json" in le:
                    error_type = "JSON_PARSE_ERROR"
                
                error_notifier.notify_error(error_type, last_err, f"IDs: {batch_ids[:3]}")
                
                # Check if critical
                if is_critical_gemini_error(last_err):
                    logging.critical(f"🚨 CRITICAL ERROR - Shutting down: {last_err}")
                    shutdown_flag.set(f"Critical Gemini error: {last_err}")
                    raise RuntimeError(shutdown_flag.reason())
                
                # Retry with delay
                if attempt < int(CONFIG["MAX_RETRIES"]) - 1:
                    delay = float(CONFIG["RETRY_DELAY_SEC"]) * (attempt + 1)  # Exponential backoff
                    logging.info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
        
        # All retries failed
        logging.error(f"❌ Batch FAILED after {CONFIG['MAX_RETRIES']} attempts: {last_err[:200]}")
        return self._fallback_results(products_for_prompt, f"API_FAIL: {last_err or 'unknown'}")

# ============================ REPORTING THREAD ============================
class MinuteReporter(threading.Thread):
    def __init__(self, stop_event: threading.Event, interval_sec: int):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.interval_sec = max(10, int(interval_sec))
    
    def run(self):
        while not self.stop_event.is_set():
            slept = 0
            while slept < self.interval_sec and not self.stop_event.is_set():
                time.sleep(1)
                slept += 1
            
            if self.stop_event.is_set():
                break
            
            if shutdown_flag.is_set():
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                break
            
            if bool(CONFIG.get("WATCHDOG_ENABLED", False)):
                current_processed = stats.snapshot()["processed"]
                should_stop, reason = rate_watchdog.check_rate(current_processed)
                if should_stop:
                    logging.critical(f"🚨 RATE WATCHDOG TRIGGERED: {reason}")
                    shutdown_flag.set(reason)
                    telegram.send(format_shutdown_message(reason))
                    break
            
            telegram.send(format_progress_message(stats.snapshot()))

# ============================ MAIN ============================
def main():
    print(f"=== AI PRODUCT CLASSIFIER V11 (ACCURACY OPTIMIZED) ===")
    print(f"Model: {CONFIG['MODEL_NAME']}")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Taxonomy: {TAXONOMY_FILE}")
    print(f"Workers: {CONFIG['MAX_WORKERS']} | Batch: {CONFIG['BATCH_SIZE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}")
    print(f"Retries: {CONFIG['MAX_RETRIES']} | Retry delay: {CONFIG['RETRY_DELAY_SEC']}s")
    print(f"Two-pass verification: {CONFIG.get('ENABLE_TWO_PASS_VERIFICATION', False)}")
    print(f"Reject invalid categories: {CONFIG.get('REJECT_INVALID_CATEGORIES', True)}")
    print("")
    
    # Load data
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, dtype={"item_id": "string"})
        df["item_id"] = df["item_id"].astype(str).str.replace(",", "", regex=False)
        total_rows = len(df)
        print(f"Total Products: {total_rows:,}")
    except Exception as e:
        print(f"Failed to read input file: {e}")
        telegram.send(f"[tf_menu] FAILED to read input file: {e}")
        return
    
    # Check processed
    processed_ids = get_processed_ids(OUTPUT_FILE)
    print(f"Already Processed: {len(processed_ids):,} products")
    
    df_remaining = df[~df["item_id"].isin(processed_ids)]
    remaining = len(df_remaining)
    
    if remaining == 0:
        print("All products processed! Script finished.")
        telegram.send("[tf_menu] All products already processed. Nothing to do.")
        return
    
    print(f"Remaining to Process: {remaining:,}")
    
    # Initialize output file
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=[
            "item_id", "category_name", "item_title", "item_description",
            "level_1", "level_2", "reason"
        ]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    # Initialize classifier
    try:
        engine = ClassifierEngine(TAXONOMY_FILE)
    except Exception as e:
        print(f"Failed to load taxonomy or init model: {e}")
        telegram.send(f"[tf_menu] FAILED to load taxonomy or init model: {e}")
        return
    
    # Start reporting
    stop_event = threading.Event()
    reporter = MinuteReporter(stop_event=stop_event, interval_sec=CONFIG["REPORT_EVERY_SECONDS"])
    reporter.start()
    
    # Send start notification
    telegram.send(
        f"[tf_menu] Started V11 - ACCURACY OPTIMIZED\n"
        f"Model: {CONFIG['MODEL_NAME']}\n"
        f"Remaining: {remaining:,}\n"
        f"Workers: {CONFIG['MAX_WORKERS']} | Batch: {CONFIG['BATCH_SIZE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}\n"
        f"Retries: {CONFIG['MAX_RETRIES']} | RetryDelay: {CONFIG['RETRY_DELAY_SEC']}s\n"
        f"Two-pass verification: {CONFIG.get('ENABLE_TWO_PASS_VERIFICATION', False)}\n"
        f"Invalid category rejection: {CONFIG.get('REJECT_INVALID_CATEGORIES', True)}\n\n"
        f"✅ Enhanced with:\n"
        f"- Few-shot examples (10 examples)\n"
        f"- Explicit valid category constraints\n"
        f"- Strict taxonomy validation\n"
        f"- Category mismatch detection\n"
        f"- Invalid category tracking\n"
    )
    
    # Process batches
    max_in_flight = max(1, int(CONFIG["MAX_WORKERS"]) * int(CONFIG["MAX_IN_FLIGHT_MULTIPLIER"]))
    batch_iter = batch_iter_from_df(df_remaining, int(CONFIG["BATCH_SIZE"]))
    
    def submit_next(executor, futures_map) -> bool:
        if shutdown_flag.is_set():
            return False
        try:
            batch = next(batch_iter)
        except StopIteration:
            return False
        fut = executor.submit(engine.classify_batch, batch)
        futures_map[fut] = batch
        return True
    
    total_batches = (remaining + int(CONFIG["BATCH_SIZE"]) - 1) // int(CONFIG["BATCH_SIZE"])
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting processing: {remaining:,} items in {total_batches:,} batches")
    logging.info(f"Workers: {CONFIG['MAX_WORKERS']} | Batch size: {CONFIG['BATCH_SIZE']}")
    logging.info(f"{'='*60}\n")
    
    batches_completed = 0
    
    try:
        with ThreadPoolExecutor(max_workers=int(CONFIG["MAX_WORKERS"])) as executor:
            futures_map: Dict[Any, List[Dict[str, Any]]] = {}
            
            # Submit initial batches
            for _ in range(max_in_flight):
                if not submit_next(executor, futures_map):
                    break
            
            # Process completed batches
            while futures_map:
                if shutdown_flag.is_set():
                    break
                
                for fut in as_completed(list(futures_map.keys())):
                    batch_input = futures_map.pop(fut)
                    batches_completed += 1
                    progress_pct = (batches_completed / total_batches) * 100
                    logging.info(f"📊 Progress: {batches_completed}/{total_batches} batches ({progress_pct:.1f}%)")
                    
                    if shutdown_flag.is_set():
                        break
                    
                    try:
                        results = fut.result()
                        result_map = {str(r.get("id")): r for r in results if isinstance(r, dict)}
                        
                        processed_rows: List[Dict[str, Any]] = []
                        for input_row in batch_input:
                            input_id = normalize_item_id(input_row.get("item_id", ""))
                            api_res = result_map.get(input_id, {})
                            row_out = {
                                "item_id": input_id,
                                "category_name": input_row.get("category_name"),
                                "item_title": input_row.get("item_title"),
                                "item_description": input_row.get("item_description"),
                                "level_1": api_res.get("level_1", "ERROR"),
                                "level_2": api_res.get("level_2", "ERROR"),
                                "reason": api_res.get("reason", ""),
                            }
                            processed_rows.append(row_out)
                        
                        # Save to file
                        pd.DataFrame(processed_rows).to_csv(
                            OUTPUT_FILE, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                        
                        # Update stats
                        stats.update_from_rows(processed_rows)
                        
                    except RuntimeError as e:
                        logging.critical(f"🚨 RuntimeError in main loop: {e}")
                        shutdown_flag.set(str(e))
                        break
                    except Exception as e:
                        msg = f"Error in main loop batch handling: {e}"
                        logging.error(f"❌ {msg}")
                        telegram.send(f"[tf_menu] {msg}")
                    
                    if shutdown_flag.is_set():
                        break
                    
                    # Submit next batch
                    while len(futures_map) < max_in_flight:
                        if not submit_next(executor, futures_map):
                            break
                    
                    break
            
            # Handle shutdown
            if shutdown_flag.is_set():
                print("\n🚨 SHUTDOWN TRIGGERED. Cancelling remaining in-flight requests...")
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                for f in list(futures_map.keys()):
                    f.cancel()
    
    finally:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing loop ended. Batches completed: {batches_completed}/{total_batches}")
        logging.info(f"{'='*60}\n")
        stop_event.set()
    
    # Check if shutdown
    if shutdown_flag.is_set():
        print("\n═══════════════════════════════════════════════════════════")
        print("🚨 APPLICATION STOPPED DUE TO CRITICAL ERROR")
        print("Reason:", shutdown_flag.reason())
        print("Progress:", stats.snapshot())
        print("Cost:", cost_tracker.summary_str())
        print("═══════════════════════════════════════════════════════════")
        sys.exit(1)
    
    # Complete
    print("\n✅ Processing Complete!")
    snap = stats.snapshot()
    save_final_reports(snap["level1_counts"], snap["level12_counts"])
    telegram.send(format_final_message(snap))
    print(cost_tracker.summary_str())
    print("Saved final reports: final_level1_counts.csv, final_level2_counts.csv")

if __name__ == "__main__":
    main()