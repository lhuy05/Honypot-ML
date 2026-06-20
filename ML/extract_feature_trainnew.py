import pandas as pd
import numpy as np
import re
from urllib.parse import unquote_plus

# ======================
# Keyword sets (Đã nâng cấp XSS_KEYWORDS để bắt các biến thể DOM/Stored XSS)
# ======================

SQL_KEYWORDS = [  "union", "select", "insert", "update", "delete", "drop",
    " or ", " and ", "'", "--", "sleep", "benchmark", "information_schema",
    "concat", "char", "exec", "ascii", "substring", "order by"]
XSS_KEYWORDS = ["<script", "javascript:", "onerror=", "onload=", "alert(",
    "onmouseover=", "onfocus=", "onclick=", "srcdoc=", "confirm(", "prompt(",
    "svg", "img", "iframe", "body", "src=", "href=", "onload"]

PATH_KEYWORDS = [
    "../",
    "..\\",
    "/etc/passwd",
    "boot.ini",
    "win.ini"
]

CMD_KEYWORDS = [
    "cmd.exe",
    "/bin/sh",
    "powershell",
    "wget",
    "curl",
    "whoami",
    "net user"
]
SQLI_PATTERNS = [
    r"'\s*or\s*'?\d+'?\s*=\s*'?\d+'?",
    r"'\s*or\s*'?[a-z]+'?\s*=\s*'?[a-z]+'?",
    r"\bor\b\s+1\s*=\s*1",
    r"union\s+select",
    r"--",
    r"/\*",
    r"#"
]

# ======================
# Helper functions
# ======================
def decode_nginx_hex(text):
    """Giải mã các ký tự bị Nginx mã hóa dạng \x3C, \x3E trong access.log thô"""
    text = str(text)
    return re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), text)


def contains_keywords(text, keywords):
    text = str(text).lower()

    for k in keywords:
        k_clean = k.strip()

        if k_clean in ["or", "and"]:
            if re.search(r'\b' + re.escape(k_clean) + r'\b', text):
                return 1
        elif k in text:
            return 1

    return 0
def count_sqli_patterns(text):
    text = str(text).lower()

    count = 0

    for pattern in SQLI_PATTERNS:
        count += len(re.findall(pattern, text))

    return count
def digit_ratio(text):
    text = str(text)

    if len(text) == 0:
        return 0

    digits = sum(c.isdigit() for c in text)
    return digits / len(text)


def special_char_count(text):
    text = str(text)

    chars = set(
        "<>'\";()=%/\\{}[]:$&|"
    )

    return sum(c in chars for c in text)


def percent_encoding_count(text):
    text = str(text)

    return len(
        re.findall(r"%[0-9a-fA-F]{2}", text)
    )


# --- THÊM CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG ĐỊNH DANH MỚI (ĐỒNG BỘ VỚI REALTIME) ---
def count_html_brackets(text):
    text = str(text)
    return text.count("<") + text.count(">")


def count_quotes(text):
    text = str(text)
    return text.count("'") + text.count('"')


def count_js_brackets(text):
    text = str(text)
    return sum(text.count(char) for char in "()[]{}")


# ======================
# Feature extraction
# ======================

def extract_features(row):
    method = str(row["method"]).upper()

    # Bước 1: Giải mã mã hóa Hex của Nginx phòng trường hợp tập dataset gốc chứa chuỗi log thô
    path = decode_nginx_hex(str(row["path"]))
    query = decode_nginx_hex(str(row["query"]))
    body = decode_nginx_hex(str(row["body"]))

    # Bước 2: Giải mã mã hóa URL Percent Encoding thông thường
    path = unquote_plus(path) if pd.notna(row["path"]) and str(path).lower() != 'nan' else ""
    query = unquote_plus(query) if pd.notna(row["query"]) and str(query).lower() != 'nan' else ""
    body = unquote_plus(body) if pd.notna(row["body"]) and str(body).lower() != 'nan' else ""

    # Bước 3: Tạo văn bản toàn diện và làm sạch tất cả khoảng trắng thừa, tab, xuống dòng (\s+)
    full_text = " ".join([path, query, body]).lower()
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    method_map = {
        "GET": 0,
        "POST": 1,
        "PUT": 2,
        "DELETE": 3,
        "PATCH": 4
    }

    return pd.Series({

        # ---- HTTP method ----
        "method":
            method_map.get(method, -1),

        # ---- lengths ----
        "path_length":
            len(path),

        "query_length":
            len(query),

        "body_length":
            len(body),

        # ---- structure ----
        "slash_count":
            path.count("/"),

        "dot_count":
            path.count("."),

        # ---- suspicious chars ----
        "path_special_chars": special_char_count(path),
        "query_special_chars": special_char_count(query),
        "body_special_chars": special_char_count(body),

        "digit_ratio":
            round(digit_ratio(full_text), 6),  # Đồng bộ làm tròn 6 chữ số thập phân với Realtime

        "percent_encoding_count":
            percent_encoding_count(full_text),

        # ---- attack indicators ----
        "has_sql_keyword":
            contains_keywords(full_text, SQL_KEYWORDS),

        "sqli_pattern_count":
            count_sqli_patterns(full_text),
        "has_xss_keyword":
            contains_keywords(full_text, XSS_KEYWORDS),

        "has_path_traversal":
            contains_keywords(full_text, PATH_KEYWORDS),

        "has_cmd_keyword":
            contains_keywords(full_text, CMD_KEYWORDS),

        # ---- 3 ĐẶC TRƯNG ĐỊNH DANH MỚI ĐƯỢC NỐI TIẾP VÀO CUỐI ----
        "html_bracket_count":
            count_html_brackets(full_text),

        "quote_count":
            count_quotes(full_text),

        "js_bracket_count":
            count_js_brackets(full_text)
    })


# ======================
# Load CSV
# ======================

df = pd.read_csv(
    "realtime_string_features_train_filtered.csv"
)

print("=== [SYSTEM] Đang tiến hành trích xuất đặc trưng nâng cấp cho tập Train... ===")

# Extract features
features = df.apply(
    extract_features,
    axis=1
)

# Add label back
features["label"] = df["label"]

# Save
features.to_csv(
    "train_featuresnew1.csv",
    index=False
)

print("=== [SUCCESS] Đã lưu file train_featuresnew.csv mới thành công! ===")