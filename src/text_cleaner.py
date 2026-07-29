import re
import unicodedata

# HWP·PDF 원본에서 글머리 기호가 사용자 정의 영역(PUA) 문자로 추출되어 그대로 남는다.
# 줄 앞에 오면 목록 구조를 살리기 위해 "- "로, 문장 중간이면 공백으로 바꾼다.
PRIVATE_USE_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))
SOFT_HYPHEN = "\xad"
SPACE_LIKE = {"\xa0", " ", " ", " ", " ", " ", " ",
              " ", " ", " ", " ", " ", " ", " ",
              "　", "​", "﻿"}
REPLACEMENT_CHARACTER = "�"


# 문자가 사용자 정의 영역(PUA)에 속하는지 확인합니다.
def is_private_use(character: str) -> bool:
    code = ord(character)
    return any(start <= code <= end for start, end in PRIVATE_USE_RANGES)


# PUA 글머리 기호, 제어문자, 특수 공백을 표준 문자로 바꿉니다.
def normalize_characters(text: str) -> tuple[str, dict]:
    counts = {"pua_replaced": 0, "invalid_removed": 0, "space_normalized": 0}
    result = []
    at_line_start = True

    for character in text:
        # 줄바꿈과 탭도 제어문자(Cc)라서 아래 C 카테고리 삭제보다 먼저 처리해야 한다.
        if character in {"\n", "\t"}:
            result.append(character)
            at_line_start = character == "\n"
            continue

        # 글머리 기호로 쓰인 PUA 문자와 소프트 하이픈은 줄 앞에서만 목록 기호로 살린다.
        if is_private_use(character) or character in {SOFT_HYPHEN, REPLACEMENT_CHARACTER}:
            counts["pua_replaced"] += 1
            result.append("- " if at_line_start else " ")
            at_line_start = False
            continue

        if character in SPACE_LIKE:
            counts["space_normalized"] += 1
            result.append(" ")
            continue

        # 제어문자(Cc)와 유니코드에 할당되지 않은 번호(Cn)는 어떤 글꼴로도 표시할 수 없어 지운다.
        if unicodedata.category(character).startswith("C"):
            counts["invalid_removed"] += 1
            continue

        result.append(character)
        at_line_start = at_line_start and character.isspace()

    return "".join(result), counts


# 줄 끝 공백과 연속 공백을 줄이고, 빈 줄이 지정한 수보다 많이 이어지지 않게 합니다.
def collapse_whitespace(text: str, max_blank_lines: int) -> tuple[str, dict]:
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.split("\n")]

    collapsed = []
    blank_run = 0
    blank_lines_removed = 0
    for line in lines:
        if line:
            blank_run = 0
            collapsed.append(line)
            continue
        blank_run += 1
        if blank_run <= max_blank_lines:
            collapsed.append(line)
        else:
            blank_lines_removed += 1

    return "\n".join(collapsed).strip(), {"blank_lines_removed": blank_lines_removed}


# 바로 뒤에 똑같이 반복되는 문단을 제거합니다.
# 표의 같은 값이 여러 행에 나뉘어 반복되는 경우를 지우지 않도록 연속된 중복만 대상으로 합니다.
# 길이는 서식용 정렬 공백을 뺀 실제 내용 기준으로 재서, 공백이 많은 줄이 문턱을 넘지 않게 합니다.
def remove_repeated_paragraphs(text: str, min_length: int) -> tuple[str, dict]:
    kept = []
    removed_lines = []
    for line in text.split("\n"):
        if kept and line == kept[-1] and len(" ".join(line.split())) >= min_length:
            removed_lines.append(line)
            continue
        kept.append(line)

    # 의도적으로 제거한 분량을 보존 검증에서 손실과 구분하기 위해 숫자 개수를 함께 센다.
    removed_digits = len(re.sub(r"[^0-9]", "", "".join(removed_lines)))
    return "\n".join(kept), {
        "duplicate_paragraphs_removed": len(removed_lines),
        "duplicate_digits_removed": removed_digits,
    }


# 원본 본문에 클리닝 규칙을 순서대로 적용하고 처리 통계를 함께 반환합니다.
def clean_document_text(
    text: str,
    *,
    max_blank_lines: int,
    min_duplicate_paragraph_length: int,
) -> tuple[str, dict]:
    stats = {"original_length": len(text)}

    text, character_stats = normalize_characters(text)
    stats.update(character_stats)

    text, duplicate_stats = remove_repeated_paragraphs(text, min_duplicate_paragraph_length)
    stats.update(duplicate_stats)

    text, whitespace_stats = collapse_whitespace(text, max_blank_lines)
    stats.update(whitespace_stats)

    stats["cleaned_length"] = len(text)
    return text, stats


# 클리닝 후에도 남아 있어야 하는 핵심 정보를 뽑아 전후 비교에 사용합니다.
def extract_preservation_keys(text: str, requirement_id_pattern: str) -> dict:
    digits = re.sub(r"[^0-9]", "", text)
    return {
        "requirement_ids": sorted(set(re.findall(requirement_id_pattern, text))),
        "digit_count": len(digits),
    }


# 공백 정규화의 영향을 받지 않도록 비교용으로 공백을 하나로 줄입니다.
def _comparable(text: str) -> str:
    return " ".join(text.split())


# 메타데이터의 핵심 값과 요구사항 번호가 클리닝 전후에 그대로 남아 있는지 확인합니다.
# 연속 중복 문단은 의도적으로 지우므로, 그만큼의 숫자 감소는 손실로 보지 않습니다.
def verify_preservation(
    original_text: str,
    cleaned_text: str,
    metadata_values: list[str],
    requirement_id_pattern: str,
    duplicate_digits_removed: int,
) -> dict:
    before = extract_preservation_keys(original_text, requirement_id_pattern)
    after = extract_preservation_keys(cleaned_text, requirement_id_pattern)

    # 원본에 있던 메타데이터 값 중 클리닝 후 사라진 값을 찾는다.
    comparable_original = _comparable(original_text)
    comparable_cleaned = _comparable(cleaned_text)
    # 원본에 없는 값은 대조할 수 없으므로 검사 대상에서 빼고, 실제 검사한 개수를 함께 남긴다.
    checked_values = [
        value for value in metadata_values if _comparable(value) in comparable_original
    ]
    lost_values = [
        value for value in checked_values if _comparable(value) not in comparable_cleaned
    ]
    lost_requirement_ids = sorted(set(before["requirement_ids"]) - set(after["requirement_ids"]))
    unexplained_digit_loss = (
        before["digit_count"] - after["digit_count"] - duplicate_digits_removed
    )

    return {
        "requirement_id_count": len(after["requirement_ids"]),
        "lost_requirement_ids": ";".join(lost_requirement_ids),
        "digit_count_before": before["digit_count"],
        "digit_count_after": after["digit_count"],
        "unexplained_digit_loss": unexplained_digit_loss,
        "checked_values": len(checked_values),
        "lost_metadata_values": ";".join(lost_values),
        "preserved": not lost_values
        and not lost_requirement_ids
        and unexplained_digit_loss == 0,
    }
