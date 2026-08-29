import os
import re


def _trace_prefix(doc) -> str:
    raw_trace = doc.get("reasoning_trace", "").strip()
    if raw_trace:
        answer_start = raw_trace.rfind("<channel|><answer>")
        if answer_start >= 0:
            answer_start += len("<channel|>")
        else:
            answer_start = raw_trace.rfind("<answer>")
        if answer_start >= 0:
            raw_trace = raw_trace[:answer_start].rstrip()
    max_chars = int(os.environ.get("COT_TRACE_MAX_CHARS", "0") or 0)
    if max_chars > 0 and len(raw_trace) > max_chars:
        raw_trace = raw_trace[:max_chars].rstrip()
    return raw_trace


def _append_trace(prompt: str, doc) -> str:
    prompt = doc.get("stage1_chat_prompt") or prompt
    raw_trace = _trace_prefix(doc)
    if raw_trace:
        prompt += "\n\n" + raw_trace
    return prompt + "\n\n<answer>"


def global_mmlu_doc_to_text_cot(doc) -> str:
    prompt = (
        f"{doc['question'].strip()}\n"
        f"A. {doc['option_a']}\n"
        f"B. {doc['option_b']}\n"
        f"C. {doc['option_c']}\n"
        f"D. {doc['option_d']}"
    )
    return _append_trace(prompt, doc)


def global_mmlu_doc_to_choice_letters(doc) -> list[str]:
    return ["A", "B", "C", "D"]


def global_mmlu_doc_to_target_letter(doc) -> str:
    return doc["answer"]


def belebele_doc_to_text_cot(doc) -> str:
    prompt = (
        f"P: {doc['flores_passage']}\n"
        f"Q: {doc['question'].strip()}\n"
        f"1. {doc['mc_answer1']}\n"
        f"2. {doc['mc_answer2']}\n"
        f"3. {doc['mc_answer3']}\n"
        f"4. {doc['mc_answer4']}"
    )
    return _append_trace(prompt, doc)


def belebele_doc_to_choice_numbers(doc) -> list[str]:
    return ["1", "2", "3", "4"]


def belebele_doc_to_target_index(doc) -> int:
    return int(doc["correct_answer_num"]) - 1


def arc_easy_doc_to_text_cot(doc) -> str:
    labels = doc["choices"]["label"]
    choices = doc["choices"]["text"]
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    prompt = f"Питання: {doc['question']}\n{options}"
    return _append_trace(prompt, doc)


def arc_easy_doc_to_choice_text(doc) -> list[str]:
    return doc["choices"]["text"]


def arc_easy_doc_to_target_index(doc) -> int:
    return doc["choices"]["label"].index(doc["answerKey"])


def winogrande_doc_to_text_cot(doc) -> str:
    prompt = (
        f"Речення з пропуском: {doc['sentence']}\n"
        f"1. {doc['option1']}\n"
        f"2. {doc['option2']}"
    )
    return _append_trace(prompt, doc)


def winogrande_doc_to_choice_options(doc) -> list[str]:
    return [doc["option1"], doc["option2"]]


def winogrande_doc_to_target_index(doc) -> int:
    return int(doc["answer"]) - 1


def squad_doc_to_text_cot(doc) -> str:
    prompt = doc.get("stage1_chat_prompt") or (
        f"Контекст: {doc['Context']}\n\n"
        f"Питання: {doc['Question']}\n\n"
        "Знайди найкоротший фрагмент з контексту, який відповідає на питання.\n"
        "Якщо відповідь не підтримується контекстом, виведи "
        "<answer>немає відповіді</answer>.\n"
        "Інакше виведи <answer>фрагмент з контексту</answer>."
    )
    trace = _trace_prefix(doc)
    if trace:
        prompt += "\n\n" + trace
    return (
        prompt
        + "\n\nПісля наведеного міркування визнач, чи є відповідь у контексті.\n"
        "Виведи лише одне слово: немає або є.\n\n"
        "<answer_status>"
    )


def squad_prediction_from_trace(doc) -> str:
    trace = doc.get("reasoning_trace", "")
    matches = re.findall(r"<answer>(.*?)</answer>", trace, flags=re.DOTALL)
    if not matches:
        return ""
    answer = matches[-1].strip()
    if answer.casefold() == "немає відповіді":
        return ""
    return answer
