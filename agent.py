"""
Агент-аналитик данных.

LLM (Nemotron 3 Ultra через OpenRouter) работает как агент: у неё есть один
инструмент `run_python`, который выполняет Python-код в песочнице, где уже
загружен датафрейм `df`. Модель сама пишет код, получает реальные результаты
вычислений и на их основе строит выводы — НЕ перефразирует характеристики из
промпта. 

Цикл: модель -> tool_call(run_python) -> выполнение кода -> результат обратно
в модель -> ... -> финальный текстовый вывод (инсайды).
"""

import io
import os
import json
import time
import contextlib
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI


def _load_env(path):
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env(Path(__file__).resolve().parent / ".env")

MODEL      = "nvidia/nemotron-3-super-120b-a12b:free"
MAX_STEPS  = 8

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Выполняет Python-код в песочнице. Уже доступны df "
                        "(pandas.DataFrame с данными пользователя), pd, np, plt. "
                        "Другие библиотеки (seaborn, sklearn, plotly) НЕ установлены — "
                        "не импортируй их, строй графики только через matplotlib (plt). "
                        "Возвращает то, что напечатано через print(); новые фигуры "
                        "matplotlib сохраняются автоматически. НЕ вызывай plt.show(), "
                        "plt.close() и plt.savefig() — фигуру сохранит приложение само."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python-код для выполнения"}
            },
            "required": ["code"],
        },
    },
}]

SYSTEM_PROMPT = """Ты — старший аналитик данных. Тебе дан датафрейм df и
КОНКРЕТНАЯ задача от пользователя. Главное — ответить именно на эту задачу,
а не пройтись шаблоном «обо всём».

ВАЖНО: все цифры, метрики и выводы получай ТОЛЬКО через инструмент run_python,
выполняя реальные вычисления на df. Не выдумывай числа и не оценивай на глаз.

В песочнице доступны только pandas (pd), numpy (np) и matplotlib (plt).
НЕ импортируй seaborn, sklearn, plotly и другие пакеты — их нет. Нужный столбец
(например month из даты) создавай сам внутри кода.

Подход к работе:
- Сначала пойми, что в данных есть для ответа на задачу, и считай именно то,
  что к ней относится. Не гоняй полный EDA ради галочки, если задача узкая.
- Работай ЭФФЕКТИВНО — 3-5 шагов, каждый содержательный (несколько связанных
  вычислений за вызов). В каждом вызове run_python обязательно print() результат.
- Построй 1-2 графика, которые реально иллюстрируют ОТВЕТ на задачу (с заголовком
  и подписями осей), а не «график ради графика». Не вызывай plt.show(),
  plt.close(), plt.savefig() — фигуру сохранит приложение само.

Когда вычислений достаточно — НЕ вызывай инструмент, а напиши финальный отчёт
на русском, строго ПО ДЕЛУ, в Markdown:
## Ответ на задачу
Прямой вывод по вопросу пользователя с конкретными числами.
## Инсайды
3-5 коротких выводов; каждый — наблюдение из данных с цифрой, 1 строка.
## Рекомендации
2-4 практических действия, которые следуют из найденного.

НЕ пересказывай структуру датасета и названия столбцов, НЕ повторяй формулировку
задачи, НЕ описывай, какой код выполнял, НЕ вставляй сырые таблицы и банальные
советы уровня «нужно собирать больше данных». Если данных для вывода не хватает —
скажи об этом одной строкой."""


def _chat(**kwargs):
    """Вызов OpenRouter с повторами. На бесплатном тарифе модель иногда отдаёт
    пустой ответ, служебный «мусор» (ломает разбор JSON в SDK) или ответ 200
    без поля choices (ошибка провайдера / лимит). Все эти случаи — повторяем."""
    last = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**kwargs)
            if getattr(resp, "choices", None):
                return resp
            last = getattr(resp, "error", None) or "ответ без choices (лимит/ошибка провайдера)"
        except Exception as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"OpenRouter не ответил корректно после повторов: {last}")


class Sandbox:
    """Постоянное пространство имён, общее для всех вызовов run_python."""

    def __init__(self, df: pd.DataFrame):
        self.env = {"pd": pd, "np": np, "plt": plt, "df": df}
        self.figures = []         

    def run(self, code: str) -> str:
        out = io.StringIO()
        err = None
        try:
            with contextlib.redirect_stdout(out):
                exec(code, self.env)
        except Exception:
            err = "ОШИБКА выполнения:\n" + traceback.format_exc(limit=2)

        for num in plt.get_fignums():
            fig = plt.figure(num)
            if not fig.axes:
                plt.close(fig)
                continue
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            title = (fig.axes[0].get_title() or f"График {num}")
            self.figures.append((title, buf.getvalue()))
            plt.close(fig)

        if err:
            return err

        text = out.getvalue().strip()
        if len(text) > 4000:
            text = text[:4000] + "\n...[вывод обрезан]"
        return text or "(код выполнен, вывод пуст)"


def _data_preview(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return (f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов\n"
            f"Столбцы и типы:\n{buf.getvalue()}\n"
            f"Первые строки:\n{df.head(5).to_string()}")


def analyze(df: pd.DataFrame, user_goal: str = "", on_step=None, should_stop=None):
    """
    Запускает агентный цикл. on_step(step_dict) — колбэк для отображения хода.
    should_stop() — функция без аргументов; если возвращает True, цикл
    останавливается перед следующим шагом (по нажатию «Остановить» в UI).
    Возвращает (final_report:str, steps:list, figures:list).
    """
    def _stopped():
        return bool(should_stop and should_stop())

    sandbox = Sandbox(df)
    has_goal = bool(user_goal.strip())
    goal = user_goal.strip() or "Проведи разведочный анализ и выдели главное в данных."
    focus = ("Отвечай строго на эту задачу, не отвлекайся на лишнее."
             if has_goal else
             "Сам реши, что в данных важнее всего, и сосредоточься на этом.")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"ЗАДАЧА: {goal}\n{focus}\n\nДанные (df):\n{_data_preview(df)}"},
    ]
    steps = []

    for _ in range(MAX_STEPS):
        if _stopped():
            return "⏹ Анализ остановлен пользователем.", steps, sandbox.figures
        resp = _chat(
            model=MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", temperature=0.2, max_tokens=1500,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip(), steps, sandbox.figures

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            try:
                code = json.loads(tc.function.arguments).get("code", "")
            except Exception:
                code = ""
            result = sandbox.run(code)
            step = {"code": code, "output": result}
            steps.append(step)
            if on_step:
                on_step(step)
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "name": "run_python", "content": result,
            })

    if _stopped():
        return "⏹ Анализ остановлен пользователем.", steps, sandbox.figures
    messages.append({"role": "user",
                     "content": "Достаточно вычислений. Дай финальный отчёт без вызова инструментов."})
    resp = _chat(model=MODEL, messages=messages, temperature=0.2, max_tokens=3000)
    return (resp.choices[0].message.content or "").strip(), steps, sandbox.figures