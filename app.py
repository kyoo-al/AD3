"""
Веб-интерфейс мини-продукта «ИИ-аналитик данных» (Streamlit).

Пользователь загружает CSV/Excel -> LLM-агент анализирует данные, выполняя
Python-код через инструмент run_python -> на экране: ключевые метрики, шаги
агента (код + результаты), графики и текстовые инсайды.

Запуск:  streamlit run app.py
"""

import io
import time

import pandas as pd
import streamlit as st

import agent


st.set_page_config(page_title="ИИ-аналитик данных", layout="wide")
st.title("ИИ-аналитик данных")
st.caption("Загрузите таблицу — LLM-агент сам проанализирует её, выполняя код "
           f"(модель: {agent.MODEL} через OpenRouter).")


def _stretch(fn, *args, **kwargs):
    """Показ виджета на всю ширину контейнера, совместимый со старым и новым
    Streamlit. В новых версиях use_container_width устарел в пользу
    width='stretch' — пробуем сначала новый API, потом старый."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except (TypeError, ValueError):
        return fn(*args, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes) -> pd.DataFrame:
    """Читает CSV/Excel из байтов. Кэшируется по (имя, содержимое),
    поэтому повторный рендер страницы не перечитывает файл заново."""
    lname = name.lower()
    if lname.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    for enc in ("utf-8", "cp1251"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data))


with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader("CSV или Excel", type=["csv", "xlsx", "xls"])
    goal = st.text_area(
        "Что проанализировать? (необязательно)",
        placeholder="Например: найди, что влияет на выручку, и где аномалии",
        height=90,
    )
    run = st.button("Запустить ИИ-анализ", type="primary", use_container_width=True)

if not agent.client.api_key:
    st.error("Не найден OPENROUTER_API_KEY. Добавьте его в файл .env рядом с app.py.")
    st.stop()

if uploaded is None:
    st.info("Загрузите файл в боковой панели, чтобы начать.")
    st.stop()

try:
    df = read_table(uploaded.name, uploaded.getvalue())
except Exception as e:
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

st.subheader("Предпросмотр данных")
c1, c2, c3 = st.columns(3)
c1.metric("Строк", f"{df.shape[0]:,}".replace(",", " "))
c2.metric("Столбцов", df.shape[1])
c3.metric("Пропусков", int(df.isna().sum().sum()))
_stretch(st.dataframe, df.head(20))

if run:
    steps_box = st.container()
    with steps_box:
        st.subheader("Ход работы агента")
    step_n = {"i": 0}

    def show_step(step):
        step_n["i"] += 1
        with steps_box.expander(f"Шаг {step_n['i']}: выполнение кода", expanded=False):
            st.code(step["code"], language="python")
            st.text(step["output"][:2000])

    t0 = time.time()
    with st.spinner("Агент анализирует данные… (это может занять несколько минут)"):
        try:
            report, steps, figures = agent.analyze(df, goal, on_step=show_step)
        except Exception as e:
            st.error(f"Ошибка анализа: {e}")
            st.stop()

    elapsed = time.time() - t0
    st.success(f"Готово за {elapsed:.0f} с. Агент сделал {len(steps)} шаг(ов) вычислений.")

    st.subheader("Графики, построенные агентом")
    if figures:
        cols = st.columns(min(2, len(figures)))
        for i, (title, png) in enumerate(figures):
            _stretch(cols[i % len(cols)].image, png, caption=title)
    else:
        st.info("Агент не построил графики на этом прогоне. "
                "Попробуйте более быструю модель или повторите запуск.")

    st.subheader("Выводы и инсайды")
    st.markdown(report or "_Агент не вернул текстовый отчёт._")

    st.download_button(
        "Скачать отчёт (.md)",
        report or "",
        file_name="analysis_report.md",
        mime="text/markdown",
    )