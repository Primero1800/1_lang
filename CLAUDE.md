# Инструкции для всех агентов этого проекта

## Память — обязательно при старте

Прочитай эти файлы перед любой работой:

1. `/home/primero/.claude/projects/-home-primero-Python-python-codes-langgraph-1-lang/memory/AGENTS.md` — протокол работы с памятью, префиксы файлов, правила
2. `/home/primero/.claude/projects/-home-primero-Python-python-codes-langgraph-1-lang/memory/project_state.md` — живой статус воркеров, схема БД, ключевые решения
3. `/home/primero/.claude/projects/-home-primero-Python-python-codes-langgraph-1-lang/memory/feedback_code_style.md` — правила стиля кода, обязательны

Полный индекс памяти: `memory/MEMORY.md`

## Правила памяти (кратко)

- `MEMORY.md` и общие файлы — только **Edit**, никогда **Write**
- Личные файлы — только с **твоим префиксом** (qa_, main_, review_, ...). Чужие — только читать
- `project_state.md` — обновляй только **свою секцию** через Edit
- Если твоей секции нет — добавь через Edit в конец

## Проект

FastAPI + LangGraph пайплайн наполнения Qdrant.
Рабочая директория: `/home/primero/Python/python_codes/langgraph/1_lang`
Референсный проект (паттерны кода): `/home/primero/Python/python_codes/other/5_data_aggregator`
