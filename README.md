# Columbina AI

Telegram-компаньон Коломбина: личность по досье, когнитивный движок, память, эмоции, восприятие
мультимедиа и инициатива.

- Архитектура: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Отчёт, API и стоимость: [docs/REPORT.md](docs/REPORT.md)
- Личность: [data/persona/](data/persona/) (исходное досье — `data/persona/source/`)

## Запуск

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
sudo apt-get install -y ffmpeg      # видео, кружки, признаки голоса
cp .env.example .env                # TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, ALLOWED_USER_IDS, ...
python -m columbina.main
```

## Проверки

```bash
python -m pytest                     # 75 детерминированных тестов, без API
python scripts/token_report.py       # оценка токенов и стоимости
python -m evals.run_evals            # живые сценарии характера + модель-судья (нужен ключ, ~$0.5)
```

## Команды бота

`/name` · `/city` · `/tz` · `/initiative off|rare|normal` · `/quiet 23 9` · `/memory` · `/hall` · `/forget` · `/stats` (админ)
