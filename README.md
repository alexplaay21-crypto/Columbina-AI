# Columbina AI

Telegram-компаньон Коломбина: личность по досье, когнитивный движок, память, эмоции, восприятие
мультимедиа и инициатива.

- Архитектура: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Отчёт, API и стоимость: [docs/REPORT.md](docs/REPORT.md)
- Личность: [data/persona/](data/persona/) (исходное досье — `data/persona/source/`)

## Запуск в Docker (PostgreSQL)

```bash
cp .env.example .env     # TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, ALLOWED_USER_IDS, POSTGRES_PASSWORD
docker compose up -d --build
docker compose logs -f bot
```

База — `postgres:17` в отдельном контейнере, том `pgdata`, порт наружу не открыт.
Бэкап: `docker compose exec db pg_dump -U columbina columbina > backup.sql`.
Перенос старой SQLite-памяти: `python scripts/migrate_sqlite_to_pg.py var/columbina.sqlite3 postgresql://…`.
Сервер за TLS-прокси: `docker build --secret id=pip_ca,src=ca.crt .`

## Запуск без Docker (SQLite — для разработки)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
sudo apt-get install -y ffmpeg      # видео, кружки, признаки голоса
cp .env.example .env                # TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, ALLOWED_USER_IDS, ...
python -m columbina.main
```

## Проверки

```bash
python -m pytest                     # 85 детерминированных тестов на SQLite, без API
TEST_DATABASE_URL=postgresql://user@localhost/columbina_test python -m pytest   # +31 тот же набор на PostgreSQL
python scripts/token_report.py       # оценка токенов и стоимости
python -m evals.run_evals            # живые сценарии характера + модель-судья (нужен ключ, ~$0.5)
```

## Команды бота

`/name` · `/city` · `/tz` · `/initiative off|rare|normal` · `/quiet 23 9` · `/memory` · `/hall` · `/forget` · `/stats` (админ)
