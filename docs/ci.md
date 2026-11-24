# CI

Проверки и релизы идут на своём сервере Woodpecker CI: https://oysters31.ru/woodpecker. GitHub Actions не используется.

| Файл | Когда | Что делает |
|---|---|---|
| `.woodpecker/ci.yaml` | PR и push в `main` | `gofmt`, `go vet ./...`, `go test -race ./...` |
| `.woodpecker/release.yaml` | тег `v*` | тесты, сборка архивов под 5 платформ, релиз на GitHub |

## Как это работает

- GitHub шлёт событие в Woodpecker, сборка создаётся и ждёт одобрения. Одобряет мейнтейнер после просмотра диффа: код из PR не исполняется без него. Это касается и форков.
- Результат приходит в PR статусом `woodpecker/pr/ci` (push в `main` — `woodpecker/push/ci`, тег — `woodpecker/tag/release`). Ссылка в статусе ведёт на лог.
- Перезапуск: кнопка «Restart» на странице сборки в Woodpecker.
- Образы берутся из `public.ecr.aws/docker/library`, не с Docker Hub.

## Локально

```
test -z "$(gofmt -l .)"
go vet ./...
go test -race ./...
```

## Релиз

Пушнуть тег `vX.Y.Z` в `main`. Публикация использует секрет Woodpecker `github_release_token` (fine-grained токен с правом `Contents: write` только на этот репозиторий, доступен только событию `tag`). Сборку архивов можно повторить локально: `VERSION=vX.Y.Z sh scripts/build-release.sh`.
