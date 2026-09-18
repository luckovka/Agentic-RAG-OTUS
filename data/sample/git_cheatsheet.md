# Git Cheatsheet

Короткая памятка для повседневной работы с Git и GitHub.

> Главное правило: **Git не читает мысли. Сначала `git add`, потом `git commit`.**
>
> И да, `git commit -m "..."` — встроенный антиквакатель против внезапного Vim. 😄

---

## 1. Где я вообще?

Показать текущую папку:

```bash
pwd
```

Показать текущую ветку:

```bash
git branch
```

Показать локальные и удалённые ветки:

```bash
git branch -a
```

Проверить, к какому удалённому репозиторию привязан проект:

```bash
git remote -v
```

---

## 2. Что происходит в репозитории?

Главная диагностическая команда:

```bash
git status
```

Она показывает:

- какие файлы изменены;
- какие файлы удалены;
- какие файлы новые (`untracked`);
- какие изменения уже добавлены в staging area;
- на какой ветке вы находитесь;
- синхронизирована ли ветка с `origin`.

Посмотреть изменения, которые ещё **не добавлены** через `git add`:

```bash
git diff
```

Посмотреть изменения, которые уже добавлены в staging area:

```bash
git diff --staged
```

---

## 3. Добавить изменения в staging area

Добавить конкретный файл:

```bash
git add README.md
```

Добавить несколько файлов:

```bash
git add README.md requirements.txt src/app.py
```

Добавить изменения в текущей папке:

```bash
git add .
```

Добавить **все** изменения в репозитории, включая новые, изменённые и удалённые файлы:

```bash
git add -A
```

`git add -A` особенно удобно после перемещения или реорганизации файлов.

После `git add` полезно проверить:

```bash
git status
```

---

## 4. Создать коммит

Создать коммит и сразу передать сообщение:

```bash
git commit -m "Update README"
```

Флаг `-m` означает **message**.

Примеры нормальных сообщений:

```bash
git commit -m "Add initial RAG data corpus"
git commit -m "Organize project structure"
git commit -m "Add PDF ingestion"
git commit -m "Update README with MVP architecture"
```

Если выполнить просто:

```bash
git commit
```

Git откроет текстовый редактор для ввода сообщения коммита.

### Если внезапно открылся Vim

Выйти без сохранения:

```text
Esc
:q!
Enter
```

Сохранить сообщение и выйти:

```text
Esc
:wq
Enter
```

Но проще использовать:

```bash
git commit -m "Your message"
```

---

## 5. Отправить изменения на GitHub

После коммита:

```bash
git push
```

При первом push новой ветки иногда нужно:

```bash
git push -u origin branch-name
```

Флаг `-u` связывает локальную ветку с удалённой, после чего обычно достаточно обычного:

```bash
git push
```

---

## 6. Забрать изменения с GitHub

Обычный вариант:

```bash
git pull
```

Часто удобнее использовать rebase:

```bash
git pull --rebase
```

---

## 7. Клонировать репозиторий на новый компьютер

Сначала перейти в папку, где должен появиться проект:

```bash
cd ~/ML/projects
```

Затем:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

Или через SSH:

```bash
git clone git@github.com:USERNAME/REPOSITORY.git
```

После этого:

```bash
cd REPOSITORY
```

---

## 8. Ветки

Создать новую ветку и сразу перейти на неё:

```bash
git switch -c feature/my-feature
```

Перейти на существующую ветку:

```bash
git switch main
```

Показать ветки:

```bash
git branch
```

Удалить локальную ветку после завершения работы:

```bash
git branch -d feature/my-feature
```

---

## 9. Перемещение и переименование файлов

Переименовать файл:

```bash
mv old_name.md new_name.md
```

Переместить файл:

```bash
mv file.pdf data/
```

Переместить несколько файлов:

```bash
mv file1.pdf file2.pdf notes.txt data/
```

Если в имени есть пробелы:

```bash
mv "API и машинное обучение.pdf" data/
```

После перемещения:

```bash
git add -A
git status
```

---

## 10. Отменить изменения осторожно

Вернуть файл к состоянию последнего коммита:

```bash
git restore filename
```

**Осторожно:** несохранённые изменения в этом файле будут потеряны.

Убрать файл из staging area, но оставить изменения в самом файле:

```bash
git restore --staged filename
```

---

## 11. История коммитов

Краткая история:

```bash
git log --oneline
```

Последний коммит:

```bash
git log -1
```

Удобный граф истории:

```bash
git log --oneline --graph --decorate --all
```

---

## 12. `.gitignore`

Для Python / LLM-проекта часто полезно:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.DS_Store
data/private/
```

### Важно про секреты

Реальные API-ключи:

```text
.env
```

не должны попадать в GitHub.

В репозитории лучше хранить:

```text
.env.example
```

например:

```text
OPENAI_API_KEY=
LANGFUSE_SECRET_KEY=
DATA_DIR=data/sample
```

---

## 13. Типичный рабочий цикл

```bash
git status
git add -A
git status
git commit -m "Describe changes"
git push
```

Если перед началом работы нужно забрать свежие изменения:

```bash
git pull --rebase
```

Тогда полный цикл:

```bash
git pull --rebase
# работа с файлами
git status
git add -A
git commit -m "Describe changes"
git push
```

---

## 14. Если что-то пошло не так

Первое заклинание почти всегда:

```bash
git status
```

Не угадывать, не паниковать, не нажимать случайные команды.

Сначала прочитать, что именно Git считает текущим состоянием.

---

## Мини-шпаргалка

```text
ГДЕ Я?
pwd
git branch

ЧТО ПРОИСХОДИТ?
git status
git diff

ДОБАВИТЬ:
git add .
git add -A

ЗАКОММИТИТЬ:
git commit -m "message"

ОТПРАВИТЬ:
git push

ЗАБРАТЬ:
git pull --rebase

ИСТОРИЯ:
git log --oneline

ВЕТКИ:
git branch -a
```

---

## Мантра

```text
git status
git add -A
git commit -m "..."
git push
```

**Git не читает мысли.**
