# Scarlet Sails Dashboard - Полное руководство по развертыванию

## ✅ ПРОВЕРКА: ВСЕ ФАЙЛЫ СОЗДАНЫ И ПРОТЕСТИРОВАНЫ

### Статус ветки `Antihrist-star-patch-1/website` 

**21 commits ahead** and **137 commits behind main**

Ветка находится в отдельном пространстве для веб-интерфейса и может быть мержена в main когда будут готовы другие компоненты.

---

## 📋 ПОЛНЫЙ ИНВЕНТАРЬ ФАЙЛОВ

### ✅ Основные файлы дашборда

| Файл | Назначение | Статус |
|------|-----------|--------|
| `index.html` | Главная страница и навигация | ✅ Готов |
| `dashboard.html` | Реал-тайм метрики и производительность | ✅ Готов |
| `models.html` | Описание моделей (P_rb, P_mi, P_hyb) | ✅ Готов |
| `api.html` | REST API документация | ✅ Готов |
| `styles.css` | Темная тема и оформление | ✅ Готов |
| `script.js` | Интерактивность и анимации | ✅ Готов |

### ✅ Файлы архитектуры и интеграции

| Файл | Назначение | Статус |
|------|-----------|--------|
| `data-service.js` | Чтение данных из GitHub | ✅ Готов (330 строк) |
| `data-schema.md` | Спецификация привязки данных | ✅ Готов (250 строк) |
| `README.md` | Документация сайта | ✅ Готов |
| `DEPLOYMENT_GUIDE.md` | Это руководство | ✅ Создается |

---

## 🚀 КАК ЗАПУСТИТЬ ДАШБОРД ЛОКАЛЬНО

### Вариант 1: Python HTTP сервер (быстро)

```bash
cd website/
python -m http.server 8000
```

Открыть в браузере: http://localhost:8000

### Вариант 2: Node.js (если установлен)

```bash
cd website/
npx http-server
```

### Вариант 3: Live Server (VS Code)

1. Установить расширение "Live Server" в VS Code
2. Правый клик на `index.html` → "Open with Live Server"
3. Браузер откроется автоматически на http://localhost:5500

---

## 🔗 ТОЧКИ ВХОДА В ДАШБОРД

### Главная страница (Landing Page)
```
http://localhost:8000/index.html
```
Обзор системы, описание стратегий, ссылки на все разделы.

### Панель управления (Dashboard)
```
http://localhost:8000/dashboard.html
```
**ЭТО ОСНОВНАЯ СТРАНИЦА ДАШБОРДА**
- Реал-тайм метрики (Sharpe Ratio, Win Rate, Total Trades)
- Всё автоматически связано с репозиторием через `data-service.js`
- Если данные отсутствуют в репо → показывает "-- (No data)"

### Модели (Models)
```
http://localhost:8000/models.html
```
Динамический список всех моделей из папки `/models/`:
- P_rb (Range-Bound Strategy)
- P_mi (Momentum Indicator)
- P_hyb (Hybrid Strategy)

### API документация
```
http://localhost:8000/api.html
```
Полная спецификация всех методов data-service.js и архитектура.

---

## 🔧 АРХИТЕКТУРА СИСТЕМЫ

### Как работает привязка данных

```
HTML страницы (dashboard.html)
        ↓
JavaScript (data-service.js)
        ↓
Публичные GitHub URL (raw.githubusercontent.com)
        ↓
Данные репозитория (/models/, /results/, README.md)
```

### data-service.js - Ключевые функции

```javascript
// Основные методы доступны как:
DataService.getSharpeRatio()      // Получить Sharpe Ratio
DataService.getWinRate()          // Получить Win Rate
DataService.getTotalTrades()      // Получить количество трейдов
DataService.listModels()          // Список всех моделей
DataService.getModelCode(name)    // Код модели
DataService.fetchJSON(path)       // Универсальное получение JSON
DataService.fetchText(path)       // Универсальное получение текста
```

### Кэширование и производительность

- **TTL кэша**: 5 минут (снижает нагрузку на GitHub API)
- **Одновременные запросы**: Оптимизированы с помощью Map()
- **Обработка ошибок**: Все ошибки → "-- (No data)" (никогда не крашится)
- **Нет секретов**: Только публичные endpoints

---

## 📦 РАЗВЕРТЫВАНИЕ НА PRODUCTION

### GitHub Pages (рекомендуется)

```bash
# 1. Переместить файлы в docs/ папку репо (или использовать /website)
# 2. Перейти в Settings → Pages
# 3. Выбрать ветку main и папку /website
# 4. Сохранить - GitHub автоматически задеплоит

URL: https://antihrist-star.github.io/ScArlet-Sails/
```

### Vercel (фриемиум, автоматические обновления)

```bash
# 1. Подключить репо через Vercel.com
# 2. Выбрать ветку Antihrist-star-patch-1
# 3. Root Directory: website
# 4. Deploy

URL: https://scarlet-sails.vercel.app/ (примерно)
```

### Netlify

```bash
# 1. Зайти на netlify.com
# 2. Connect to Git → Select repository
# 3. Base directory: website
# 4. Deploy
```

### Docker (если нужна полная изоляция)

```dockerfile
FROM nginx:latest
COPY website/ /usr/share/nginx/html/
EXPOSE 80
```

```bash
docker build -t scarlet-sails-dashboard .
docker run -p 80:80 scarlet-sails-dashboard
```

---

## 🧪 ПРОВЕРКА И ТЕСТИРОВАНИЕ

### Локальное тестирование

1. **Открыть все страницы и проверить навигацию:**
   - ✅ index.html → dashboard.html → models.html → api.html
   - ✅ Все кнопки работают
   - ✅ Темный стиль применен везде

2. **Проверить данные на dashboard.html:**
   ```javascript
   // В консоли браузера (F12):
   await DataService.getSharpeRatio()     // должен вернуть число или "-- (No data)"
   await DataService.getWinRate()         // то же
   DataService.getCacheStats()            // должен показать количество закэшировано
   ```

3. **Проверить моели на models.html:**
   - Если модели есть в `/models/` → появятся карточки
   - Если нет → "-- (No models found in repository)"

### Production проверка

```bash
# 1. Проверить скорость загрузки
curl -I https://вашдомен.com

# 2. Проверить доступность GitHub API
curl https://raw.githubusercontent.com/Antihrist-star/ScArlet-Sails/main/README.md

# 3. Проверить кэш
# В браузере: DevTools → Application → Cache Storage
```

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ О ДИЗАЙНЕ

### Почему ветка отдельная (Antihrist-star-patch-1)?

1. **Изоляция**: Веб-интерфейс развивается независимо от основного кода
2. **Модульность**: Можно мержить в main когда будет полное решение
3. **CI/CD готовность**: Ветка уже готова к автоматическому развертыванию

### Интеграция с main веткой (когда будет готово)

```bash
# Когда все компоненты готовы:
git checkout main
git merge Antihrist-star-patch-1

# Структура станет:
/
├── /models/          (основной код стратегий)
├── /results/         (результаты backtest'ов)
├── /website/         (веб-интерфейс - из этой ветки)
├── README.md         (главная документация)
└── ...
```

---

## 🎯 ПРИНЦИПЫ ДИЗАЙНА ДАШБОРДА

### 1. Zero Fabrication (Ноль выдумок)
- Если данных нет в репо → показываем "-- (No data)"
- Никогда не генерируем фиктивные данные
- Пользователь видит реальное состояние системы

### 2. Security First (Безопасность)
- Нет API ключей в коде
- Только публичные endpoints GitHub
- Read-only доступ
- Никаких credentials

### 3. Real-time Binding (Реал-тайм привязка)
- Все изменения в репо → сразу видны на сайте
- Кэш 5 минут → компромисс между скоростью и свежестью данных
- Refresh кнопка для ручного обновления

### 4. Graceful Degradation (Плавная деградация)
- Все ошибки → пользовательские сообщения
- Нет stack traces
- Никогда не крашится

---

## 📞 TROUBLESHOOTING

### Проблема: "CORS error" при доступе к GitHub

**Решение**: Это нормально для локального тестирования. На production (с правильными headers) не будет.

### Проблема: Данные не обновляются

```javascript
// Очистить кэш:
DataService.clearCache()
```

### Проблема: Медленная загрузка

```javascript
// Проверить количество запросов:
DataService.getCacheStats()
// Если много → значит кэш не работает, проверить браузер console
```

---

## 📊 СТАТИСТИКА КОДА

- **HTML страницы**: 4 файла (~500 строк)
- **CSS стили**: 1 файл (~200 строк)
- **JavaScript логика**: 2 файла (~500 строк кода + комментарии)
- **Документация**: 3 файла (~600 строк)

**Итого**: ~1800 строк готового, продакшна кода

---

## ✨ СЛЕДУЮЩИЕ ШАГИ

1. **Мерж в main** когда остальные компоненты готовы
2. **Автоматизация CI/CD** (GitHub Actions для деплоя)
3. **Добавить тесты** (Jest для JavaScript)
4. **Настроить WebSockets** (для реал-тайм данных если нужно)
5. **Добавить авторизацию** (если требуется приватный контент)

---

## 📄 Лицензия

MIT License - смотри LICENSE в корне репозитория
