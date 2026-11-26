# Аудит скрытой предвзятости в моделях кредитного скоринга (SHAP + LIME)

Репозиторий содержит программный эксперимент, таблицы прогонов, итоговые графики и текст научной статьи (НИТУ «МИСиС») по выявлению скрытой (косвенной) предвзятости в моделях кредитного скоринга методами объяснимого искусственного интеллекта **SHAP** и **LIME**.

## О проекте

Даже при исключении чувствительного признака $A$ из матрицы признаков модель машинного обучения может восстанавливать его влияние через коррелирующие признаки-прокси. В эксперименте на базе открытого датасета **UCI German Credit** (`credit-g`, чувствительный атрибут `personal_status` — индикатор `female`) моделируются **4 сценария × 5 повторных прогонов** с контролируемым внедрением истинных прокси-признаков $P^*$, маскированных признаков-приманок `proxy_decoy_*` (коррелируют с $A$, но обнуляются при обучении модели) и чистого шума:

- **Уровень 1** — оценка межгруппового диспаритета решений при стратификации по допустимым факторам риска $R$ (терцили `credit_amount` × `duration`): метрика $B_R(f)$, перестановочный порог $\delta$, а также `DP`, `EO_good`, `EO_bad` и `AUC`.
- **Уровень 2** — отбор кандидатов по взаимной информации (`MI` выше перестановочного порога $\varepsilon_{\text{MI}}$), расчёт локальных вкладов **SHAP** (`TreeSHAP`) и **LIME** (`LimeTabularExplainer`), сравнение распределений вкладов между группами $A = 0$ и $A = 1$ (критерии Колмогорова–Смирнова и Манна–Уитни с поправкой Бенджамини–Хохберга) и перекрёстная верификация **SHAP ∩ LIME**.

## Структура репозитория

```text
credit-scoring-bias-article/
├── run_audit_experiment.py        # Основной скрипт эксперимента (4 сценария × 5 прогонов)
├── requirements.txt               # Зафиксированные зависимости Python
├── audit_results.csv              # Метрики обнаружения прокси по методам, сценариям и seed
├── audit_level1.csv               # Метрики 1-го уровня (B_R, delta, DP, EO, AUC, разрыв SHAP)
├── fig1_precision_recall_fpr.png  # Precision, Recall (TPR) и FPR для 4 методов по сценариям
├── fig2_signed_shap_groups.png    # Топ-10 признаков по |SHAP| и средний знаковый SHAP по группам
├── fig3_level1_measurement.png    # Сравнение B_R(f), разности DP и порога delta по сценариям
├── article.tex                    # Исходный текст статьи (LaTeX, схема архитектуры на TikZ)
├── misisarticle.cls               # Класс оформления статьи НИТУ «МИСиС» (LPPL 1.3c)
├── Makefile                       # Цели сборки PDF статьи, примера и архива для Overleaf
├── example/                       # Эталонная статья шаблона МИСиС для проверки класса
├── .github/workflows/build.yml    # GitHub Actions для сборки PDF через Tectonic
├── LICENSE                        # Лицензия LPPL 1.3c для класса misisarticle.cls
└── CHANGELOG.md                   # История изменений LaTeX-шаблона
```

## Требования

- **Python:** рекомендуется **Python 3.12** (эксперимент проверен на `Python 3.12.3`; зафиксированные версии `numpy`, `scipy`, `xgboost`, `shap` требуют Python `>= 3.12`).
- **Доступ в интернет (при первом запуске):** скрипт загружает датасет `credit-g` (version 1) из OpenML с помощью `sklearn.datasets.fetch_openml` и кэширует его локально средствами `scikit-learn`.
- **Системные зависимости:**
  - Для Python-эксперимента графический сервер не нужен (используется non-interactive бэкенд `matplotlib.use("Agg")`); на Linux для `xgboost` при необходимости должна присутствовать библиотека `libgomp1` (обычно уже установлена).
  - Для сборки PDF статьи из `article.tex` требуется **Tectonic** (`0.17.0`+) или **XeLaTeX** (TeX Live / MacTeX / MiKTeX), а также шрифты `Times New Roman` и `Courier New` (при их отсутствии `misisarticle.cls` автоматически переключается на `TeX Gyre Termes`).

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/m2307203-creator/credit-scoring-bias-article.git
cd credit-scoring-bias-article
```

### 2. Создание и активация виртуального окружения

**Linux / WSL / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Запуск эксперимента

Запускайте скрипт из корня репозитория (по умолчанию используются параметры статьи: 5 прогонов, `seed=2026`, модель `xgb`):

```bash
python run_audit_experiment.py
```

При необходимости параметры можно переопределить через аргументы командной строки:

```bash
python run_audit_experiment.py --repeats 5 --seed 2026 --model xgb
```

- `--repeats` — количество прогонов на сценарий (по умолчанию `5`);
- `--seed` — базовое значение генератора случайных чисел (по умолчанию `2026`);
- `--model {xgb,gbc}` — классификатор (`xgb` — `XGBClassifier`, `gbc` — `GradientBoostingClassifier`).

После завершения работы скрипт выводит сводные таблицы в консоль и сохраняет **в корень репозитория (текущий рабочий каталог)** следующие файлы:
- `audit_results.csv`
- `audit_level1.csv`
- `fig1_precision_recall_fpr.png`
- `fig2_signed_shap_groups.png`
- `fig3_level1_measurement.png`

## Воспроизводимость и результаты

В коде `run_audit_experiment.py` зафиксированы все параметры эксперимента:

- **Начальные состояния (`seeds`):** `2026, 2027, 2028, 2029, 2030` (`seed + r` для `r = 0..4`).
- **Разбиение и модель:** `TEST_SIZE = 0.40` (стратифицированное по целевой переменной), `XGBClassifier` (`n_estimators=200`, `max_depth=3`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`).
- **Порог одобрения кредита:** `TAU = 0.30` (одобрение при прогнозируемой вероятности дефолта $< 0.30$).
- **Параметры аудита:** `ALPHA = 0.05` (критерии Колмогорова–Смирнова и Манна–Уитни с поправкой Бенджамини–Хохберга), `EFFECT_MIN = 0.10`, `GAIN_MIN = 0.01`, `BACKGROUND_SIZE = 100` (TreeSHAP), `LIME_ROWS_PER_GROUP = 100`, `LIME_SAMPLES = 600`, `MI_NEIGHBORS = 5`, `N_MI_NULL = 50`, `N_DELTA_NULL = 200`, `MIN_CELL = 5`, `PROXY_LOGIT_SD = 0.8`, `DECOY_RHO = 0.7`.
- **Конфигурация сценариев (`SCENARIOS`):**
  - `Контроль`: `0` прокси, `0` приманок, `2` шумовых признака;
  - `Слабая связь`: `2` прокси (`rho = 0.30`), `2` приманки, `2` шумовых признака;
  - `Умеренная связь`: `3` прокси (`rho = 0.55`), `2` приманки, `2` шумовых признака;
  - `Сильная связь`: `3` прокси (`rho = 0.85`), `2` приманки, `2` шумовых признака.

Готовые результаты уже сохранены в корне репозитория:

- `audit_results.csv` (80 строк данных: 4 сценария × 5 прогонов × 4 метода) — столбцы: `scenario`, `seed`, `method` (`Корреляция (MI)`, `SHAP`, `LIME`, `SHAP ∩ LIME`), `precision`, `recall`, `fpr`, `n_flagged`, `flagged`.
- `audit_level1.csv` (20 строк данных: 4 сценария × 5 прогонов) — столбцы: `B_R`, `delta`, `DP`, `EO_good`, `EO_bad`, `signal`, `auc`, `shap_additivity_gap`, `scenario`, `seed`, `P_star`, `eps_MI`, `max_decoy_gain`.
- Итоговые графики: `fig1_precision_recall_fpr.png`, `fig2_signed_shap_groups.png`, `fig3_level1_measurement.png`.

По сохранённым прогонам качество моделей составляет `AUC = 0.756–0.841`, максимальный разрыв аддитивности TreeSHAP не превышает `4.29e-6`, `max_decoy_gain = 0.0`, а в сценарии «Сильная связь» метод `SHAP ∩ LIME` обеспечивает `Precision = 1.000`, `Recall = 1.000` и `FPR = 0.000`.

## Сборка статьи (LaTeX)

Для компиляции PDF из `article.tex` с использованием `Makefile` и Tectonic:

```bash
make article
# Результат: build/article.pdf
```

Прямая сборка без `Makefile`:

```bash
tectonic article.tex
# либо:
xelatex article.tex && xelatex article.tex
```

Дополнительные цели `Makefile`:
- `make dist` — сборка архива `dist/misis-article.zip` для загрузки в Overleaf (компилятор **XeLaTeX**, главный файл `article.tex`);
- `make example` — сборка эталонной статьи шаблона (`example/build/example.pdf`);
- `make clean` — очистка каталогов сборки и временных файлов LaTeX.

Класс `misisarticle.cls` распространяется под лицензией [LPPL 1.3c](https://www.latex-project.org/lppl/) (см. `LICENSE`).
