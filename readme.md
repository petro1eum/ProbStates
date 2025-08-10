# ProbStates

![ProbStates Logo](https://example.com/probstates-logo.png)

**ProbStates** — это библиотека Python для работы с иерархией вероятностных состояний, представляющих собой концептуальный мост между классическими и квантовыми вычислениями.

## Уровни иерархии

Библиотека реализует пять уровней вероятностных состояний, каждый из которых является обобщением предыдущего:

1. **Классические биты** (`ClassicalBit`): Дискретные значения 0 или 1
2. **Вероятностные биты** (`ProbabilisticBit`): Вероятность p ∈ [0,1]
3. **P-биты** (`PBit`): Пара (p, s), где p ∈ [0,1], s ∈ {+1,-1}
4. **Фазовые состояния** (`PhaseState`): Пара (p, e^(iφ)), где p ∈ [0,1], φ ∈ [0,2π)
5. **Квантовые состояния** (`QuantumState`): Вектор |ψ⟩ в комплексном пространстве

## Установка

Установка из исходников (рекомендуется):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e .
# Для запуска графических примеров:
pip install matplotlib
```

Запуск примеров и тестов:

```bash
python examples.py      # генерирует изображения, выводит результаты сценариев
python run_tests.py     # прогонит все автотесты
```

## Быстрый старт

```python
import numpy as np
from probstates import ClassicalBit, ProbabilisticBit, PBit, PhaseState, QuantumState
from probstates import lift, project
import matplotlib.pyplot as plt
from probstates.visualization import visualize_state

# Создание состояний различных уровней
c_bit = ClassicalBit(1)                       # Классический бит = 1
p_bit = ProbabilisticBit(0.7)                 # Вероятностный бит = 0.7
pb_bit = PBit(0.6, -1)                        # P-бит = (0.6, -1)
ph_state = PhaseState(0.8, np.pi/4)           # Фазовое состояние = (0.8, e^(iπ/4))
q_state = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])  # Квантовое состояние = |+⟩

# Визуализация состояний
plt.figure(figsize=(8, 6))
visualize_state(ph_state)
plt.show()

# Операции над состояниями
bit1 = ClassicalBit(0)
bit2 = ClassicalBit(1)
result = bit1 | bit2  # Операция OR для классических битов
print(result)  # Выведет: ClassicalBit(1)

# Подъем и проекция состояний
prob_bit = lift(c_bit, 2)  # Подъем классического бита до вероятностного
print(prob_bit)  # Выведет: ProbabilisticBit(1.0000)

classic_bit = project(p_bit, 1)  # Проекция вероятностного бита на классический
print(classic_bit)  # Выведет: ClassicalBit(1)
```

## Функциональность

### Базовые операции для всех уровней

На каждом уровне определены три основные операции:

- **AND** (`&`): Аналог логического AND
- **OR** (`|`): Аналог логического OR
- **NOT** (`~`): Аналог логического NOT

### Переходы между уровнями

Библиотека предоставляет функции для подъема и проекции состояний:

- `lift(state, to_level)`: Поднимает состояние до указанного уровня
- `project(state, to_level)`: Проецирует состояние на указанный уровень

### Визуализация

Модуль `probstates.visualization` предоставляет функции для визуализации состояний и операций:

- `visualize_state(state)`: Визуализирует состояние любого уровня
- `visualize_operation(state1, state2, operation, result)`: Визуализирует операцию между состояниями
- `visualize_lifting(state, lifted_state)`: Визуализирует подъем состояния
- `visualize_projection(state, projected_state)`: Визуализирует проекцию состояния

### Энтропийные характеристики

Модуль `probstates.entropy` предоставляет функции для расчета энтропийных характеристик состояний:

- **Энтропия Шеннона** для вероятностных битов (уровень 2)
- **Обобщенная энтропия** для p-битов (уровень 3)
- **Квазиквантовая энтропия** для фазовых состояний (уровень 4)
- **Энтропия фон Неймана** для квантовых состояний (уровень 5)

```python
from probstates.entropy import calculate_entropy, information_loss

# Расчет энтропии состояния
prob_bit = ProbabilisticBit(0.7)
entropy = calculate_entropy(prob_bit)
print(f"Энтропия: {entropy}")  # Выведет: Энтропия: 0.8813

# Анализ потери информации при проекции
phase_state = PhaseState(0.7, np.pi/4)
pbit = project(phase_state, 3)
loss = information_loss(phase_state, pbit)
print(f"Потеря информации: {loss}")  # Выведет примерную оценку потери
```

Подробное описание теоретических основ энтропийных характеристик доступно в файле `entropy_theory.md`.

### Фазовый регистр и алгоритм Дойча–Йожи (раздел §5)

Доступны утилиты для регистров уровня 4 и прототипа алгоритма Дойча–Йожи:

- `PhaseRegister.uniform(n)` — равномерное состояние размера 2^n
- `PhaseRegister.apply_oracle(f)` — фазовый оракул α_x ← α_x·(−1)^{f(x)}
- `PhaseRegister.hadamard_all()` — FWHT (аналог H^{⊗n})
- `deutsch_jozsa(oracle, n)` — различение константной/сбалансированной функции

```python
from probstates import deutsch_jozsa

n = 3
f_const = lambda x: 1
f_bal   = lambda x: bin(x).count("1") & 1  # чётность

print(deutsch_jozsa(f_const, n))  # ('constant', ~1.0)
print(deutsch_jozsa(f_bal, n))    # ('balanced', ~0.0)
```

Переключение режима операции ⊕₄ (фазовое OR):

```python
from probstates import set_phase_or_mode
set_phase_or_mode('quant')                     # по умолчанию: сумма амплитуд
set_phase_or_mode('opt', delta_phi=3.14159/2)  # оптимизированное правило из статьи
set_phase_or_mode('norm')                      # F = min(1, F_quant)
set_phase_or_mode('weight')                    # F = p1⊕2p2 + (2√(p1p2)cosΔφ)/(1+max(p1,p2))
```

Пользовательская политика для ⊕₄:

```python
from probstates import set_phase_or_custom

def my_oplus_policy(p1, phi1, p2, phi2):
    # пример: берём максимум по вероятности и среднюю фазу
    return (max(p1, p2), (phi1 + phi2) / 2)

set_phase_or_custom(my_oplus_policy)
set_phase_or_mode('custom')
```

## Примеры

### Интерференция на разных уровнях

```python
from probstates import PBit, PhaseState, QuantumState
import numpy as np

# Уровень 3: P-биты с противоположными полярностями
pb1 = PBit(0.5, +1)
pb2 = PBit(0.5, -1)
pb_result = pb1 | pb2  # Деструктивная интерференция
print(f"P-биты: {pb1} OR {pb2} = {pb_result}")

# Уровень 4: Фазовые состояния с противофазой
ph1 = PhaseState(0.5, 0)
ph2 = PhaseState(0.5, np.pi)
ph_result = ph1 | ph2  # Полное гашение
print(f"Фазовые состояния: {ph1} OR {ph2} = {ph_result}")

# Уровень 5: Квантовые состояния
q1 = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])    # |+⟩
q2 = QuantumState([1/np.sqrt(2), -1/np.sqrt(2)])   # |-⟩
q_result = q1 | q2                                # Интерференция
print(f"Квантовые состояния: {q1} OR {q2} = {q_result}")
```

### Моделирование подбрасывания монеты

```python
from probstates import ClassicalBit, ProbabilisticBit, PhaseState, QuantumState
import numpy as np

# Фиксируем seed для воспроизводимости
rng = np.random.RandomState(42)

# Детерминированная монета (уровень 1)
deterministic_coin = ClassicalBit(1)  # Всегда орел

# Вероятностная монета (уровень 2)
prob_coin = ProbabilisticBit(0.5)  # Честная монета

# Генерируем 10 результатов подбрасывания
results = [prob_coin.sample(rng) for _ in range(10)]
print(f"Вероятностные подбрасывания: {results}")

# Квантовая монета (уровень 5)
q_coin = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])  # Равная суперпозиция

# Генерируем 10 результатов измерения
q_results = [q_coin.measure(rng) for _ in range(10)]
print(f"Квантовые измерения: {q_results}")
```

### Анализ энтропийных характеристик

```python
from probstates import ProbabilisticBit, PBit, PhaseState
from probstates.entropy import calculate_entropy
import numpy as np
import matplotlib.pyplot as plt

# Анализ энтропии для разных вероятностей
probabilities = np.linspace(0.01, 0.99, 99)
shannon_entropies = []
pbit_entropies = []
phase_entropies = []

for p in probabilities:
    # Создаем состояния разных уровней
    prob_bit = ProbabilisticBit(p)
    pbit = PBit(p, -1)  # Отрицательная полярность
    phase_state = PhaseState(p, np.pi/4)
    
    # Рассчитываем энтропии
    shannon_entropies.append(calculate_entropy(prob_bit))
    pbit_entropies.append(calculate_entropy(pbit))
    phase_entropies.append(calculate_entropy(phase_state))

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(probabilities, shannon_entropies, label='Шеннон (уровень 2)')
plt.plot(probabilities, pbit_entropies, label='P-бит (уровень 3)')
plt.plot(probabilities, phase_entropies, label='Фазовое состояние (уровень 4)')
plt.xlabel('Вероятность p')
plt.ylabel('Энтропия')
plt.title('Энтропийные характеристики разных уровней')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Прикладные сценарии

Риск‑скоринг (объединение независимых факторов риска):

```python
from probstates import ProbabilisticBit
from probstates.entropy import calculate_entropy

a = ProbabilisticBit(0.12)
b = ProbabilisticBit(0.08)
c = ProbabilisticBit(0.18)
combined = a | b | c
print("P(any)=", combined.probability)
print("H(any)=", calculate_entropy(combined))
```

Слияние датчиков (учёт согласованности/противофазы):

```python
import numpy as np
from probstates import PhaseState

s1 = PhaseState(0.6, 0.0)
s2_aligned = PhaseState(0.6, 0.0)
s2_opposed = PhaseState(0.6, np.pi)

print((s1 | s2_aligned).probability)  # конструктивная интерференция
print((s1 | s2_opposed).probability)  # деструктивная интерференция
```

A/B‑тест (расхождение и лифт):

```python
from probstates.entropy import shannon_entropy, kl_divergence

pA, pB = 0.12, 0.145
print("H(A)=", shannon_entropy(pA), "H(B)=", shannon_entropy(pB))
print("D_KL(A||B)=", kl_divergence(pA, pB))
print("D_KL(B||A)=", kl_divergence(pB, pA))
print("Lift=", (pB - pA)/pA)
```

### Уровень 4: когерентность и шум

```python
from probstates import PhaseState
from probstates import coherence_l1, phase_drift, amp_damp
from probstates.coherence import dephase

s = PhaseState(0.25, 0.0)
print("C_l1=", coherence_l1(s))
s = phase_drift(s, 0.3)
s = amp_damp(s, 0.2)
s = dephase(s, sigma_phi=0.1)
```

### Фазовый регистр: тензоризация и POVM

```python
from probstates import PhaseRegister
import numpy as np

a = PhaseRegister.uniform(1)
b = PhaseRegister.uniform(1)
ab = a.tensor(b)                     # тензорное произведение
p0, p1 = ab.partial_measure(0)       # маргинальные вероятности старшего разряда

# Диагональная POVM из двух эффектов (пополам)
N = 1 << ab.num_qubits
E0 = np.zeros(N); E0[:N//2] = 1.0
E1 = 1.0 - E0
probs, posts = ab.povm_measure([E0, E1])
print(probs)  # ~[0.5, 0.5] для равномерного регистра
```

### Ноутбук: режимы ⊕₄ под шумом

См. `examples_oplus4_noise.ipynb`: сравнение 'quant'/'norm'/'weight'/'opt' по F(Δφ) и под шумами `dephase`/`amp_damp`.

### Практические кейсы

См. файл `applied_usage.md` — риск‑менеджмент, сенсоры (sensor fusion), A/B‑анализ с готовыми код‑фрагментами и рекомендациями по выбору режима ⊕₄.

## Теоретические основы

Данная библиотека основана на формальной теории иерархии вероятностных состояний, которая представляет собой математический мост между классической булевой логикой и квантовой механикой.

Каждый уровень добавляет новую "степень свободы" и расширяет возможности предыдущего:

- **Уровень 1** добавляет **дискретность** (классические биты)
- **Уровень 2** добавляет **непрерывность** (вероятностные биты)
- **Уровень 3** добавляет **дискретную фазу** (P-биты)
- **Уровень 4** добавляет **непрерывную фазу** (фазовые состояния)
- **Уровень 5** добавляет **линейную суперпозицию и запутанность** (квантовые состояния)

## Лицензия

MIT License

## Благодарности

Эта библиотека основана на теоретическом формализме, разработанном в [ссылка на работу или автора].

## Контакты

Если у вас есть вопросы или предложения, пожалуйста, откройте issue в репозитории проекта или свяжитесь с нами по электронной почте: [edcherednik@gmail.com].
