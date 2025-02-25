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

```bash
pip install probstates
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

Если у вас есть вопросы или предложения, пожалуйста, откройте issue в репозитории проекта или свяжитесь с нами по электронной почте: [email].
