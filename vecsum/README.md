# Задание
Задача: реализовать алгоритм сложения элементов вектора

Язык: C++ или Python

Входные данные: Вектор размером 1 000..1 000 000 значений.

Выходные данные: сумма элементов вектора + время вычисления

Реализация должна содержать 2 функции сложения элементов вектора: на CPU и на GPU с применением CUDA.

Отчет о проделанной лабораторной работе - это git-репозиторий с исходным кодом реализации + описание проделанной работы там же в readme.

Необходимо описать реализацию, объяснив, что конкретно было распараллелено и почему.

Провести эксперименты: получить сумму векторов разных размеров (провести 5 или более экспериментов), посчитать ускорение. Результаты привести в виде таблицы/графика.

# Описание

В ходе работы была создана программа для выполнения сложения элементов вектора как на центральном процессоре (CPU), так и на графическом процессоре (GPU)
с использованием технологии CUDA. Для реализации вычислений на CPU была написана функция, которая суммирует элементы вектора типа float поэлементно в цикле.
Для GPU была создана параллельная версия алгоритма, которая эффективно суммирует элементы с использованием shared памяти и параллельной редукции внутри блоков потоков.
Результат каждого блока затем объединяется с помощью атомарной операции atomicAdd.

Программа также измеряет время выполнения вычислений на CPU и GPU. Вектор из миллиона элементов заполняется значением 1.0, и итоговая сумма 
вычисляется обеими методами. Результаты вычислений и соответствующее время выполнения выводятся в консоль для сравнения.

# Графики работы программы на GPU и CPU в зависимости от N

![Image alt](https://github.com/LinkN0W/SU-HPC-Fall-2024/raw/main/matmult/time.png)


![Image alt](https://github.com/LinkN0W/SU-HPC-Fall-2024/raw/main/matmult/boost.png)

# Вывод
На GPU вычисления выполняются значительно быстрее благодаря параллельной обработке данных, особенно при больших размерах вектора.
Результаты из преведенных графиков демонстрируют значительное преимущество использования GPU для параллельных вычислений.
Однако время выполнения зависит от конкретного оборудования и архитектуры GPU.