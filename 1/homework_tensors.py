import torch 


def task_1_1() -> None:
    """
    1.1 Создание тензоров

    Создание и вывод следующих тензоров (РАЗМЕРНОСТЬ:   ЗАПОЛНЕНИЕ):
        1) 3x4:     0-1
        2) 2x3x4:   0
        3) 5x5:     1
        4) 4x4:     0-15    <- Данный тензор реализован с использованием ф-ии reshape
    
    Вывод каждого тензора происходит через print()
    """
    # Тензор 3x4 со случайными числами от 0 до 1
    tensor_a = torch.rand(3, 4)
    
    # Тензор 2x3x4, заполненный нулями
    tensor_b = torch.zeros(2, 3, 4)
    
    # Тензор 5x5, заполненный единицами
    tensor_c = torch.ones(5, 5)
    
    # Тензор 4x4 с числами от 0 до 15
    tensor_d = torch.arange(16).reshape(4, 4)
    
    print(' ' * 50 + "1.1 Создание тензоров:")
    print("Тензор 3x4 случайных чисел:\n", tensor_a)
    print("\nТензор 2x3x4 нулей:\n", tensor_b)
    print("\nТензор 5x5 единиц:\n", tensor_c)
    print("\nТензор 4x4 от 0 до 15:\n", tensor_d)    
    print(f"\n{'=' * 100}\n")


def task_1_2() -> None:
    """
    1.2 Операции с тензорами

    Создание изначальных тензеров A(3x4), B(4x3) и манипуляции с ними:
        1) транспонирование тензерa A
        2) Матричное умножение тензерa A и B
        3) Поэлементное умножение A и транспонированного B
        4) Сумма всех элементов тензора A

    Вывод каждого тензора происходит через print()
    """
    tensor_a = torch.rand(3, 4)
    tensor_b = torch.rand(4, 3)
    
    # Транспонирование тензора A
    a_transposed = tensor_a.t()
    
    # Матричное умножение A и B
    matmul = torch.matmul(tensor_a, tensor_b)
    
    # Поэлементное умножение A и транспонированного B
    b_transposed = tensor_b.t()
    elementwise = tensor_a * b_transposed[:tensor_a.size(0), :tensor_a.size(1)]
    
    # Сумма всех элементов тензора A
    sum_a = tensor_a.sum()
    print(' ' * 50 + "1.2 Операции с тензорами:")
    print("Тензор A:\n", tensor_a)
    print("\nТензор B:\n", tensor_b)
    print("\nТранспонированный A:\n", a_transposed)
    print("\nМатричное умножение A и B:\n", matmul)
    print("\nПоэлементное умножение A и транспонированного B:\n", elementwise)
    print("\nСумма всех элементов A:", sum_a.item())
    print(f"\n{'=' * 100}\n")


def task_1_3() -> None:
    """
    1.3 Индексация и срезы

    Создание тензора(5х5х5) с последующим извлечением тензоров:
        1) Первая строка 
        2) Последний столбец 
        3) Подматрица 2x2 из центра 
        4) Все элементы с четными индексами

    Вывод каждого тензора происходит через print()
    """
    tensor = torch.rand(5, 5, 5)
    
    # Первая строка 
    first_row = tensor[0, :, :]
    
    # Последний столбец 
    last_column = tensor[:, :, -1]
    
    # Подматрица 2x2 из центра 
    center_submatrix = tensor[2:4, 2:4, 2:4]
    
    # Все элементы с четными индексами
    even_indices = tensor[::2, ::2, ::2]
    
    print(' ' * 50 + "1.3 Индексация и срезы:")
    print("Тензор 5x5x5:\n", tensor[:2, :2, :2])
    print("\nПервая строка:\n", first_row)
    print("\nПоследний столбец:\n", last_column)
    print("\nПодматрица 2x2x2 из центра:\n", center_submatrix)
    print("\nВсе элементы с четными индексами:\n", even_indices)
    print(f"\n{'=' * 100}\n")


def task_1_4() -> None:
    """
    1.4 Работа с формами

    Создание первоначального тензора размером в 24 элемента и изменение его формы на следующие форматы:
        1)    (2, 12)
        2)    (3, 8)
        3)    (4, 6)
        4)    (2, 3, 4)
        5)    (2, 2, 2, 3)

    Вывод каждого тензора происходит через print()
    """
    tensor = torch.arange(24)
    
    # Преобразуем в различные формы
    shapes = [
        (2, 12),
        (3, 8),
        (4, 6),
        (2, 3, 4),
        (2, 2, 2, 3)
    ]
    reshaped_tensors = []
    for shape in shapes:
        reshaped_tensors.append(tensor.reshape(shape))
    
    print(' ' * 50 + "1.4 Работа с формами:")
    print("Исходный тензор:\n", tensor)
    for i, (shape, reshaped) in enumerate(zip(shapes, reshaped_tensors)):
        print(f"\nФорма {shape}:\n", reshaped)
    print(f"\n{'=' * 100}\n")
    


def main() -> None:
    """
    Основная функция

    Вызывает все функции-задания из этого файла
    """
    task_1_1()
    task_1_2()
    task_1_3()
    task_1_4()  
    

if __name__ == "__main__":
    main()