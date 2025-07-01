import torch
import time

def create_matrices() -> dict:
    """
    Создает большие матрицы заданных размеров, заполненные случайными числами.
    
    Возвращает словарь с матрицами на CPU и GPU (если доступно)
    """
    matrices = {
        '64x1024x1024': torch.randn(64, 1024, 1024),
        '128x512x512': torch.randn(128, 512, 512),
        '256x256x256': torch.randn(256, 256, 256),
    }
    
    # Создаем копии на GPU если доступно
    if torch.cuda.is_available():
        matrices_gpu = {k: v.cuda() for k, v in matrices.items()}
    else:
        matrices_gpu = None
        
    return {'cpu': matrices, 'gpu': matrices_gpu}


def measure_time(device, func, *args, **kwargs) -> float | None:
    """
    Измеряет время выполнения функции на указанном устройстве (CPU/GPU).
    
    Аргументы:
        1) device (str): 'cpu' или 'gpu'
        2) func (callable): Функция для измерения
        3) *args: Аргументы функции
        4) **kwargs: Ключевые аргументы функции
        
    Возвращает время выполнения в миллисекундах
    """
    if device == 'cpu':
        start = time.time()
        func(*args, **kwargs)
        return (time.time() - start) * 1000  # мс
    
    elif device == 'gpu' and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        start_event.record()
        func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize() 
        return start_event.elapsed_time(end_event)  
    else:
        return None


def compare_operations(matrices) -> dict:
    """
    Сравнивает производительность операций на CPU и GPU.
    
    Аргументы:
        matrices (dict): Словарь с матрицами на CPU и GPU
        
    Возвращает результаты сравнения для каждой операции и размера матрицы
    """
    results = {}
    
    operations = {
        'Матричное умножение': lambda x: torch.matmul(x, x),
        'Поэлементное сложение': lambda x: x + x,
        'Поэлементное умножение': lambda x: x * x,
        'Транспонирование': lambda x: x.transpose(-2, -1),
        'Сумма всех элементов': lambda x: x.sum(),
    }
    
    for name, matrix in matrices['cpu'].items():
        results[name] = {}
        
        for op_name, op_func in operations.items():
            # Измеряем на CPU
            cpu_time = measure_time('cpu', op_func, matrix)
            
            # Измеряем на GPU если доступно
            if matrices['gpu'] is not None:
                gpu_time = measure_time('gpu', op_func, matrices['gpu'][name])
                speedup = cpu_time / gpu_time if gpu_time else None
            else:
                gpu_time = None
                speedup = None
                
            results[name][op_name] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
    
    return results


def print_results(results) -> None:
    """
    Печатает результаты сравнения в табличном виде.
    
    Аргументы:
        results (dict): Результаты сравнения операций
    """
    for matrix_name, matrix_results in results.items():
        print(f"\nРезультаты для матрицы {matrix_name}:")
        print("-" * 70)
        print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
        print("-" * 70)
        
        for op_name, op_result in matrix_results.items():
            cpu_time = f"{op_result['cpu_time']:.1f}" if op_result['cpu_time'] is not None else "N/A"
            gpu_time = f"{op_result['gpu_time']:.1f}" if op_result['gpu_time'] is not None else "N/A"
            speedup = f"{op_result['speedup']:.1f}x" if op_result['speedup'] is not None else "N/A"
            
            print(f"{op_name:<25} | {cpu_time:<10} | {gpu_time:<10} | {speedup:<10}")


def main():
    """
    Основная функция

    Вызывает все функции-задания из этого файла

    Выводы: 
    1)Операции матричного умножения и суммы обычно получают наибольшее ускорение
    2) Простые операции (сложение, умножение) могут иметь меньшее ускорение из-за накладных расходов
    3) Большие матрицы обычно лучшее ускоряются
    4) Передача данных между CPU и GPU требует времени, это может снизить производительность при большом кол-ве данных
    """
    print("Подготовка данных...")
    matrices = create_matrices()
    
    print("Сравнение операций...")
    results = compare_operations(matrices)
    
    print("\nРезультаты сравнения производительности CPU vs CUDA:")
    print_results(results)
    

if __name__ == "__main__":
    main()