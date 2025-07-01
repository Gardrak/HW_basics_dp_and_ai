import torch


def task_2_1() -> None:
    """
    2.1 Простые вычисления с градиентами

    1) Создание тензоров x, y, z с requires_grad=True
    2) Вычисляет функцию:
        
        f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z ;

       и находит градиенты по всем переменным

    Вывод каждого тензора происходит через print()
    """
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)
    
    # Вычисляем функцию
    f = x**2 + y**2 + z**2 + 2*x*y*z
    
    # Вычисляем градиенты
    f.backward()
    
    # Аналитические производные
    df_dx_analytical = 2*x + 2*y*z
    df_dy_analytical = 2*y + 2*x*z
    df_dz_analytical = 2*z + 2*x*y
    
    print(' ' * 50 + "2.1 Простые вычисления с градиентами:")
    print(f"f(x,y,z) = {f.item():.2f}")
    print(f"df/dx: autograd= {x.grad.item():.2f} | analytical= {df_dx_analytical.item():.2f}")
    print(f"df/dy: autograd= {y.grad.item():.2f} | analytical= {df_dy_analytical.item():.2f}")
    print(f"df/dz: autograd= {z.grad.item():.2f} | analytical= {df_dz_analytical.item():.2f}")
    print(f"\n{'=' * 100}\n")


def task_2_2() -> None:
    """
    2.2 Градиент функции потерь MSE(Mean Squared Error)

    Реализует MSE loss и находит градиенты по w и b

    Вывод каждого тензора происходит через print()
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 4.0, 6.0])
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    
    # Прямой проход
    y_pred = w * x + b # линейная функция
    mse = torch.mean((y_pred - y_true)**2)
    
    # Обратный проход
    mse.backward()
    
    # Аналитические производные
    n = len(x)
    dw_analytical = 2 * torch.sum((y_pred - y_true) * x) / n
    db_analytical = 2 * torch.sum(y_pred - y_true) / n
    
    print(' ' * 50 + "2.2 Градиент функции потерь MSE:")
    print(f"MSE: {mse.item():.2f}")
    print(f"dw: autograd={w.grad.item():.2f} | analytical={dw_analytical.item():.2f}")
    print(f"db: autograd={b.grad.item():.2f} | analytical={db_analytical.item():.2f}")
    print(f"\n{'=' * 100}\n")


def task_2_3() -> None:
    """
    2.3 Цепное правило

        1) Реализует составную функцию f(x) = sin(x^2 + 1)
        2) Находит градиент df/dx
        3) Cверяясь с результатом через torch.autograd.grad

    Вывод каждого тензора происходит через print()
    """
    x = torch.tensor(2.0, requires_grad=True)
    
    # Вычисляем функцию
    f = torch.sin(x**2 + 1)
    
    # Вычисляем градиент
    grad_autograd = torch.autograd.grad(f, x, retain_graph=False)[0]
    
    # Аналитическая производная
    df_dx_analytical = torch.cos(x**2 + 1) * 2 * x
    
    print(' ' * 50 + "2.3 Цепное правило:")
    print(f"f(x) = {f.item():.2f}")
    print(f"df/dx: autograd={grad_autograd.item():.2f} | analytical={df_dx_analytical.item():.2f}")
    print(f"\n{'=' * 100}\n")


def main() -> None:
    """
    Основная функция

    Вызывает все функции-задания из этого файла
    """
    task_2_1()
    task_2_2()
    task_2_3()


if __name__ == "__main__":
    main()