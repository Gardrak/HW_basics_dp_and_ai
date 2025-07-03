import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import Dataset, DataLoader
from homework_datasets import CustomCSVDataset, evaluate_model
from tqdm import tqdm
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




class PolyDataset(Dataset):
    """
    Кастомный Dataset для полиномиальных признаков
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        

    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
    def get_test_data(self):
        return self.X, self.y


def experiment_hyperparameters(dataset):
    """
    Проводит эксперименты с различными гиперпараметрами
    """
    # Параметры для экспериментов
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']
    
    results = []
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for optim_name in optimizers:
                logging.info(f"Эксперимент: lr={lr}, batch_size={batch_size}, optimizer={optim_name}")
                
                try:
                    # Создание DataLoader
                    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    
                    # Инициализация модели
                    input_size = dataset.data['train']['X'].shape[1]
                    model = torch.nn.Linear(input_size, 1)
                    criterion = torch.nn.MSELoss()
                    
                    # Выбор оптимизатора
                    if optim_name == 'SGD':
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                    elif optim_name == 'Adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    elif optim_name == 'RMSprop':
                        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                    
                    # Обучение модели
                    for epoch in range(100):
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            outputs = model(X_batch).squeeze()
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            optimizer.step()
                    
                    # Оценка модели
                    mse = evaluate_model(model, dataset)
                    results.append({
                        'lr': lr,
                        'batch_size': batch_size,
                        'optimizer': optim_name,
                        'mse': mse
                    })
                except Exception as e:
                    logging.error(f"Ошибка в эксперименте: {str(e)}")
                    continue
    
    # Визуализация результатов
    plot_hyperparameter_results(results)
    return results


def plot_hyperparameter_results(results):
    """
    Визуализирует результаты экспериментов с гиперпараметрами
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # График для learning rate
    for optim in set(r['optimizer'] for r in results):
        lr_values = sorted(set(r['lr'] for r in results))
        mse_values = [
            np.mean([r['mse'] for r in results if r['optimizer'] == optim and r['lr'] == lr])
            for lr in lr_values
        ]
        axes[0].plot(lr_values, mse_values, label=optim)
    
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Влияние learning rate')
    axes[0].legend()
    axes[0].grid(True)
    
    # График для batch size
    for optim in set(r['optimizer'] for r in results):
        bs_values = sorted(set(r['batch_size'] for r in results))
        mse_values = [
            np.mean([r['mse'] for r in results if r['optimizer'] == optim and r['batch_size'] == bs])
            for bs in bs_values
        ]
        axes[1].plot(bs_values, mse_values, label=optim)
    
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Влияние batch size')
    axes[1].legend()
    axes[1].grid(True)
    
    # График для оптимизаторов
    optim_names = list(set(r['optimizer'] for r in results))
    mse_values = [
        np.mean([r['mse'] for r in results if r['optimizer'] == optim])
        for optim in optim_names
    ]
    axes[2].bar(optim_names, mse_values)
    axes[2].set_xlabel('Optimizer')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('Сравнение оптимизаторов')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/hyperparameter_results.png')
    plt.close()


def feature_engineering_experiment(dataset):
    """
    Проводит эксперименты с feature engineering
    """
    # Получаем исходные данные
    X_train = dataset.data['train']['X'].numpy()
    y_train = dataset.data['train']['y'].numpy()
    X_test = dataset.data['test']['X'].numpy()
    y_test = dataset.data['test']['y'].numpy()
    
    # Создаем полиномиальные признаки (степень 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Нормализация новых признаков
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)
    
    # Преобразуем в тензоры
    X_train_poly = torch.FloatTensor(X_train_poly)
    X_test_poly = torch.FloatTensor(X_test_poly)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Создаем новые датасеты
    train_poly_dataset = PolyDataset(X_train_poly, y_train)
    test_poly_dataset = PolyDataset(X_test_poly, y_test)
    
    # Обучаем модели на оригинальных и новых признаках
    input_size_original = dataset.data['train']['X'].shape[1]
    input_size_poly = X_train_poly.shape[1]
    
    # Оригинальные признаки
    model_original = torch.nn.Linear(input_size_original, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_original.parameters(), lr=0.01)
    
    # Полиномиальные признаки
    model_poly = torch.nn.Linear(input_size_poly, 1)
    optimizer_poly = torch.optim.Adam(model_poly.parameters(), lr=0.01)
    
    # Обучение на оригинальных признаках
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in tqdm(range(100), desc="Обучение на оригинальных признаках"):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model_original(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Обучение на полиномиальных признаках
    train_loader_poly = DataLoader(train_poly_dataset, batch_size=32, shuffle=True)
    for epoch in tqdm(range(100), desc="Обучение на полиномиальных признаках"):
        for X_batch, y_batch in train_loader_poly:
            optimizer_poly.zero_grad()
            outputs = model_poly(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer_poly.step()
    
    # Оценка моделей
    with torch.no_grad():
        # Оригинальные признаки
        predictions_original = model_original(dataset.data['test']['X']).squeeze()
        mse_original = torch.mean((predictions_original - y_test) ** 2).item()
        
        # Полиномиальные признаки
        predictions_poly = model_poly(test_poly_dataset.X).squeeze()
        mse_poly = torch.mean((predictions_poly - y_test) ** 2).item()
    
    logging.info(f"MSE оригинальные признаки: {mse_original:.4f}")
    logging.info(f"MSE полиномиальные признаки: {mse_poly:.4f}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.bar(['Original', 'Polynomial'], [mse_original, mse_poly])
    plt.ylabel('MSE')
    plt.title('Сравнение качества моделей')
    plt.grid(True)
    
    plt.savefig('plots/feature_engineering_results.png')
    plt.close()
    
    return {'original': mse_original, 'poly': mse_poly}


if __name__ == "__main__":
    try:
        # Загрузка данных
        dataset = CustomCSVDataset(csv_path='C:/Users/Dismas/Study/programming/practice/2/data/train.csv')
        
        # Эксперименты с гиперпараметрами
        logging.info("Эксперименты с гиперпараметрами")
        hyper_results = experiment_hyperparameters(dataset)
        
        # Эксперименты с feature engineering
        logging.info("Эксперименты с feature engineering")
        fe_results = feature_engineering_experiment(dataset)
        
    except Exception as e:
        logging.error(f"Произошла ошибка: {str(e)}")
        raise