import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




class CustomCSVDataset(Dataset):
    """
    Кастомный Dataset класс для работы с CSV файлами.
    Предназначен для задач регрессии с одним признаком (x) и целевой переменной (y).
    
    Параметры:
        csv_path (str): Путь к CSV файлу
        target_column (str): Название целевой переменной (по умолчанию 'y')
        test_size (float): Размер тестовой выборки (по умолчанию 0.2)
        random_state (int): Seed для разбиения данных (по умолчанию 42)
    """

    def __init__(self, csv_path, target_column='y', test_size=0.2, random_state=42):
        super().__init__()
        self.csv_path = csv_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        # Загрузка и предобработка данных
        self.data = self._load_and_preprocess_data()


    def _load_and_preprocess_data(self):
        """Загружает данные из CSV и выполняет предобработку."""
        logging.info(f"Загрузка данных из {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Проверка наличия целевой переменной
        if self.target_column not in df.columns:
            raise ValueError(f"Целевая переменная {self.target_column} не найдена в данных")
        
        # Удаление строк с NaN значениями
        df = df.dropna()
        
        # Разделение на признаки и целевую переменную
        X = df[['x']]  # Используем только признак 'x'
        y = df[self.target_column]
        
        # Нормализация числовых признаков
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Преобразование в тензоры PyTorch
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        return {
            'train': {'X': X_train_tensor, 'y': y_train_tensor},
            'test': {'X': X_test_tensor, 'y': y_test_tensor}
        }
    

    def __len__(self):
        """Возвращает количество образцов в обучающей выборке."""
        return len(self.data['train']['X'])
    

    def __getitem__(self, idx):
        """Возвращает один образец данных (признаки и целевая переменная)."""
        return self.data['train']['X'][idx], self.data['train']['y'][idx]
    

    def get_test_data(self):
        """Возвращает тестовые данные."""
        return self.data['test']['X'], self.data['test']['y']


def train_linear_regression(dataset, lr=0.01, batch_size=32, epochs=100):
    """
    Обучает модель линейной регрессии на переданном датасете.
    
    Параметры:
        dataset (CustomCSVDataset): Объект датасета
        lr (float): Скорость обучения (по умолчанию 0.01)
        batch_size (int): Размер батча (по умолчанию 32)
        epochs (int): Количество эпох (по умолчанию 100)
    
    Возвращает:
        model: Обученная модель
        losses: Список значений функции потерь
    """
    # Создание DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Инициализация модели
    input_size = dataset.data['train']['X'].shape[1]
    model = torch.nn.Linear(input_size, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Обучение модели
    losses = []
    for epoch in tqdm(range(epochs), desc="Обучение линейной регрессии"):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
    
    return model, losses


def evaluate_model(model, dataset):
    """
    Оценивает производительность модели на тестовых данных.
    
    Параметры:
        model: Обученная модель
        dataset (CustomCSVDataset): Объект датасета
    
    Возвращает:
        mse: Значение MSE на тестовых данных
    """
    X_test, y_test = dataset.get_test_data()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
    
    # Проверка на NaN в предсказаниях
    if torch.isnan(predictions).any():
        raise ValueError("Обнаружены NaN значения в предсказаниях модели")
    
    mse = mean_squared_error(y_test.numpy(), predictions.numpy())
    logging.info(f"Test MSE: {mse:.4f}")
    return mse


def save_model(model, model_name, models_dir='models'):
    """
    Сохраняет модель в указанную директорию.
    
    Параметры:
        model: Модель PyTorch для сохранения
        model_name (str): Имя файла модели
        models_dir (str): Директория для сохранения (по умолчанию 'models')
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, model_name)
    torch.save(model.state_dict(), model_path)
    logging.info(f"Модель сохранена в {model_path}")


def plot_losses(losses, title='Loss during training'):
    """Визуализирует процесс обучения (значения функции потерь)."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Создаем папку plots, если ее нет
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.savefig('plots/training_loss.png')
    plt.close()



if __name__ == "__main__":
    try:
        # Загрузка и подготовка данных
        train_dataset = CustomCSVDataset(csv_path='C:/Users/Dismas/Study/programming/practice/2/data/train.csv')
        test_dataset = CustomCSVDataset(csv_path='C:/Users/Dismas/Study/programming/practice/2/data/test.csv')
        
        # Обучение модели
        model, losses = train_linear_regression(train_dataset, lr=0.01, epochs=200)
        
        # Оценка модели
        mse = evaluate_model(model, test_dataset)
        
        # Сохранение модели и графиков
        save_model(model, 'linear_regression_model.pth')
        plot_losses(losses)
        
    except Exception as e:
        logging.error(f"Произошла ошибка: {str(e)}")
        raise