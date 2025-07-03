import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import unittest
from typing import Dict, Optional
from utils import *
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from utils import make_classification_data, ClassificationDataset

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LinearRegression(nn.Module):
    """
    Линейная регрессия с L1/L2 регуляризацией и early stopping.
    
    Параметры:
        input_dim (int): Размерность входных признаков
        l1_lambda (float): Коэффициент для L1 регуляризации (по умолчанию 0.0)
        l2_lambda (float): Коэффициент для L2 регуляризации (по умолчанию 0.0)
        learning_rate (float): Скорость обучения (по умолчанию 0.01)
    """
    def __init__(self, input_dim: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0, learning_rate: float = 0.01):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.loss_history = {'train': [], 'val': []}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели."""
        return self.linear(x)
    
    def loss_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Функция потерь с L1 и L2 регуляризацией.
        
        Параметры:
            y_pred (torch.Tensor): Предсказанные значения
            y_true (torch.Tensor): Истинные значения
            
        Возвращает:
            torch.Tensor: Значение функции потерь
        """
        mse_loss = nn.MSELoss()(y_pred, y_true)
        
        # L1 регуляризация (Lasso)
        l1_loss = torch.tensor(0., requires_grad=True)
        if self.l1_lambda > 0:
            for param in self.parameters():
                l1_loss = l1_loss + torch.norm(param, 1)
        
        # L2 регуляризация (Ridge)
        l2_loss = torch.tensor(0., requires_grad=True)
        if self.l2_lambda > 0:
            for param in self.parameters():
                l2_loss = l2_loss + torch.norm(param, 2)
        
        total_loss = mse_loss + self.l1_lambda * l1_loss + self.l2_lambda * l2_loss
        return total_loss
    
    def fit(
        self, 
        dataset: RegressionDataset, 
        val_dataset: Optional[RegressionDataset] = None,
        epochs: int = 1000, 
        batch_size: int = 32,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Обучение модели с early stopping.
        
        Параметры:
            dataset (RegressionDataset): Обучающий датасет
            val_dataset (RegressionDataset, optional): Валидационный датасет
            epochs (int): Максимальное количество эпох
            batch_size (int): Размер батча
            patience (int): Количество эпох для early stopping
            verbose (bool): Выводить ли процесс обучения
            
        Возвращает:
            Dict[str, list]: История значений функции потерь
        """
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
            use_validation = True
        else:
            use_validation = False
            
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        no_improvement = 0
        
        progress_bar = tqdm(range(epochs), disable=not verbose)
        for epoch in progress_bar:
            # Обучение
            self.train()
            total_train_loss = 0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.loss_history['train'].append(avg_train_loss)
            
            # Валидация
            if use_validation:
                self.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        val_outputs = self(X_val)
                        val_loss = self.loss_fn(val_outputs, y_val)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                self.loss_history['val'].append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    no_improvement = 0
                    best_weights = self.state_dict()
                else:
                    no_improvement += 1
                    
                if no_improvement >= patience:
                    self.load_state_dict(best_weights)
                    logger.info(f'Early stopping at epoch {epoch}')
                    break
                    
                progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                log_epoch(epoch+1, avg_train_loss, val_loss=avg_val_loss)
            else:
                progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")
                log_epoch(epoch+1, avg_train_loss)
                
        return self.loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание модели.
        
        Параметры:
            X (np.ndarray): Входные данные
            
        Возвращает:
            np.ndarray: Предсказанные значения
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self(X_tensor).numpy()
        return predictions
    
    def plot_loss_history(self):
        """Визуализация истории функции потерь."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history['train'], label='Train Loss')
        if 'val' in self.loss_history and len(self.loss_history['val']) > 0:
            plt.plot(self.loss_history['val'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss History')
        plt.legend()
        plt.grid(True)
        plt.show()

class TestLinearRegression(unittest.TestCase):
    """Unit-тесты для класса LinearRegression."""
    
    def setUp(self):
        """Инициализация тестовых данных."""
        X, y = make_regression_data(n=100, noise=0.5)
        self.dataset = RegressionDataset(X, y)
        
    def test_model_initialization(self):
        """Тест инициализации модели."""
        model = LinearRegression(input_dim=1)
        self.assertEqual(model.linear.in_features, 1)
        self.assertEqual(model.linear.out_features, 1)
        
    def test_training(self):
        """Тест обучения модели."""
        model = LinearRegression(input_dim=1, l1_lambda=0.1, l2_lambda=0.1)
        loss_history = model.fit(self.dataset, epochs=100, verbose=False)
        self.assertLess(loss_history['train'][-1], loss_history['train'][0])
        
    def test_prediction(self):
        """Тест предсказания модели."""
        model = LinearRegression(input_dim=1)
        model.fit(self.dataset, epochs=100, verbose=False)
        test_X = np.random.rand(10, 1)
        predictions = model.predict(test_X)
        self.assertEqual(predictions.shape, (10, 1))

def main_lin():
    """Основная функция для демонстрации работы модели."""
    # Генерация данных
    X, y = make_regression_data(n=1000, noise=0.5)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    train_dataset = RegressionDataset(X_train, y_train)
    test_dataset = RegressionDataset(X_test, y_test)
    
    # Обучение моделей с разными типами регуляризации
    models = {
        'No Reg': LinearRegression(input_dim=1, learning_rate=0.01),
        'L1 (Lasso)': LinearRegression(input_dim=1, l1_lambda=0.5, learning_rate=0.01),
        'L2 (Ridge)': LinearRegression(input_dim=1, l2_lambda=0.5, learning_rate=0.01),
        'ElasticNet': LinearRegression(input_dim=1, l1_lambda=0.3, l2_lambda=0.3, learning_rate=0.01)
    }
    
    for name, model in models.items():
        logger.info(f"\nTraining {name} model")
        model.fit(train_dataset, val_dataset=test_dataset, epochs=500, patience=20)
        model.plot_loss_history()
        
        # Оценка модели
        y_pred = model.predict(X_test.numpy())
        test_mse = mse(torch.tensor(y_pred), y_test)
        logger.info(f"{name} Model - Test MSE: {test_mse:.4f}")




class LogisticRegression(nn.Module):
    """
    Расширенная логистическая регрессия с поддержкой многоклассовой классификации.
    
    Args:
        in_features (int): Количество входных признаков
        num_classes (int): Количество классов (по умолчанию 2 для бинарной классификации)
    """
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        if num_classes == 2:
            self.linear = nn.Linear(in_features, 1)  # Бинарная классификация
        else:
            self.linear = nn.Linear(in_features, num_classes)  # Многоклассовая классификация

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            x (torch.Tensor): Входные данные формы (batch_size, in_features)
            
        Возвращает выходные логиты формы (batch_size, num_classes) 
        """
        return self.linear(x)

class MetricsCalculator:
    """
    Класс для вычисления метрик классификации.
    """
    @staticmethod
    def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Вычисляет точность (accuracy).
        
        Args:
            y_pred (torch.Tensor): Предсказанные вероятности классов
            y_true (torch.Tensor): Истинные метки классов
            
        Возвращает значение точности
        """
        y_pred_class = torch.argmax(y_pred, dim=1)
        y_true_class = torch.argmax(y_true, dim=1) if y_true.dim() > 1 else y_true
            
        return (y_pred_class == y_true_class).float().mean().item()

    @staticmethod
    def precision_recall_f1(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
        """
        Вычисляет precision, recall и F1-score.
        
        Args:
            y_pred (torch.Tensor): Предсказанные вероятности классов
            y_true (torch.Tensor): Истинные метки классов
            num_classes (int): Количество классов
            
        Возвращает словарь с метриками
        """
        if num_classes == 2:
            y_pred_class = (y_pred > 0.5).float()
            tp = ((y_pred_class == 1) & (y_true == 1)).sum().item()
            fp = ((y_pred_class == 1) & (y_true == 0)).sum().item()
            fn = ((y_pred_class == 0) & (y_true == 1)).sum().item()
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            return {'precision': precision, 'recall': recall, 'f1': f1}
        else:
            y_pred_class = torch.argmax(y_pred, dim=1)
            y_true_class = y_true if y_true.dim() == 1 else torch.argmax(y_true, dim=1)
            
            metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            for c in range(num_classes):
                tp = ((y_pred_class == c) & (y_true_class == c)).sum().item()
                fp = ((y_pred_class == c) & (y_true_class != c)).sum().item()
                fn = ((y_pred_class != c) & (y_true_class == c)).sum().item()
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                metrics['precision'] += precision
                metrics['recall'] += recall
                metrics['f1'] += f1
            
            # Усредняем по классам
            metrics['precision'] /= num_classes
            metrics['recall'] /= num_classes
            metrics['f1'] /= num_classes
            
            return metrics

    @staticmethod
    def roc_auc_score(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 2) -> float:
        """
        Вычисляет ROC-AUC score.
        
        Args:
            y_pred (torch.Tensor): Предсказанные вероятности классов
            y_true (torch.Tensor): Истинные метки классов
            num_classes (int): Количество классов
            
        Returns:
            float: ROC-AUC score
        """
        y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
        
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true_np, y_pred_np)
            return auc(fpr, tpr)
        else:
            # Бинаризуем метки для многоклассового случая
            y_true_bin = label_binarize(y_true_np, classes=range(num_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_np[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Усредненный ROC-AUC
            return sum(roc_auc.values()) / num_classes


def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 2) -> None:
    """
    Визуализирует матрицу ошибок (confusion matrix).
    
    Args:
        y_true (torch.Tensor): Истинные метки классов
        y_pred (torch.Tensor): Предсказанные метки классов
        num_classes (int): Количество классов
    """
    if num_classes == 2:
        y_pred_class = (y_pred > 0.5).int()
    else:
        y_pred_class = torch.argmax(y_pred, dim=1)
    
    y_true_class = y_true.int() if y_true.dim() == 1 else torch.argmax(y_true, dim=1)
    
    cm = confusion_matrix(y_true_class.numpy(), y_pred_class.numpy())
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f'Class {i}' for i in range(num_classes)])
    plt.yticks(tick_marks, [f'Class {i}' for i in range(num_classes)])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Добавляем текст с количеством в каждой ячейке
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 2) -> None:
    """
    Визуализирует ROC-кривую.
    
    Args:
        y_true (torch.Tensor): Истинные метки классов
        y_pred (torch.Tensor): Предсказанные вероятности классов
        num_classes (int): Количество классов
    """
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true_np, y_pred_np)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    else:
        # Бинаризуем метки для многоклассового случая
        y_true_bin = label_binarize(y_true_np, classes=range(num_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Усредненная ROC-кривая
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        
        plt.figure()
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], lw=1, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=2, label='Mean ROC')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Multiclass)')
        plt.legend(loc="lower right")
        plt.show()


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    log_interval: int = 10,
    num_classes: int = 2) -> Dict[str, list[float]]:
    """
    Обучает модель логистической регрессии.
    
    Args:
        model (nn.Module): Модель для обучения
        dataloader (DataLoader): Загрузчик данных
        criterion (nn.Module): Функция потерь
        optimizer (optim.Optimizer): Оптимизатор
        epochs (int): Количество эпох обучения
        log_interval (int): Интервал логирования
        num_classes (int): Количество классов
        
    Returns:
        Dict[str, list[float]]: История метрик обучения
    """
    history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    # Инициализация прогресс-бара
    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Сохраняем предсказания и метки для вычисления метрик
            if num_classes == 2:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            
            all_preds.append(probs.detach())
            all_targets.append(batch_y.detach())
        
        # Вычисляем метрики
        avg_loss = total_loss / len(dataloader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        acc = MetricsCalculator.accuracy(all_preds, all_targets)
        metrics = MetricsCalculator.precision_recall_f1(all_preds, all_targets, num_classes)
        roc_auc = MetricsCalculator.roc_auc_score(all_preds, all_targets, num_classes)
        
        # Сохраняем историю
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        history['precision'].append(metrics['precision'])
        history['recall'].append(metrics['recall'])
        history['f1'].append(metrics['f1'])
        history['roc_auc'].append(roc_auc)
        
        # Обновляем прогресс-бар
        pbar.set_postfix({
            'loss': avg_loss,
            'acc': acc,
            'f1': metrics['f1'],
            'roc_auc': roc_auc
        })
        
        # Логируем прогресс
        if epoch % log_interval == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"Loss: {avg_loss:.4f}, "
                f"Accuracy: {acc:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"ROC-AUC: {roc_auc:.4f}"
            )
    
    pbar.close()
    return history


def plot_training_history(history: Dict[str, list[float]]) -> None:
    """
    Визуализирует историю обучения.
    
    Args:
        history (Dict[str, list[float]]): История метрик обучения
    """
    plt.figure(figsize=(12, 8))
    
    # График потерь
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # График точности
    plt.subplot(2, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    # График precision, recall, F1
    plt.subplot(2, 2, 3)
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.plot(history['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall and F1 Score')
    plt.legend()
    
    # График ROC-AUC
    plt.subplot(2, 2, 4)
    plt.plot(history['roc_auc'], label='ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def test_model(model: nn.Module, dataloader: DataLoader, num_classes: int = 2) -> Dict[str, float]:
    """
    Тестирует модель на тестовых данных.
    
    Args:
        model (nn.Module): Обученная модель
        dataloader (DataLoader): Загрузчик тестовых данных
        num_classes (int): Количество классов
        
    Returns:
        Dict[str, float]: Метрики на тестовых данных
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # Используем tqdm для отображения прогресса тестирования
        pbar = tqdm(dataloader, desc="Testing", unit="batch")
        for batch_X, batch_y in pbar:
            logits = model(batch_X)
            
            if num_classes == 2:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            
            all_preds.append(probs)
            all_targets.append(batch_y)
        pbar.close()
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = {
        'accuracy': MetricsCalculator.accuracy(all_preds, all_targets),
        **MetricsCalculator.precision_recall_f1(all_preds, all_targets, num_classes),
        'roc_auc': MetricsCalculator.roc_auc_score(all_preds, all_targets, num_classes)
    }
    
    # Визуализация
    plot_confusion_matrix(all_targets, all_preds, num_classes)
    plot_roc_curve(all_targets, all_preds, num_classes)
    
    return metrics


def main_log():
    """Основная функция для демонстрации работы модели."""
    # Генерируем данные для бинарной классификации
    X_bin, y_bin = make_classification_data(n=500, source='random')
    dataset_bin = ClassificationDataset(X_bin, y_bin)
    train_loader_bin = DataLoader(dataset_bin, batch_size=32, shuffle=True)
    
    # Создаем модель для бинарной классификации
    model_bin = LogisticRegression(in_features=2, num_classes=2)
    criterion_bin = nn.BCEWithLogitsLoss()
    optimizer_bin = optim.SGD(model_bin.parameters(), lr=0.1)
    
    # Обучаем модель
    logger.info("Training binary classification model...")
    history_bin = train_model(
        model_bin, train_loader_bin, criterion_bin, optimizer_bin, 
        epochs=100, num_classes=2
    )
    
    # Визуализируем историю обучения
    plot_training_history(history_bin)
    
    # Тестируем модель
    logger.info("Testing binary classification model...")
    test_metrics_bin = test_model(model_bin, train_loader_bin, num_classes=2)
    logger.info(f"Test metrics: {test_metrics_bin}")
    
    X_multi = torch.randn(500, 4)
    w_multi = torch.tensor([[2.0, -1.0, 0.5, -0.5], 
                           [-1.0, 1.5, -0.5, 1.0], 
                           [0.5, -0.5, 1.0, -1.5]])
    b_multi = torch.tensor([0.1, -0.1, 0.2])
    logits_multi = X_multi @ w_multi.T + b_multi
    y_multi = torch.argmax(logits_multi, dim=1)
    
    dataset_multi = ClassificationDataset(X_multi, y_multi)
    train_loader_multi = DataLoader(dataset_multi, batch_size=32, shuffle=True)
    
    # Создаем модель
    model_multi = LogisticRegression(in_features=4, num_classes=3)
    criterion_multi = nn.CrossEntropyLoss()
    optimizer_multi = optim.SGD(model_multi.parameters(), lr=0.1)
    
    # Обучаем модель
    logger.info("Training multiclass classification model...")
    history_multi = train_model(
        model_multi, train_loader_multi, criterion_multi, optimizer_multi, 
        epochs=100, num_classes=3
    )
    
    # Визуализируем историю обучения
    plot_training_history(history_multi)
    
    # Тестируем модель
    logger.info("Testing multiclass classification model...")
    test_metrics_multi = test_model(model_multi, train_loader_multi, num_classes=3)
    logger.info(f"Test metrics: {test_metrics_multi}")




if __name__ == "__main__":
    # Запуск unit-тестов
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Запуск демонстрации
    main_lin()
    main_log()