"""
Explainability utilities for ecg using SHAP and other interpretability tools.

This module provides model interpretability features crucial for medical AI applications,
including SHAP values, feature importance, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

# LIME imports (for image explanations)
try:
    import lime
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Install with: pip install lime")


class ModelExplainer:
    """
    Explainability wrapper for PyTorch models.
    Uses DeepExplainer for deep learning models.
    """
    
    def __init__(self, model, X_train_sample=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained PyTorch model
            X_train_sample: Sample of training data (for background)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        import torch
        
        self.model = model
        self.model.eval()  # Set to evaluation mode
        
        if X_train_sample is not None:
            if isinstance(X_train_sample, np.ndarray):
                X_train_sample = torch.FloatTensor(X_train_sample)
            # Use subset for faster computation
            if len(X_train_sample) > 100:
                X_train_sample = X_train_sample[:100]
            self.explainer = shap.DeepExplainer(model, X_train_sample)
        else:
            self.explainer = None
    
    def explain_instance(self, instance, plot=True, class_idx=None):
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain
            plot: Whether to plot SHAP values
            class_idx: Class index to explain
            
        Returns:
            shap_values: SHAP values
        """
        import torch
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        
        if len(instance.shape) == 1:
            instance = instance.unsqueeze(0)
        
        if self.explainer is None:
            # Create explainer on-the-fly
            self.explainer = shap.DeepExplainer(self.model, instance)
        
        shap_values = self.explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            if class_idx is not None:
                shap_values = shap_values[class_idx]
            else:
                shap_values = shap_values[0]
        
        if plot:
            # Convert to numpy for plotting
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.detach().cpu().numpy()
            if isinstance(instance, torch.Tensor):
                instance = instance.detach().cpu().numpy()
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=0,
                    data=instance[0]
                )
            )
        
        return shap_values
    
    def explain_dataset(self, X, max_instances=50, plot=True, class_idx=None):
        """
        Explain multiple instances.
        
        Args:
            X: Instances to explain
            max_instances: Maximum number of instances
            plot: Whether to plot summary
            class_idx: Class index to explain
            
        Returns:
            shap_values: SHAP values
        """
        import torch
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        if len(X) > max_instances:
            X = X[:max_instances]
        
        if self.explainer is None:
            sample = X[:min(10, len(X))]
            self.explainer = shap.DeepExplainer(self.model, sample)
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            if class_idx is not None:
                shap_values = shap_values[class_idx]
            else:
                shap_values = shap_values[0]
        
        if plot:
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.detach().cpu().numpy()
            if isinstance(X, torch.Tensor):
                X = X.detach().cpu().numpy()
            
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            plt.show()
        
        return shap_values


def create_lime_explainer(model, preprocess_fn=None):
    """
    Create LIME explainer for image models.
    
    Args:
        model: Trained model
        preprocess_fn: Optional preprocessing function
        
    Returns:
        explainer: LIME explainer
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    explainer = lime_image.LimeImageExplainer()
    return explainer


def explain_with_lime(explainer, image, model, top_labels=5, num_features=10):
    """
    Explain image prediction using LIME.
    
    Args:
        explainer: LIME explainer
        image: Input image
        model: Trained model
        top_labels: Number of top labels to explain
        num_features: Number of features to show
        
    Returns:
        explanation: LIME explanation
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    explanation = explainer.explain_instance(
        image.astype('double'),
        model.predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=1000
    )
    
    return explanation


def plot_lime_explanation(explanation, label=1, figsize=(10, 5)):
    """
    Plot LIME explanation.
    
    Args:
        explanation: LIME explanation object
        label: Label to explain
        figsize: Figure size
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=10,
        hide_rest=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(temp)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask)
    axes[1].set_title('LIME Explanation (Important Regions)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
