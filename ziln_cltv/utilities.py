from typing import Sequence
import lifetime_value as ltv
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn import metrics
from tensorflow.keras import backend as K


def feature_dict(df, numerical_features, categorical_features):
    """
    Converting dataFrame to dictionary for model inputs
    """
    features = {k: v.values for k, v in dict(df[categorical_features]).items()}
    features["numeric"] = df[numerical_features].values
    return features

def spearmanrank(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculates spearmanr rank correlation coefficient.
    Args:
      x1: 1D array_like.
      x2: 1D array_like.

    Returns:
      correlation: float.

    More info on the metric here: https://docs.scipy.org/doc/scipy/reference/stats.html.
    """
    return stats.spearmanr(x, y, nan_policy='raise')[0]

def embedding_dim(x):
    """
    Finds total dimensions for embeddings based on google recommended formula:
    """
    return int(x**.25) + 1

def embedding_layer(vocab_size):
    """
    Build embedding object from vocab
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim(vocab_size),
            input_length=1),
        tf.keras.layers.Flatten(),
    ])

def ltv_performance(model, x_test, y_test, y0_test):
    """
    Calculates the following metrics:
        1. Regression(predicting total ltv  (if your target variable was ltv for for day 2 till day 365, 
            then your prediction will also be for the same) - Spearman R, Decile MAPE, Gini 
        2. Classficiation(predicting churn probability.)  - Precision, Recall, AUC,  F1 Score
    """
    # making predictions 
    logits = model.predict(x=x_test, batch_size=1024)

    #This is customer lifetime value prediction.  
    ltv_pred = ltv.zero_inflated_lognormal_pred(logits).numpy().flatten()

    #apllying sigmoid function to convert ltv to probability which becomes your churn prediction
    churn_predictions = K.sigmoid(logits[..., :1]).numpy().flatten()

    #actual churn value
    churn_actual = (y_test > 0).astype('float32')

    # Calculating metrics for lifetime value predictions:
    df = pd.DataFrame({
        'churn_actual':churn_actual,
        'churn_pred':churn_predictions,

        'ltv_actual': y_test,
        'ltv_pred': ltv_pred
    })

    gain = pd.DataFrame({
        'lorenz': ltv.cumulative_true(y_test, y_test),
        'baseline': ltv.cumulative_true(y_test, y0_test),
        'model': ltv.cumulative_true(y_test, ltv_pred),
    })

    total_customers = np.float32(gain.shape[0])
    gain['cumulative_customer'] = (np.arange(total_customers) + 1.) / total_customers

    #Plotting Gain chart
    ax = gain[[
        'cumulative_customer',
        'lorenz',
        'baseline',
        'model']].plot( x='cumulative_customer', figsize=(8, 5), legend=True)
    ax.legend(['Groundtruth', 'Baseline', 'Model'], loc='upper left')
    ax.set_xlabel('Cumulative Fraction of Customers')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim((0, 1.))
    ax.set_ylabel('Cumulative Fraction of Total Lifetime Value')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim((0, 1.05))
    ax.set_title('Gain Chart')

    # Deciles for MAPE
    ltv_pred=np.clip(ltv_pred, y_test.min(), y_test.max())
    df_decile = ltv.decile_stats(y_test, ltv_pred)

    #Plotting Decile CHart
    ax = df_decile[['label_mean', 'pred_mean']].plot.bar(rot=0)
    ax.set_title('Decile Chart')
    ax.set_xlabel('Prediction bucket')
    ax.set_ylabel('Average bucket value')
    ax.legend(['Label', 'Prediction'], loc='upper left')

    spearman_correlation = spearmanrank(y_test, ltv_pred)

    # Gini coefficient = 2 * (area between lorenz curve and 45 degree line)
    gini_coefficient = ltv.gini_from_gain(gain[['lorenz', 'baseline', 'model']])


    # Calculating metrics for churn probability
    precision = metrics.precision_score(churn_actual,1 * (churn_predictions > 0.5))
    recall = metrics.recall_score(churn_actual, 1 * (churn_predictions > 0.5))
    pr_auc = metrics.average_precision_score(churn_actual, churn_predictions)
    f1_score = metrics.f1_score(churn_actual, 1 * (churn_predictions > 0.5))
    auc = metrics.roc_auc_score(churn_actual, churn_predictions)

    results = pd.DataFrame(
        {   'Total Customers in data': total_customers,
            'Mean actual LTV': y_test.mean(),
            'Mean predicted LTV': ltv_pred.mean(),
            'Percent of customers with non-zero LTV': np.mean(y_test > 0),
            'LTV Decile MAPE': df_decile['decile_mape'].mean(),
            'LTV Baseline GINI': gini_coefficient['normalized'][1],
            'LTV Gini Coefficient': gini_coefficient['normalized'][2],
            'LTV Spearman Correlation': spearman_correlation,
            'Mean actual Churn probability': churn_actual.mean(),
            'Mean predicted Churn probability': churn_predictions.mean(),
            'ROC AUC': auc,
            'PR AUC': pr_auc,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1_score
        }, index=[0]
    )
    return results, df
