import os
import lifetime_value as ltv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection, preprocessing
from tensorflow import keras

from .utilities import *


def preprocess(data ,target_ltv, day1_purchaseAmt_col="", numerical_features=[] ,categorical_features=[], sample_data=1, testing_size=0.2):
    """
    data : Either pandas df or a path containing pandas to a csv or parquet file
    
    target_ltv : Target column representing total purchase amount by the customer in the year following the customer's 
                                first date of purchase ie total purchase amount from day 2 to day 365.
    
    day1_purchaseAmt_col: Column name in `data` representing the total purchase amount by the customer on their first date of purchase.
                            This variable  to create additional data during train_test_split calculate metrics later on.
                            You have to specify it separately in the `numerical_features` or `categorical_features` arguments
                            if you want to ensure its included in the model.
                            

    numerical_features: List of numerical features in your model. If not specified, all the numerical columns will be used for training
    
    categorical_features: List of cateogorical features in your model. If not specified, all the categorical columns will be used for training
    
    sample: What percentage of data you wanna use for modelling(train, test)
    
    """
    ##Check if user specified path to the csv or parquet file
    if isinstance(data, str):
        path, file_type = os.path.splitext(data)
        if file_type==".csv":
            data=pd.read_csv(data)
        elif file_type==".parquet":
            data=pd.read_parquet(data, engine='pyarrow')
    
    if 0>sample_data or sample_data>1:
        raise ValueError("Error - `sample` must be a value between 0 & 1 representing total percentage of data to be used for modelling(train & test). Default is to use 100%")

    if 0>=testing_size or testing_size>=1:
        raise ValueError("Error - `testing_size` must be a value between 0 & 1 representing total percentage of data to be used for testing). Default is to use 20%")

    ##Sampling data according to user-specified `sample` 
    data=data.sample(frac=sample_data, random_state=123)

    feature_map={}

    #Dictionary to store model features & categorical mappings for later use while predicting
    if categorical_features==[]:
        ##getting list of categorical columns to encode them into either onehot encoding or label encoding
        categorical_features=data.select_dtypes(["object"]).columns.tolist()
        

    if numerical_features==[]:
        numerical_features=data.select_dtypes(["number"]).columns.tolist()
        numerical_features.remove(target_ltv)
        
    feature_map["categorical_features"]=categorical_features
    feature_map["numerical_features"]=numerical_features
    feature_map["target"]=target_ltv
    feature_map["day1_purchaseAmt_col"]=day1_purchaseAmt_col
    
    data=data[categorical_features+numerical_features+[target_ltv , day1_purchaseAmt_col]]


    if data[target_ltv].dtype!="float32":
        data[target_ltv]=data[target_ltv].astype("float32")

    ##Label Encoding
    for col in categorical_features:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(data[col])
        ##storing levels for each feature since its required at the time of prediction on unseen data
        levels = encoder.classes_
        feature_map[col] = {levels[i]: i for i in range(len(levels))}
        data[col] = encoder.transform(data[col])



    ##We'll need to split the day1 purchase amount column as well to calculate metrics later one. It's not used in training the model
    y0=data[day1_purchaseAmt_col].values
    
    ##Removing day1_purchaseAmt_col column if its not included in numerical_features
    if day1_purchaseAmt_col not in numerical_features:
        print(day1_purchaseAmt_col+" was removed from the feature list since user has not specified it in `numerical_features` argument")
        del data[day1_purchaseAmt_col]


    train, test, y0_train, y0_test= model_selection.train_test_split(data, y0, test_size=testing_size, random_state=123)
    x_train=feature_dict(train, numerical_features, categorical_features)
    y_train=train[target_ltv].values
    
    x_test=feature_dict(test, numerical_features, categorical_features)
    y_test=test[target_ltv].values

    return feature_map, x_train, x_test, y_train, y_test, y0_test


def dnn_model( feature_map, layers):

    numeric_features=feature_map["numerical_features"]
    categorical_features=feature_map["categorical_features"]

    numeric_input = tf.keras.layers.Input(
        shape=(len(numeric_features),),
        name='numeric'
    )
    embedding_inputs = [
        tf.keras.layers.Input(shape=(1,), name=key, dtype=np.int64)
        for key in categorical_features
    ]
    embedding_outputs = [
        embedding_layer(vocab_size=len(feature_map[key]))(input)
        for key, input in zip(categorical_features, embedding_inputs)
    ]
    deep_input = tf.keras.layers.concatenate(
        [*[numeric_input], *embedding_outputs]
    )
    deep_model = tf.keras.Sequential([
        *[tf.keras.layers.Dense(i, activation='relu') for i in layers],
        *[tf.keras.layers.Dense(3)]
    ])
    return tf.keras.Model(
        inputs=[*[numeric_input], *embedding_inputs],
        outputs=deep_model(deep_input)
    )


def fit_model(feature_map, x_train, y_train, x_test, y_test,
            layers=[64,32,3], epochs=100, learning_rate=0.0001,batch_size=1024,
            callback_patience=20, callback_lr=1e-06,verbose=0, log_directory_name="logs"):


        """
        Training a  Neural network model based on user specifications
        """

        model = dnn_model(feature_map , layers)

        ##Specifying the ziln loss to minimise instead of the default.
        model.compile(loss=ltv.zero_inflated_lognormal_loss, 
        optimizer=keras.optimizers.Adam(lr=learning_rate) )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=callback_patience),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=callback_lr ),
            keras.callbacks.TensorBoard(log_dir=log_directory_name)
        ]
        
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=(x_test, y_test)
        )
        return model

def model_predict(model, data, feature_map, print_performance=True ):
    """
    Function to make predictions on out of sample data. You have to encode the categorical data the same
    way as your training set. This function lets you do that along with optionally calculating
    performance metrics for on your new data.

    `model`: trained model to be used for prediction
    `data`: either pandas DataFrame or a link to a csv or parquet file
    `feature_map`: The feature mapping variable you got when running the preprocess() function when creating 
                train-test split. This is required to create identical encodings on the new data
    `show_performance`: Default False. Whether to calculate performance metrics on the new data
    """
    
    ##Reading in data if not a pandas DataFrame
    if isinstance(data, str):
        path, file_type = os.path.splitext(data)
        if file_type==".csv":
            data=pd.read_csv(data)
        elif file_type==".parquet":
            data=pd.read_parquet(data, engine='pyarrow')

    all_variables= feature_map["categorical_features"]+feature_map["numerical_features"]+[feature_map["target"] , feature_map["day1_purchaseAmt_col"]]
    
    for col in all_variables:
        if col not in data.columns:
            raise ValueError("Error -"+ col +" column not found in `data`. Please keep all column names identical to the one used while modelling ")
    
    
    data=data[all_variables]
    if data[feature_map["target"]].dtype!="float32":
        data[feature_map["target"]]=data[feature_map["target"]].astype("float32")
    
    for cat in feature_map["categorical_features"]:
        levels=list(feature_map[cat].keys())
        ##Replacing new categorical levels with UNDEFINED
        data[cat] = data[cat].apply( lambda t: t if t in levels else 'UNDEFINED' )
        # Mappings levels to the corresponding number.
        data[cat] = data[cat].apply( lambda t: feature_map[cat][t])
    y0=data[feature_map["day1_purchaseAmt_col"]].values

    x_test=feature_dict(data, feature_map["numerical_features"], feature_map["categorical_features"])
    x_test = { feat: np.array(x_test[feat]) for feat in x_test.keys()}

    logits = model.predict(x_test, batch_size=1024)

    ltv_pred = ltv.zero_inflated_lognormal_pred(logits).numpy().flatten()
    churn_predictions = K.sigmoid(logits[..., :1]).numpy().flatten()

    df = pd.DataFrame({
        'churn_predictions':churn_predictions,
        'ltv_prediction': ltv_pred
    })

    if print_performance:
        metrics,preds= ltv_performance(model, x_test, data[feature_map["target"]].values, y0)
        print(metrics.transpose())

    return df
    




