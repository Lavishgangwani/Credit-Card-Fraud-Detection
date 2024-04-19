import numpy as np
import os 
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object

## Data transformation config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    





## Data transformation class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            
            numeric_features=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
            'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            logging.info("pipeline Initiate")
            
            numerical_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ])


            preprocessor = ColumnTransformer(
                transformers=[
                    ('trf2',StandardScaler(),numeric_features),
                ],remainder='passthrough') 

            return preprocessor

        
        except Exception as e:            
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Head :\n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head :\n {test_df.head().to_string()}")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_column_name="default.payment.next.month"
            drop_columns=[target_column_name,"ID"]            
            
            # Feature devide  into independet and depedent features
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name] 



            input_feature_test_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=train_df[target_column_name] 
            
            ## apply the transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datsets.")
            
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            
            logging.info("Preprocessor pickle in create and saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
            
            
        except Exception as e:
            logging.info("Error in Data preprocessing Object")
            raise CustomException(e,sys)
        
        


