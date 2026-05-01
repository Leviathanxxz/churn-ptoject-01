from imblearn.pipeline import Pipeline as ImbPipeline # Untuk menangani imbalanced data jika diperlukan 
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from src.b_transformers import ChurnFeatureEngineer

def get_full_pipeline(model):
    # 1. Fitur Numerik
    numeric_features = [
        'Tenure Months', 'Monthly Charges', 'Total Charges','Is Streaming User',
        'Total Services','Risky Loyalist','High Cost Month To Month',
        'Is Fiber Optic High Risk','Has Security Bundle','Payment Method Risk Level'
    ]
    
    # 2. Fitur Ordinal
    ordinal_features = ['Tenure Group', 'Contract']
    
    ordinal_order = [
        ['Newbie', 'Active', 'Loyal'],          # Urutan TenureGroup
        ['Month-to-month', 'One year', 'Two year'] # Urutan Contract
    ]
    
    # 3. Fitur Nominal
    nominal_features = [
        'Partner', 'Dependents', 
        'Multiple Lines', 'Internet Service', 
        'Online Security', 'Online Backup', 
        'Tech Support', 'Streaming TV', 'Streaming Movies', 
        'Paperless Billing', 'Payment Method'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('ord', OrdinalEncoder(categories=ordinal_order), ordinal_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), nominal_features)
        ],
        remainder='drop'
    )

    return Pipeline([
        ('feature_eng', ChurnFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', model)
    ])