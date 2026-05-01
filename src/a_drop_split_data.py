import pandas as pd
from sklearn.model_selection import train_test_split

def load_split_data(path=
                    'C:/ml-churn-project/data/Telco_customer_churn.csv',
                    test_size=
                    0.2,):
    # Load data
    data = pd.read_csv(path)
    
#  Hapus kolom yang tidak relevan SEBELUM data di-split
    cols_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 
                    'Zip Code', 'Lat Long', 'Latitude', 'Longitude','Gender',
                    'Churn Score', 'Churn Reason', 'Churn Label']
    
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
        
    # Pisahkan fitur dan target
    X = data.drop(columns=['Churn Value'])
    y = data['Churn Value']
    
    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=42,
                                                        stratify=y)  # Stratify untuk menjaga proporsi kelas
    
    return X_train, X_test, y_train, y_test

