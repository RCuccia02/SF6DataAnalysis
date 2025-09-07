import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.utils import prepare_dataset, prepare_tensors, prepare_split, create_callbacks, prepare_dataLoaders, create_trainer

model_type = "onehot"
df = pd.read_csv('./data.csv')
df_long = pd.read_csv('./data_long.csv')
all_chars = [
    'Jamie', 'Terry', 'Zangief', 'Kimberly', 'A.K.I.', 'Edmond Honda',
    'Ken', 'Dee Jay', 'Ryu', 'Manon', 'Marisa', 'Mai', 'Ed', 'Cammy',
    'Akuma', 'Lily', 'Luke', 'JP', 'Blanka', 'Juri', 'M. Bison',
    'Dhalsim', 'Guile', 'Chun-Li', 'Random', 'Rashid'
]
num_chars = len(all_chars)

df = prepare_dataset(df, df_long)

X, y = prepare_tensors(df, all_chars, model_type=model_type)

print(X.shape, y.shape)
print("-------------------------")


X=X.numpy()
y=y.numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=1000,          # più alberi per dare al modello più capacità di apprendimento
    max_depth=8,               # leggermente più basso per evitare overfitting
    learning_rate=0.005,        # mantiene aggiornamenti stabili
    subsample=0.8,             # usa solo l'80% dei dati per ogni albero → più robusto
    colsample_bytree=0.8,      # usa solo l'80% delle feature per ogni albero
    reg_lambda=0.1,            # regolarizzazione L2 più forte per ridurre overfitting
    objective='binary:logistic',
    random_state=42,
    verbosity=1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of the Logistic Regression Model: ",accuracy)


