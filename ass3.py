import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import log2 as log
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.tree import export_text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from sklearn.decomposition import PCA

datafile = './data/abalone.data'
df = pd.read_csv(datafile, header=None)
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df['Sex'] = df['Sex'].replace({'M': 0, 'F': 1, 'I': 2})

#Heat map
def draw_heatmap(dataset, map_name):
    correlation_matrix = dataset.corr(numeric_only=True)
    plt.figure(figsize = (10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation matrix heatmap')
    plt.savefig(f'{map_name}_heatmap.png')

df['Class'] = pd.cut(df['Rings'], bins=[0, 7, 10, 15, np.inf], labels=[1, 2, 3, 4], right=True)
draw_heatmap(df, 'Abalone')
# Distribution of four classes
features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
plt.figure(figsize = (20, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x='Class', y=feature, data=df, hue='Class', palette='Set3', legend=False)
    plt.title(f"{feature} class distribution")
plt.tight_layout()
plt.savefig('feature_distribution.png')

# Distribution of number
def count_plot(colunm, dataset, image_name):
    plt.figure(figsize = (10, 10))
    ax = sns.countplot(x=colunm, data=dataset, hue=colunm, palette='Set2', legend=False)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha = 'center', va = 'baseline',
        fontsize=12, color='black', xytext=(0, 5),
        textcoords='offset points')
    plt.xlabel(colunm)
    plt.ylabel('Number')
    plt.savefig(f"{image_name}.png")

count_plot('Class', df, 'sample_number_distribution')

# Split dataset
def data_split(X, y, seed):
    '''
    X_train —— train data
    X_test —— test data
    y_train —— train label
    y_test —— test label
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    return X_train, X_test, y_train, y_test


def tune_train_classification_tree(X_train, y_train, n_iter=5):
    param_dist = {
        'max_depth': [None] + list(range(2,6,2)),
        'min_samples_split': [2, 5],  
        'min_samples_leaf': [1, 2],    
        'criterion': ['gini', 'entropy']     
    }

    dtree = DecisionTreeClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=dtree, 
        param_distributions=param_dist, 
        n_iter=n_iter,   
        cv=3,            
        verbose=1,       
        random_state=42, 
        n_jobs=1        
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_


datafile = './data/abalone.data'
df = pd.read_csv(datafile, header=None)
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df['Sex'] = df['Sex'].replace({'M': 0, 'F': 1, 'I': 2})
df['Class'] = pd.cut(df['Rings'], bins=[0, 7, 10, 15, np.inf], labels=[1, 2, 3, 4], right=True)

X = df.drop(['Rings', 'Class'], axis=1)
y = df['Class']


def dt_fivetrain():
    print("Training Desicion Tree model five times after using HO:")
    dt_train = []
    dt_test = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

        best_tree_model = tune_train_classification_tree(X_train, y_train, n_iter=5)

        y_train_pred = best_tree_model.predict(X_train)
        y_test_pred = best_tree_model.predict(X_test)

        dt_train_accuracy = round(accuracy_score(y_train, y_train_pred),4)
        dt_test_accuracy = round(accuracy_score(y_test, y_test_pred),4)

        dt_train.append(dt_train_accuracy)
        dt_test.append(dt_test_accuracy)
    a = round(np.mean(dt_train),4)
    b = round(np.mean(dt_test),4)
    return a,b

dt_train_accuracy,dt_test_accuracy = dt_fivetrain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
best_tree_model = tune_train_classification_tree(X_train, y_train, n_iter=5)

print(f"dt best train mean trian accuracy: {dt_train_accuracy}")
print(f"dt best test mean test accuracy: {dt_test_accuracy}")
plt.figure(figsize=(30, 15))
plot_tree(
    best_tree_model, 
    feature_names=X.columns, 
    class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'], 
    filled=True,
    impurity=False,
    proportion=True,
    precision=2,
    fontsize=6
)
plt.title("Best Decision Tree Visualization")
plt.savefig("best_tree_visualization.png")  

tree_rules = export_text(best_tree_model, feature_names=list(X.columns))
print("Tree Rules:")
print(tree_rules)

print("IF (Shell weight <= 0.11) THEN")
print("    IF (Diameter <= 0.25) THEN")
print("        IF (Shell weight <= 0.04) THEN")
print("            IF (Shell weight <= 0.03) THEN")
print("                CLASS = 1")
print("            ELSE (Shell weight > 0.03) THEN")
print("                CLASS = 1")
print("            ENDIF")
print("        ELSE (Shell weight > 0.04) THEN")
print("            IF (Shucked weight <= 0.05) THEN")
print("                CLASS = 2")
print("            ELSE (Shucked weight > 0.05) THEN")
print("                CLASS = 1")
print("            ENDIF")
print("        ENDIF")
print("    ELSE (Diameter > 0.25) THEN")
print("        IF (Sex <= 1.50) THEN")
print("            IF (Shell weight <= 0.11) THEN")
print("                CLASS = 2")
print("            ELSE (Shell weight > 0.11) THEN")
print("                CLASS = 1")
print("            ENDIF")
print("        ELSE (Sex > 1.50) THEN")
print("            IF (Viscera weight <= 0.11) THEN")
print("                CLASS = 1")
print("            ELSE (Viscera weight > 0.11) THEN")
print("                CLASS = 2")
print("            ENDIF")
print("        ENDIF")
print("    ENDIF")
print("ELSE (Shell weight > 0.11) THEN")
print("    IF (Shell weight <= 0.32) THEN")
print("        IF (Shell weight <= 0.17) THEN")
print("            IF (Shucked weight <= 0.26) THEN")
print("                CLASS = 2")
print("            ELSE (Shucked weight > 0.26) THEN")
print("                CLASS = 1")
print("            ENDIF")
print("        ELSE (Shell weight > 0.17) THEN")
print("            IF (Shucked weight <= 0.34) THEN")
print("                CLASS = 2")
print("            ELSE (Shucked weight > 0.34) THEN")
print("                CLASS = 2")
print("            ENDIF")
print("        ENDIF")
print("    ELSE (Shell weight > 0.32) THEN")
print("        IF (Shell weight <= 0.44) THEN")
print("            IF (Shucked weight <= 0.51) THEN")
print("                CLASS = 3")
print("            ELSE (Shucked weight > 0.51) THEN")
print("                CLASS = 2")
print("            ENDIF")
print("        ELSE (Shell weight > 0.44) THEN")
print("            IF (Shucked weight <= 0.63) THEN")
print("                CLASS = 4")
print("            ELSE (Shucked weight > 0.63) THEN")
print("                CLASS = 3")
print("            ENDIF")
print("        ENDIF")
print("    ENDIF")
print("ENDIF")

def tune_train_classification_forest(X_train, y_train, n_iter = 5):  
    param_dist = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [None] + list(range(10, 41, 8)),  
        'min_samples_split': [2, 5, 10, 20],  
        'min_samples_leaf': [1, 2, 5, 10],  
        'criterion': ['gini', 'entropy'],  
        'bootstrap': [True, False]  
    }

    forest = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=forest,
        param_distributions=param_dist,
        n_iter=n_iter, 
        cv=3,  
        verbose=1,
        random_state=42,
        n_jobs=1  
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

def forestfivetrain():
    print("Training Random Forest model five times after using HO:")
    rf_train = []
    rf_test = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

        best_forest_model = tune_train_classification_forest(X_train, y_train, n_iter=5)

        y_train_pred = best_forest_model.predict(X_train)
        y_test_pred = best_forest_model.predict(X_test)

        rf_train_accuracy = round(accuracy_score(y_train, y_train_pred),4)
        rf_test_accuracy = round(accuracy_score(y_test, y_test_pred),4)

        rf_train.append(rf_train_accuracy)
        rf_test.append(rf_test_accuracy)
    a = round(np.mean(rf_train),4)
    b = round(np.mean(rf_test),4)
    return a,b

rf_train_accuracy,rf_test_accuracy = forestfivetrain()
print(f"rf best train mean train accuracy: {rf_train_accuracy}")
print(f"rf best test mean test accuracy: {rf_test_accuracy}")
print()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

y = df['Class'].astype(int) - 1
def tune_train_classification_xgboost(X_train, y_train, n_iter=5):
    param_dist = {
        'n_estimators': [50, 100, 200],            
        'max_depth': [3, 5, 7, 10],                
        'learning_rate': [0.01, 0.1, 0.2, 0.3],    
        'subsample': [0.6, 0.8, 1.0],              
        'colsample_bytree': [0.6, 0.8, 1.0],       
        'gamma': [0, 0.1, 0.2, 0.3],               
        'min_child_weight': [1, 5, 10]             
    }

    xg = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    random_search = RandomizedSearchCV(
        estimator=xg,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

def XGBfiveTrain():
    print("Training XGB model five times after using HO:")
    XGB_train = []
    XGB_test = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

        best_xgboost_model = tune_train_classification_xgboost(X_train, y_train, n_iter=5)

        y_train_pred = best_xgboost_model.predict(X_train)
        y_test_pred = best_xgboost_model.predict(X_test)

        xgb_train_accuracy = round(accuracy_score(y_train, y_train_pred),4)
        xgb_test_accuracy = round(accuracy_score(y_test, y_test_pred),4)

        XGB_train.append(xgb_train_accuracy)
        XGB_test.append(xgb_test_accuracy)
    a = round(np.mean(XGB_train),4)
    b = round(np.mean(XGB_test),4)

    return a,b
xgb_train_accuracy,xgb_test_accuracy = XGBfiveTrain()
print(f"XGBoost Best train mean train accuracy: {xgb_train_accuracy}")
print(f"XGBoost Best test mean test accuracy: {xgb_test_accuracy}")
print()

def tune_train_classification_gradient_boosting(X_train, y_train, n_iter=5):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    grad = GradientBoostingClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=grad,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

def GBfivetrain():
    print("Training GB model five times after using HO:")
    GB_train = []
    GB_test = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)

        best_gradient_model = tune_train_classification_gradient_boosting(X_train, y_train, n_iter=5)

        y_train_pred = best_gradient_model.predict(X_train)
        y_test_pred = best_gradient_model.predict(X_test)

        train_acc = round(accuracy_score(y_train, y_train_pred),4)
        test_acc = round(accuracy_score(y_test, y_test_pred),4)

        GB_train.append(train_acc)
        GB_test.append(test_acc)
    
    a = round(np.mean(GB_train),4)
    b = round(np.mean(GB_test),4)

    return a,b

train_acc,test_acc = GBfivetrain()

print(f"Gradient Boosting Best train mean train accuracy: {train_acc}")
print(f"Gradient Boosting Best test mean test accuracy: {test_acc}")
print()
y = df['Class'].astype(int) + 1

def Simple_nn(X_train, y_train, X_test, y_test, optimizer):
    if optimizer == 0:
        mlp_adam = MLPClassifier(solver='adam', random_state=42, max_iter=5000)
        mlp_adam.fit(X_train, y_train)
        y_train_pred = mlp_adam.predict(X_train)
        y_test_pred = mlp_adam.predict(X_test)
    elif optimizer == 1:
        mlp_sgd = MLPClassifier(solver='sgd', random_state=42, max_iter=5000)
        mlp_sgd.fit(X_train, y_train)
        y_train_pred = mlp_sgd.predict(X_train)
        y_test_pred = mlp_sgd.predict(X_test)
    
    train_acc = round(accuracy_score(y_train, y_train_pred),4)
    test_acc = round(accuracy_score(y_test, y_test_pred),4)
    return train_acc, test_acc


experimental_num = 5
max_deep = [1, 3, 5, 7, 9, 11]

estimators = [5, 10, 15, 20]
optimizer = [0, 1]

adam_train, adam_test, sgd_train, sgd_test = [], [], [], []

print("Training NN model with optimizer Adam and SGD : ")
for i in range(0, experimental_num):
    y = df['Rings']
    y = pd.cut(y, bins=[0, 8, 11, 15, float('inf')], labels=[0, 1, 2, 3], right=False)
    X = df.drop(['Rings', 'Class'], axis=1)
    X_train, X_test, y_train, y_test = data_split(X, y, i)

    for j in optimizer:
        mlp_train_acc, mlp_test_acc = Simple_nn(X_train, y_train, X_test, y_test, j)
        if j == 0:
            adam_train.append(round(mlp_train_acc,4))
            adam_test.append(round(mlp_test_acc,4))
            #print(f"Adam optimizer: train accuracy:{round(mlp_train_acc,4)}, test accuracy: {round(mlp_test_acc,4)}")
        elif j == 1:
            sgd_train.append(mlp_train_acc)
            sgd_test.append(mlp_test_acc)
            #print(f"SGD optimizer: train accuracy:{round(mlp_train_acc,4)}, test accuracy: {round(mlp_test_acc,4)}")
    print(f"No{i+1}: Training Completed")


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

def build_and_train_model(X_train, y_train, X_test, y_test, dropout_rate, l2_lambda):
    # Build the model
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(len(np.unique(y_train)), activation='softmax')  # Assuming multi-class classification
    ])
    
    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    # Evaluate the model
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return train_accuracy, test_accuracy

# Define three different combinations of dropout rates and L2 regularization strengths
configurations = [
    {'dropout_rate': 0.0, 'l2_lambda': 0.0},
    {'dropout_rate': 0.3, 'l2_lambda': 0.01},
    {'dropout_rate': 0.5, 'l2_lambda': 0.0001},
    {'dropout_rate': 0.0, 'l2_lambda': 0.01},
    {'dropout_rate': 0.3, 'l2_lambda': 0.0}
]

# Assuming X_train, y_train, X_test, y_test are already defined as the dataset
results = []
for config in configurations:
    #print(f"Training with dropout rate: {config['dropout_rate']} and L2 lambda: {config['l2_lambda']}")
    train_accuracy, test_accuracy = build_and_train_model(
        X_train, y_train, X_test, y_test,
        dropout_rate=config['dropout_rate'],
        l2_lambda=config['l2_lambda']
    )
    results.append({
        'dropout_rate': config['dropout_rate'],
        'l2_lambda': config['l2_lambda'],
        'train_accuracy': round(train_accuracy,4),
        'test_accuracy': round(test_accuracy,4)
    })
    #print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}\n")

# Display results in a structured format
results_df = pd.DataFrame(results)
print("Experiment Results:")
print(results_df)


print(f"Decision tree accuracy after (HO): train: {dt_train_accuracy}, test: {dt_test_accuracy}")
print(f"Random forest accuracy train after (HO): {rf_train_accuracy}, test: {rf_test_accuracy}")
print(f"XGB accuracy: train after (HO): {round(xgb_train_accuracy,4)}, test: {round(xgb_test_accuracy,4)}")
print(f"Gradient boost accuracy after (HO): train: {train_acc}, test: {test_acc}")
print(f"Mean Adam accuracy: train: {round(np.mean(adam_train),4)}, test: {round(np.mean(adam_test),4)}")
print(f"Mean SGD accuracy: train: {round(np.mean(sgd_train),4)}, test: {round(np.mean(sgd_test),4)}")
print()
print()



print('Option I, part B')
datafile2 = './data/cmc.data'
df2 = pd.read_csv(datafile2, header=None)
df2.columns = ['Wife age', 'Wife education', 'Husband education', 'Number of born children', 'Wife religion', 'Wife has any job', 'Husband occupation', 'Standard-of-living index', 'Media exposure', 'Contraceptive method used']
# print(df2)
draw_heatmap(df2, 'Contraceptive')

def correlation_features(data1, data2, name_of_data1, name_of_data2):
    plt.figure(figsize=(10, 10))
    plt.scatter(data1, data2, alpha=0.6, color='orange')
    plt.xlabel(name_of_data1)
    plt.ylabel(name_of_data2)
    plt.title(f"{name_of_data1} and {name_of_data2}")
    plt.savefig(f"{name_of_data1}_{name_of_data2}.png")

def SGD_nn(X_train, y_train, X_test, y_test):
    mlp_sgd = MLPClassifier(solver='sgd', random_state=42, max_iter=1000)
    mlp_sgd.fit(X_train, y_train)
    y_train_pred = mlp_sgd.predict(X_train)
    y_test_pred = mlp_sgd.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    return train_acc, test_acc

correlation_features(df2['Wife education'], df2['Contraceptive method used'], 'wife_education', 'contraceptive_method')
count_plot('Contraceptive method used', df2, 'distribution_of_contraceptive_method')

# RF + XG boost
def adam_nn(X_train, y_train, X_test, y_test):
    grad = MLPClassifier(solver='adam', random_state=42, max_iter=5000)
    grad.fit(X_train, y_train)

    y_train_pred = grad.predict(X_train)
    y_test_pred = grad.predict(X_test)
    
    train_acc = round(accuracy_score(y_train, y_train_pred), 4)
    test_acc = round(accuracy_score(y_test, y_test_pred), 4)
    test_f1 = round(f1_score(y_test, y_test_pred, average='weighted'), 4)
    test_auc = round(roc_auc_score(y_test, grad.predict_proba(X_test), multi_class='ovr'), 4)

    print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
    print(f"Test F1 Score: {test_f1}")
    print(f"Test AUC: {test_auc}")

    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(y_test)
    for i, class_label in enumerate(unique_classes):  
        fpr, tpr, _ = roc_curve(y_test == class_label, grad.predict_proba(X_test)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class (MLP with Adam)')
    plt.legend(loc='best')
    plt.savefig('ROC Curve for Each Class made by NN')
    # plt.legend(loc='best')
    # plt.show()

    conf_mat = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (MLP with Adam)")
    plt.savefig('Confusion matrix for Each Class made by NN')
    # plt.show()

    return train_acc, test_acc, test_f1, test_auc

def RandomForest(X_train, y_train, X_test, y_test):
    grad = tune_train_classification_forest(X_train, y_train, n_iter=5)
    grad.fit(X_train, y_train)

    y_train_pred = grad.predict(X_train)
    y_test_pred = grad.predict(X_test)

    train_acc = round(accuracy_score(y_train, y_train_pred), 4)
    test_acc = round(accuracy_score(y_test, y_test_pred), 4)
    test_f1 = round(f1_score(y_test, y_test_pred, average='weighted'), 4)
    test_auc = round(roc_auc_score(y_test, grad.predict_proba(X_test), multi_class='ovr'), 4)

    print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
    print(f"Test F1 Score: {test_f1}")
    print(f"Test AUC: {test_auc}")

    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_test))):  
        fpr, tpr, _ = roc_curve(y_test == i, grad.predict_proba(X_test)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class made by RF')
    plt.legend(loc='best')
    plt.savefig("AUC-ROC result using RF")
    # plt.show()

    conf_mat = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix made by RF")
    plt.savefig("Confusion matrix result using made by RF")
    # plt.show()

    return train_acc, test_acc, test_f1, test_auc

def XGB(X_train, y_train, X_test, y_test):

    grad = tune_train_classification_xgboost(X_train, y_train, n_iter=5)
    grad.fit(X_train, y_train)

    y_train_pred = grad.predict(X_train)
    y_test_pred = grad.predict(X_test)

    train_acc = round(accuracy_score(y_train, y_train_pred), 4)
    test_acc = round(accuracy_score(y_test, y_test_pred), 4)
    test_f1 = round(f1_score(y_test, y_test_pred, average='weighted'), 4)
    test_auc = round(roc_auc_score(y_test, grad.predict_proba(X_test), multi_class='ovr'), 4)

    print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
    print(f"Test F1 Score: {test_f1}")
    print(f"Test AUC: {test_auc}")

    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(y_test)
    for i in range(len(np.unique(y_test))):  
        fpr, tpr, _ = roc_curve(y_test == i, grad.predict_proba(X_test)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class made by XGBoost')
    plt.savefig("AUC-ROC result using XGB")
    plt.legend(loc='right')
    # plt.show()

    conf_mat = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix made by XGBoost")
    plt.savefig("Confusion matrix result using XGB")
    # plt.show()

    return train_acc, test_acc, test_f1, test_auc

X = df2.drop(columns=['Contraceptive method used'])
y = df2['Contraceptive method used']
y = df2['Contraceptive method used'].astype(int) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#print("Applying XGB:")
#xgb_train_acc, xgb_test_acc, xgb_test_f1, xgb_test_auc = XGB(X_train, y_train, X_test, y_test)
#print(f"Train accuracy: {xgb_train_acc}, Test accuracy: {xgb_test_acc}, F1 score: {xgb_test_f1}, AUC :{xgb_test_auc}")
#print()
# print('Option I, part B')
print("Applying RF:")
rf_train_acc, rf_test_acc, rf_test_f1, rf_test_auc = RandomForest(X_train, y_train, X_test, y_test)
#print(f"Train accuracy: {rf_train_acc}, Test accuracy: {rf_test_acc}, F1 score: {rf_test_f1}, AUC :{rf_test_auc}")
print()
print('Option II, part B')
print("Applying Simple Neural Network with Adam Optimizer:")
nn_train_acc, nn_test_acc, nn_test_f1, nn_test_auc = adam_nn(X_train, y_train, X_test, y_test)
#print(f"Train accuracy: {nn_train_acc}, Test accuracy: {nn_test_acc}, F1 score: {nn_test_f1}, AUC :{nn_test_auc}")

def plot_feature_distribution(feature, data, target):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target, y=feature, data=data)
    plt.title(f"Distribution of {feature} across different classes")
    plt.savefig(f"Distribution of {feature} across different classes")
    # plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.savefig("Feature importances")
    # plt.show()

grad = tune_train_classification_xgboost(X_train, y_train, n_iter=5)
plot_feature_distribution('Standard-of-living index', df2, 'Contraceptive method used')
plot_feature_importance(grad, df2.drop(columns=['Contraceptive method used']).columns.tolist())

def plot_feature_scatter(data, x_feature, y_feature, target):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=target, palette='viridis', alpha=0.7)
    plt.title(f'Scatter Plot of {x_feature} vs {y_feature} with {target} as hue')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title=target)
    plt.savefig("Scatter Plot of P2")
    # plt.show()

plot_feature_scatter(df2, 'Wife age', 'Standard-of-living index', 'Contraceptive method used')

