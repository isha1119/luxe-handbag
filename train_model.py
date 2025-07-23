import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

# Step 1: Load and clean the data
df = pd.read_csv("handbags_two_brands_cleaned.csv")
df.columns = df.columns.str.strip()  # Remove trailing spaces

# Step 2: Encode features directly in df to avoid slicing warnings
le_sub = LabelEncoder()
df['Subcategory']= le_sub.fit_transform(df['Subcategory'])

le_brand = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])  # 0 = LuxeCraft, 1 = StyleNest

# Step 3: Define X and y (no slicing before encoding!)
X = df[['Subcategory', 'Price', 'Rating']]
y = df['Brand']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Create pipeline (scaler + SVM)
pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))

# Step 6: Train


pipeline.fit(X_train, y_train)

# Step 7: Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save model and encoders
joblib.dump(pipeline, "svm_brand_model.pkl")
joblib.dump(le_sub, "subcategory_encoder.pkl")
joblib.dump(le_brand, "brand_label_encoder.pkl")
