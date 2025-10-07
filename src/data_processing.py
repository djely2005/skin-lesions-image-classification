import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_metadata = pd.read_csv('data/MILK10k_Training_Metadata.csv')
print(df_metadata.head())
df_supplement = pd.read_csv('data/MILK10k_Training_Supplement.csv')

IMAGE_TYPE = {'clinical: close-up': 0, 'dermoscopic': 1}
IMAGE_MANIPULATION = {'altered': 0, 'instrument only': 1}
SEX = {'male': 0, 'female': 1}
SITE = {'head_neck_face': 0, 'lower_extremity': 1, 'upper_extremity': 2, 'trunk': 3, 'foot': 4, 'genital': 5, 'hand': 6}
DIAGNOSIS_FULL = { 'Squamous cell carcinoma, Invasive': 0, 'Nevus, Reed': 1, 'Nevus, Acral': 2, 'Basal cell carcinoma': 3, 'Squamous cell carcinoma in situ, Bowens disease': 4, 'Nevus, NOS, Dermal': 5, 'Nevus, NOS, Compound': 6, 'Melanoma in situ': 7, 'Seborrheic keratosis': 8, 'Keratoacanthoma': 9, 'Melanoma metastasis': 10, 'Lichen planus like keratosis': 11, 'Hemangioma': 12, 'Nevus': 13, 'Nevus, NOS, Junctional': 14, 'Nevus, Congenital': 15, 'Melanoma Invasive': 16, 'Inflammatory or infectious diseases': 17, 'Solar or actinic keratosis': 18, 'Dermatofibroma': 19, 'Sebaceous hyperplasia': 20, 'Angiokeratoma': 21, 'Trichoblastoma': 22, 'Solar lentigo': 23, 'Nevus, Combined': 24, 'Clear cell acanthoma': 25, 'Benign - Other': 26, 'Benign soft tissue proliferations - Fibro-histiocytic': 27, 'Blue nevus': 28, 'Collision - Only benign proliferations': 29, 'Exogenous': 30, 'Nevus, Spitz': 31, 'Mucosal melanotic macule': 32, 'Infundibular or epidermal cyst': 33, 'Benign soft tissue proliferations - Vascular': 34, 'Nevus, Recurrent or persistent': 35, 'Collision - At least one malignant proliferation': 36, 'Ink-spot lentigo': 37, 'Nevus, BAP-1 deficient': 38, 'Juvenile xanthogranuloma': 39, 'Nevus, Spilus': 40, 'Pyogenic granuloma': 41, 'Supernumerary nipple': 42, 'Porokeratosis': 43, 'Nevus, Balloon cell': 44, 'Hemangioma, Hobnail': 45, 'Molluscum': 46, 'Mastocytosis': 47,}
DIAGNOSIS_CONFIRM_TYPE = {'histopathology': 0, 'single contributor clinical assessment': 1}


df = pd.merge(df_metadata, df_supplement, on='isic_id')
df = df.dropna(subset=['site', 'age_approx'])
df = df.drop(['invasion_thickness_interval', 'attribution', 'copyright_license'], axis=1)

# Map categorical features
df['sex'] = df['sex'].map(SEX)
df['image_manipulation'] = df['image_manipulation'].map(IMAGE_MANIPULATION)
df['site'] = df['site'].map(SITE)
df['image_type'] = df['image_type'].map(IMAGE_TYPE)
df['diagnosis_full'] = df['diagnosis_full'].map(DIAGNOSIS_FULL)
df['diagnosis_confirm_type'] = df['diagnosis_confirm_type'].map(DIAGNOSIS_CONFIRM_TYPE)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(
    ['sex', 'site', 'image_type', 'image_manipulation', 'diagnosis_full', 'diagnosis_confirm_type']
)
categorical_cols = ['sex', 'site', 'image_type', 'image_manipulation', 'diagnosis_full', 'diagnosis_confirm_type']

# Scale numeric columns
scaler = StandardScaler()
df_scaled_numeric = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

# One-hot encode categorical columns
df_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)

lesion_images = pd.concat([df[['lesion_id', 'isic_id']], df_categorical[['image_manipulation', 'image_type', 'site']]], axis=1)
print(lesion_images.head())

# Combine numeric and categorical
df_final = pd.concat([df_scaled_numeric, df_categorical], axis=1)
print(df_final.info())

# Delete some ugly data
df_truth = pd.read_csv('data/MILK10k_Training_GroundTruth.csv')
