import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Loan Risk Prediction Analysis", layout="wide")

# Title
st.title("Prediccion y analisis UNAB")
st.title("Análisis Distribución y Características de Préstamos")

# Descripción general

# Detalles de cada característica
analisis_texto = """
- **credit.policy**: La mayoría de los datos está en el valor 1, lo que indica que la mayoría de los préstamos cumplen con la política de crédito, mientras que una fracción menor no cumple (valor 0).

- **int.rate**: Las tasas de interés parecen tener una distribución asimétrica con una concentración entre 10% y 15%. Esto sugiere que la mayoría de los préstamos tienen una tasa de interés en este rango.

- **installment**: La mayoría de los pagos mensuales de los préstamos están concentrados por debajo de 500, con una caída gradual en la cantidad de préstamos conforme el valor aumenta.

- **log.annual.inc**: La mayoría de los ingresos anuales (en escala logarítmica) se encuentran entre 10 y 12, lo que indica que la mayoría de los prestatarios tienen ingresos anuales moderados.

- **dti**: La razón de deuda a ingreso (dti) muestra una distribución simétrica alrededor de valores menores a 20, con algunos préstamos con valores más altos.

- **fico**: La puntuación FICO está bien distribuida entre 600 y 850, con una concentración en el rango de 700 a 800, lo que indica prestatarios con un buen historial crediticio en general.

- **days.with.cr.line**: La mayoría de los prestatarios tienen un historial crediticio de varios miles de días (alrededor de 5,000 a 10,000 días), lo que sugiere un largo tiempo con acceso a crédito.

- **revol.bal**: Los saldos renovables tienden a ser bajos en su mayoría, aunque algunos tienen saldos elevados.

- **revol.util**: La utilización del crédito renovable (en porcentaje) muestra una tendencia a acumularse en valores menores, lo cual sugiere que los prestatarios no suelen utilizar una gran parte de su crédito disponible.

- **inq.last.6mths**: La mayoría de los prestatarios tienen pocas consultas de crédito en los últimos 6 meses, con una mayor frecuencia en valores bajos.

- **delinq.2yrs**: La mayoría de los prestatarios tienen pocos o ningún atraso en pagos en los últimos 2 años, lo cual sugiere un buen cumplimiento en su historial reciente.

- **pub.rec**: La mayoría de los prestatarios tienen pocos o ningún registro público de incumplimiento, lo que sugiere un historial limpio de problemas financieros.

- **not.fully.paid**: La variable objetivo muestra que la mayoría de los préstamos fueron pagados completamente (valor 0), mientras que una fracción menor no lo fue (valor 1).

En conjunto, el analisis sugiere que los prestatarios en este conjunto de datos suelen cumplir con los términos del préstamo (como se observa en "not.fully.paid" y "delinq.2yrs"), tienen un historial crediticio razonablemente bueno ("fico" y "days.with.cr.line") y generalmente utilizan su crédito disponible de manera conservadora ("revol.util").
"""

@st.cache_data
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    return None

def prepare_data_for_feature_selection(data):
    df = data.copy()
    le = LabelEncoder()
    if 'purpose' in df.columns:
        df['purpose'] = le.fit_transform(df['purpose'])
    return df

def get_top_features(X, y, k=4):  # Changed to 4 features
    X_prepared = prepare_data_for_feature_selection(X)
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_prepared, y)
    feature_names = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    return list(feature_names), feature_scores

def preprocess_data(data, selected_features=None, is_training=True):
    df = data.copy()
    le = LabelEncoder()
    if 'purpose' in df.columns:
        df['purpose'] = le.fit_transform(df['purpose'])
    
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if is_training and 'not.fully.paid' in numeric_columns:
        numeric_columns = numeric_columns.drop('not.fully.paid')
    
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    if selected_features is not None:
        if is_training:
            return df[selected_features + ['not.fully.paid']]
        else:
            return df[selected_features]
    return df

def train_model(df, selected_features):
    X = df[selected_features]
    y = df['not.fully.paid']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def display_binary_predictor(data, model, selected_features):
    st.subheader("Predictor Inmediato de Riesgo")
    st.write("Ingrese las 4 caracterismas importantes del nuevo Cliente:")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    with st.form("binary_predictor_form"):
        for i, feature in enumerate(selected_features):
            with col1 if i % 2 == 0 else col2:
                if feature == 'fico':
                    input_data[feature] = st.slider(
                        "FICO Score",
                        min_value=300,
                        max_value=850,
                        value=700,
                        help="Client's FICO credit score"
                    )
                elif feature == 'int.rate':
                    input_data[feature] = st.slider(
                        "Interest Rate (%)",
                        min_value=0.0,
                        max_value=30.0,
                        value=10.0,
                        step=0.1,
                        help="Proposed loan interest rate"
                    )
                elif feature == 'purpose':
                    input_data[feature] = st.selectbox(
                        "Loan Purpose",
                        options=sorted(data[feature].unique()),
                        help="Purpose of the loan"
                    )
                elif feature == 'log.annual.inc':
                    annual_income = st.number_input(
                        "Annual Income ($)",
                        min_value=10000,
                        max_value=1000000,
                        value=50000,
                        step=1000,
                        help="Client's annual income"
                    )
                    input_data[feature] = np.log(annual_income)
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        value=float(data[feature].mean()),
                        step=0.1
                    )
        
        submitted = st.form_submit_button("Otorgar Prestamo (Yes/No)")
    
    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df_processed = preprocess_data(input_df, selected_features, is_training=False)
        prediction = model.predict(input_df_processed)
        prediction_proba = model.predict_proba(input_df_processed)
        
        st.markdown("### Analisis y Resultado")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction[0] == 0:
                st.success("✅ NO RISK - Prestamo Recommended")
            else:
                st.error("⚠️ RISK DETECTED - Prestamo Not Recommended")
            
            st.write("#### Nivel de Confianza del modelo")
            confidence = max(prediction_proba[0]) * 100
            st.progress(confidence/100, text=f"Confidence: {confidence:.1f}%")

def main():
    uploaded_file = st.file_uploader("Sube tu CSV file", type=['csv'])
    
    data = process_uploaded_file(uploaded_file)
    st.image("parametros.jpg", width=480)  # Ajusta la ruta y el ancho según necesites
    st.write("Bienvenido a la plataforma de analisis de riesgo, se adjunta la traduccion de las caracteristicas")
    st.write("Alvaro Ponce, Agustin Almiron, Gonzalo Delfino")

    if data is not None:
        st.subheader("Datos ingresados")
        st.dataframe(data.head())
        
        X = data.drop('not.fully.paid', axis=1)
        y = data['not.fully.paid']
        selected_features, feature_scores = get_top_features(X, y)
        
        df = preprocess_data(data, selected_features, is_training=True)
        model, X_train, X_test, y_train, y_test, y_pred = train_model(df, selected_features)
        
        st.subheader("Top 4 Caracteristicas mas importantes")
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance Score': feature_scores
        }).sort_values('Importance Score', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=feature_importance, x='Importance Score', y='Feature')
        plt.title('Analisis caracteristicas importantes')
        st.pyplot(fig)
        
        st.markdown("---")
        display_binary_predictor(data, model, selected_features)  # Added new binary predictor
        
        st.markdown("---")
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        
        with col2:
            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
        st.write(analisis_texto)
        st.image("graficos.png", width=480)  # Ajusta la ruta y el ancho según necesites

if __name__ == "__main__":
    main()