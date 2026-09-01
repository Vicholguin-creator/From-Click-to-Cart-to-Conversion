# From Click to Cart to Conversion

**Predicción de intención de compra y arquetipos de abandono en e-commerce de cosmética, con una propuesta de replicabilidad metodológica para mercados latinoamericanos.**

Trabajo Fin de Máster, Máster en Data Science, Big Data & Business Analytics (UCM).
Autora: Victoria Holguín. Tutores: Santiago Mota y Carlos Ortega.

## Descripción

Sistema de tres modelos complementarios que, a partir de señales de comportamiento de sesión de un e-commerce de cosmética, predice la intención de compra, diagnostica el tipo de abandono y estima la probabilidad de recuperación del usuario. Todo ello sin depender de datos propietarios, empleando únicamente los eventos que cualquier plataforma de e-commerce registra de forma nativa.

Los datos provienen del dataset público REES46 (eCommerce Events History in Cosmetics Shop), con 20,69 millones de eventos reales de octubre de 2019 a febrero de 2020.

## Los tres modelos

- **Modelo 1, Intención de compra (XGBoost).** Predice si una sesión terminará en compra. Hallazgo central: la edición del carrito predice la compra, no el abandono.
- **Modelo 2, Arquetipos de abandono (K-Means).** Identifica cuatro tipos de abandono, cada uno con una acción de negocio distinta.
- **Modelo 3, Recuperación (RFM).** Estima la probabilidad de retorno del usuario a partir de su patrón de navegación.

Incluye una aplicación interactiva (Streamlit) que integra los modelos y permite tanto exploración individual como predicción por lotes con carga de datos externos.

## Estructura del repositorio
├── src/
│ ├── etl/ # Ingesta, sesionización y análisis exploratorio
│ ├── features/ # Construcción del dataset de modelado
│ └── models/ # Entrenamiento, interpretabilidad y optimización
├── app/ # Aplicación Streamlit (productivización)
├── notebooks/ # Análisis exploratorio
├── environment.yml # Entorno conda reproducible
└── README.md


## Reproducibilidad

El entorno se reconstruye con conda:

```bash
conda env create -f environment.yml
conda activate tfm
```

Los datos y los modelos entrenados no se incluyen en el repositorio (se regeneran ejecutando los scripts). El dataset se descarga de Kaggle y se coloca en `data/raw/`.

## Tecnologías

Python, Apache Spark (PySpark), scikit-learn, XGBoost, SHAP, Streamlit.
